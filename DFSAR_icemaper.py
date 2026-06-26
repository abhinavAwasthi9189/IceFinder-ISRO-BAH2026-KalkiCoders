#As per the data i was able to get from PRADAN website we have to use ratio at place of CPR/DOP, it's defensible substitute for CPR given what magnitude-only data allows. Since the calibrated GRI product provides detected (magnitude-only) LH/LV channels rather than complex SLC data, we approximate the polarimetric ice signature using the LH/LV intensity ratio rather than a full Stokes-derived CPR/DOP, and note this as a methodological choice driven by data availability
import rasterio as rs
from rasterio.merge import merge
import numpy as np
import os
import tempfile
import struct
import zipfile
import shutil
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.crs import CRS
import xml.etree.ElementTree as ET
from rasterio.transform import Affine

def load_band(path, search_folder=None):
    """Opens a single-band GeoTIFF and returns its data plus georeferencing.

    If the TIF has no embedded CRS,automatically finds and parses the matching XML label to assign
    the correct corner-based geolocation from the mission metadata.
    
    search_folder: directory to look for matching XML files. if None, uses the same directory as the TIF."""
    #data is stored in the first band.
    with rs.open(path) as src:
        data   = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        crs    = src.crs
        transform = src.transform
        bounds = src.bounds

    if crs is None:
        # no embedded geolocation — find and parse the XML label
        folder = search_folder or os.path.dirname(path)
        xml_path = find_xml_for_tif(path, folder)

        if xml_path:
            print(f"no crs was found in the tif file — reading corners from {os.path.basename(xml_path)}")
            corners = parse_corners_from_xml(xml_path)

            if corners:
                from rasterio.transform import from_bounds
                from rasterio.crs import CRS

                # lunar geographic CRS — sphere radius 1737400m
                # matches the reference frame used by LOLA DEM
                # use proj4 with explicit axis order to prevent GDAL from swapping
                #this all is done cause the lola psr file is in different format than the SAR files.
                lunar_crs = CRS.from_proj4(
                    "+proj=longlat +a=1737400 +b=1737400 +no_defs +axis=enu"
                )

                height, width = data.shape
                west  = min(corners["ul_lon"], corners["lr_lon"])
                east  = max(corners["ul_lon"], corners["lr_lon"])
                south = min(corners["ul_lat"], corners["lr_lat"])
                north = max(corners["ul_lat"], corners["lr_lat"])

                transform = from_bounds(west, south, east, north, width, height)
                bounds    = rs.coords.BoundingBox(west, south, east, north)
                crs       = lunar_crs

                profile.update({"crs": crs, "transform": transform})
                
                print(f"   Assigned bounds: lon[{west:.3f} to {east:.3f}] "
                      f"lat[{south:.3f} to {north:.3f}]")
            else:
                print(f"unable parse corners from XML — bounds will be pixel-based")
        else:
            print(f"No XML found for {os.path.basename(path)} — bounds will be pixel-based")

    return data, profile, crs, transform, bounds

def merge_band_files(paths, output_path):
    """ it takes a list of single-band tif paths, normalizes their spatial transforms in memory to ensure matching pixel coordinate orientations, and mosaics them"""

    datasets = []
    
    #open all paths natively
    for p in paths:
        datasets.append(rs.open(p))

    #if you are only passing 1 file, no stitching needed
    if len(datasets) == 1:
        out_profile = datasets[0].profile.copy()
        with rs.open(output_path, "w", **out_profile) as dst:
            dst.write(datasets[0].read(1), 1)
        datasets[0].close()
        return output_path

    # Track structural memory files we generate for inverted targets
    virtual_datasets = []#stores the pointer to all the datasets then to put them all in.
    normalized_datasets = []#stores final data

    try:
        for ds in datasets:
            # If transform.e is positive, it means it's an "upside-down" raster relative to rasterio's merge logic
            if ds.transform.e > 0:  # height step more than 0 
                # 🔄 Invert the height step to be negative, and shift top boundary to match
                new_e = -ds.transform.e
                new_f = ds.bounds.top  # Force origin alignment to standard north-to-south
                
                corrected_transform = Affine(
                    ds.transform.a, ds.transform.b, ds.transform.c,
                    ds.transform.d, new_e, new_f
                )

                # Using the Rasterio's MemoryFile system to rewrite properties without modifying raw files
                from rasterio.io import MemoryFile
                memfile = MemoryFile()
                
                #changes to ds profile copied to the main one and then values are added.
                profile = ds.profile.copy()
                profile.update({
                    "transform": corrected_transform,
                    "height": ds.height,
                    "width": ds.width
                })

                #write data cleanly into the corrected virtual space
                with memfile.open(**profile) as mem_ds:
                    mem_ds.write(ds.read(1), 1)
                
                # reopen the virtual memory file handle so merge can read it
                opened_mem = memfile.open()
                virtual_datasets.append(memfile)  #but keep reference alive in scope
                normalized_datasets.append(opened_mem)
            else:
                # File is already normalized, use it directly
                normalized_datasets.append(ds)

        print("...georeferenced matrices are aligned. commencing mosaic stitching...")
        mosaic, out_transform = merge(normalized_datasets)

        # build new profile based on the mosaic's unified parameters
        out_profile = datasets[0].profile.copy()
        out_profile.update({
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_transform,
        })

        #writing the fully fledge output at the output_path.
        with rs.open(output_path, "w", **out_profile) as dst:
            dst.write(mosaic)

    finally:
        # clean up of references to prevent memory leaks or file locks
        for ds in normalized_datasets:
            try:
                ds.close()
            except Exception:
                pass
        for v_ds in virtual_datasets:
            try:
                v_ds.close()
            except Exception:
                pass
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass

    return output_path

def detect_ice_candidates(lh, lv, incidence, z_thresh=2.0, incidence_bin_size=2.0):

    # instrument noise floor i got from metadata
    NES0 = 0.009749896

    # clean out bad data pixels. if LH or LV is below the noise floor, treat them as invalid (NaN)
    lh_clean = np.where(lh <= NES0, np.nan, lh)
    lv_clean = np.where(lv <= NES0, np.nan, lv)

    #ratio is our answer to not having polarisation in our data.
    ratio = lh_clean / np.where(lv_clean == 0, np.nan, lv_clean)

    ice_mask = np.zeros_like(ratio, dtype=bool)
    confidence = np.zeros_like(ratio, dtype=np.float64)

    bins = np.arange(np.nanmin(incidence), np.nanmax(incidence) + incidence_bin_size, incidence_bin_size)
    bin_indices = np.digitize(incidence, bins)
    
    #incidence data is being used cause, for all values it is not the same hence we need to see different values differently as per their incidence value
    for b in np.unique(bin_indices):
        in_bin = (bin_indices == b)
        bin_ratios = ratio[in_bin]
        valid = ~np.isnan(bin_ratios)
        if valid.sum() < 30:
            continue
        mean = np.nanmean(bin_ratios)
        std = np.nanstd(bin_ratios)
        if std == 0:
            continue
        z = np.zeros_like(bin_ratios)
        z[valid] = (bin_ratios[valid] - mean) / std
        confidence[in_bin] = z
        ice_mask[in_bin] = z > z_thresh

    valid_count = np.count_nonzero(~np.isnan(ratio))

    print(f"Python Debug: Out of {ratio.size} total pixels, {valid_count} survived the noise floor filter.")
    print(f"Python Debug: Ratio Max: {np.nanmax(ratio):.4f} | Ratio Min: {np.nanmin(ratio):.4f}") #cause the nan has been used inbetween we use nanmin and nanmax they find max and mix while ignoring NAN values.

    return ice_mask, confidence, ratio

def graph_ice(lh_path, lv_path, incidence_path, z_thresh=2.0, incidence_bin_size=2.0,
              xml_search_folder=None, original_lh_name=None):

    search_folder = xml_search_folder or os.path.dirname(lh_path)

    #use original LH filename for XML lookup — all bands share the same scene XML
    lh_lookup_path = (os.path.join(search_folder, original_lh_name)
                      if original_lh_name else lh_path)

    # load georeferencing from original file via XML lookup
    _, profile, crs, transform, bounds = load_band(lh_lookup_path, search_folder)

    # load actual pixel data from merged files
    with rs.open(lh_path) as src:
        lh = src.read(1).astype(np.float64)
    with rs.open(lv_path) as src:
        lv = src.read(1).astype(np.float64)
    with rs.open(incidence_path) as src:
        incidence = src.read(1).astype(np.float64)

    if lh.shape != lv.shape or lh.shape != incidence.shape:
        raise ValueError(f"Shape mismatch: lh={lh.shape}, lv={lv.shape}, incidence={incidence.shape}")

    ice_mask, confidence, ratio = detect_ice_candidates(lh, lv, incidence, z_thresh, incidence_bin_size)

    return {
        "ice_mask":   ice_mask,   # its a 2d array,this tells if a pixel's LH/LV ratio was statistically unusual enough, relative to other pixels at a similar incidence angle, to be flagged as a possible ice signature.
        "confidence": confidence, # it is companion array for ice_mask. A value near 0 means "perfectly typical for its incidence angle bin." A value of 3 or 4 means "way out in the statistical tail, strong candidate.
        "ratio":      ratio,      # the original LH/LV 2d array. it is what other things are cleaned out from. here mostly only for debugging if needed.
        "profile":    profile,    # it is kept for the moment we want to save one of our derived arrays (like ice_mask or confidence) back out as a real, valid, georeferenced GeoTIFF file.
        "crs":        crs,        # this defines what kind of coordinates our x/y numbers actually mean
        "transform":  transform,  # affine pixel→world coordinate translator
        "bounds":     bounds,     # bounding box in CRS units — tells if two datasets can combine
    }

def addAllnProcess(lh_paths, lv_paths, incidence_paths, z_thresh=2.0, incidence_bin_size=2.0, work_dir=None):
    if work_dir is None:
        work_dir = tempfile.mkdtemp()

    xml_search_folder = os.path.dirname(lh_paths[0])

    # also keep the original LH filename for XML lookup
    # after merging the temp file has no recognizable name
    original_lh_name  = os.path.basename(lh_paths[0])

    merged_lh_path        = merge_band_files(lh_paths,        os.path.join(work_dir, "merged_lh.tif"))
    merged_lv_path        = merge_band_files(lv_paths,        os.path.join(work_dir, "merged_lv.tif"))
    merged_incidence_path = merge_band_files(incidence_paths, os.path.join(work_dir, "merged_incidence.tif"))

    return graph_ice(
        lh_path=merged_lh_path,
        lv_path=merged_lv_path,
        incidence_path=merged_incidence_path,
        z_thresh=z_thresh,
        incidence_bin_size=incidence_bin_size,
        xml_search_folder=xml_search_folder,
        original_lh_name=original_lh_name,  # use this for XML lookup, not merged filename
    )

def extract_zip_payloads(zip_root_folder, extraction_target_dir):
    """scans zip_root_folder for PRADAN zip bundles, looks inside their internal nested directories, and extracts ONLY the ground-range .tif files  and their matching .
    xml label files directly into extraction_target_dir.XML files carry the corner coordinate metadata needed for geolocation."""
    if not os.path.exists(zip_root_folder):
        print(f"Source folder '{zip_root_folder}' not found. creating it...")
        os.makedirs(zip_root_folder)
        return

    print("**Scanning for zipped PRADAN scene bundles**")
    zip_files_found = [f for f in os.listdir(zip_root_folder) if f.lower().endswith('.zip')]
    
    if not zip_files_found:
        print("No zip files found in directory. Proceeding with existing extracted files.")
        return

    for zip_name in zip_files_found:
        zip_path = os.path.join(zip_root_folder, zip_name)
        print(f"++Extracting target tracks from: {zip_name}++")
        
        with zipfile.ZipFile(zip_path, 'r') as archive:
            for member in archive.namelist():
                # extract .tif files from "calibrated" data folder — same as before
                if "data/calibrated" in member and member.lower().endswith(".tif"):
                    filename = os.path.basename(member)
                    if filename:
                        source_stream = archive.open(member)
                        target_path = os.path.join(extraction_target_dir, filename)
                        with open(target_path, "wb") as target_file:
                            shutil.copyfileobj(source_stream, target_file)
                        print(f"   ==> Extracted TIF: {filename}")

                # also extract .xml label files from the same "calibrated "folder
                # they carry the corner coordinates for geolocation
                elif "data/calibrated" in member and member.lower().endswith(".xml"):
                    filename = os.path.basename(member)
                    if filename:
                        source_stream = archive.open(member)
                        target_path = os.path.join(extraction_target_dir, filename)
                        with open(target_path, "wb") as target_file:
                            shutil.copyfileobj(source_stream, target_file)
                        print(f"   ==> Extracted XML: {filename}")

def parse_corners_from_xml(xml_path):
    """ This reads a DFSAR PDS4 XML label file and extracts the four corner coordinates from the Geometry_Parameters block.
    Returns a dict with ul_lat, ul_lon, lr_lat, lr_lon, or none if the file cannot be parsed or corners are missing."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # PDS4 XML uses namespaces as shown — defining them so we can search by tag
        ns = {
            "isda": "http://pds.nasa.gov/pds4/isda/v1",
            "pds":  "http://pds.nasa.gov/pds4/pds/v1",
        }

        # navigate to Geometry_Parameters block
        geo = root.find(".//isda:Geometry_Parameters", ns)
        if geo is None:
            print(f"   ** No Geometry_Parameters found in {xml_path}**")
            return None

        def get_float(tag):
            el = geo.find(f"isda:{tag}", ns)
            if el is not None:
                return float(el.text.strip())
            return None

        ul_lat = get_float("upper_left_latitude")
        ul_lon = get_float("upper_left_longitude")
        lr_lat = get_float("lower_right_latitude")
        lr_lon = get_float("lower_right_longitude")

        if None in (ul_lat, ul_lon, lr_lat, lr_lon):
            print(f"   **Missing corner values in {xml_path}**")
            return None

        return {
            "ul_lat": ul_lat,
            "ul_lon": ul_lon,
            "lr_lat": lr_lat,
            "lr_lon": lr_lon,
        }

    except ET.ParseError as e:
        print(f"   **XML parse error in {xml_path}: {e}**")
        return None
    
def find_xml_for_tif(tif_path, search_folder):
    """
    Finds the XML label file that corresponds to a given TIF file.
    
    ISRO's naming convention: TIF and XML share the same base filename,
    differing only in extension. For example:
        ch2_sar_ncxs_..._gri_xx_cp_lh_d18.tif
        ch2_sar_ncxs_..._gri_xx_cp_lh_d18.xml  ← matching label
    
    The scene-level XML (without lh/lv/in suffix) also works since it contains the same corner coordinates for the whole scene. we try both the exact match and the scene-level XML"""
    tif_name = os.path.basename(tif_path)
    base_name = os.path.splitext(tif_name)[0]

    # try exact match first: same filename, .xml extension
    exact_xml = os.path.join(search_folder, base_name + ".xml")
    if os.path.exists(exact_xml):
        return exact_xml

    # try scene-level XML: strip the polarization suffix (lh/lv/in)
    # ch2_sar_ncxs_..._gri_xx_cp_lh_d18 → ch2_sar_ncxs_..._gri_xx_cp_xx_d18
    # the scene XML replaces the polarization segment with 'xx'

    scene_xml_name = (base_name
        .replace("_cp_lh_", "_cp_xx_")
        .replace("_cp_lv_", "_cp_xx_")
        .replace("_in_cp_", "_cp_xx_")  # incidence file variant
    ) + ".xml"
    scene_xml = os.path.join(search_folder, scene_xml_name)
    if os.path.exists(scene_xml):
        return scene_xml

    return None

def compute_sigma_naught_cpr(sri_lh_path, sri_lv_path, search_folder=None):
    """Computes the Circular Polarization Ratio directly from sigma-naught radiometric images.
    CPR = S₀_LH / S₀_LV 
    """
    search = search_folder or os.path.dirname(sri_lh_path)

    with rs.open(sri_lh_path) as src:
        sigma_lh = src.read(1).astype(np.float64)

    with rs.open(sri_lv_path) as src:
        sigma_lv = src.read(1).astype(np.float64)

    # SRI values might be in dB — so check XML label
    # if values are negative and small (e.g. -15 to -5), they're in dB
    # convert to linear: S₀_linear = 10^(S₀_dB / 10)
    if np.nanmean(sigma_lh) < 0:
        sigma_lh = np.power(10.0, sigma_lh / 10.0)
        sigma_lv = np.power(10.0, sigma_lv / 10.0)
    else:
        pass

    # CPR = same-sense / opposite-sense circular
    # for hybrid-pol (transmit circular, receive LH/LV):
    # SC (same-sense) = S₀_LH & OC (opposite-sense) = S₀_LV [approx.]
    cpr = sigma_lh / np.where(sigma_lv == 0, np.nan, sigma_lv)

    valid = ~np.isnan(cpr) & ~np.isinf(cpr)
    print(f"   CPR range: {np.nanmin(cpr):.3f} to {np.nanmax(cpr):.3f}")
    print(f"   pixels with CPR > 1.0: {(cpr[valid] > 1.0).sum():,}")
    print(f"   pixels with CPR > 1.5: {(cpr[valid] > 1.5).sum():,}")

    return cpr, sigma_lh, sigma_lv

def sigma_naught_to_ice_fraction(sigma_lh, sigma_lv, incidence_deg):
    """Converts absolute S₀ values directly to ice fraction via Maxwell Garnett effective medium theory.
    No z-score proxy — uses physical backscatter values directly.

    Steps:
        1. Compute total backscatter power S₀_total = S₀_LH + S₀_LV
        2. Convert to Fresnel reflectance R using incidence angle correction
        3. Invert Fresnel to get ε_eff
        4. Apply Maxwell Garnett to get f_ice
                                                    >>>Reference: Fa & Wieczorek 2012, JGR Planets"""
    EPS_HOST = 3.0
    EPS_ICE  = 3.15

    incidence_rad = np.radians(incidence_deg)

    # total backscatter power
    sigma_total = sigma_lh + sigma_lv

    # convert S₀ to Fresnel R using surface scattering model
    # S₀ = R × cos(T) for simple specular scattering
    # R = S₀ / cos(T)
    cos_theta = np.cos(incidence_rad)
    R = sigma_total / np.where(cos_theta < 0.01, np.nan, cos_theta)

    # physical cap: R cannot exceed 1.0 (100% reflection)
    R = np.clip(R, 0.0, 0.999)

    # invert Fresnel: ε_eff = ((1 + √R) / (1 - √R))²
    sqrtR   = np.sqrt(R)
    eps_eff = ((1.0 + sqrtR) / (1.0 - sqrtR)) ** 2

    # Maxwell Garnett inversion
    numerator   = (eps_eff - EPS_HOST) * (EPS_ICE + 2.0 * EPS_HOST)
    denominator = (EPS_ICE - EPS_HOST) * (eps_eff + 2.0 * EPS_HOST)

    f_ice = np.where(
        np.abs(denominator) > 1e-10,
        numerator / denominator,
        0.0
    )

    # physical bounds: 0% to 50%
    f_ice = np.clip(f_ice, 0.0, 0.50)

    valid = ~np.isnan(f_ice)
    print(f"average of ice fraction:  {f_ice[valid].mean():.4f}")

    return f_ice

def discover_scene_files(root_folder):
    lh_paths        = []
    lv_paths        = []
    incidence_paths = []
    sri_lh_paths    = []   # sigma-naught LH
    sri_lv_paths    = []   # sigma-naught LV
    sri_ma_paths    = []   # sigma-naught magnitude (combined)
    skipped         = []
    unrecognized    = []

    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if not filename.lower().endswith(".tif"):
                continue

            full_path = os.path.join(dirpath, filename)
            name = filename.lower()

            if "fp_" in name:
                unrecognized.append(full_path)
                continue

            #not used values
            if "sli" in name:
                skipped.append(full_path)
                continue

            if "gri_in" in name:
                incidence_paths.append(full_path)
            elif "gri" in name and "cp_lh" in name:
                lh_paths.append(full_path)
            elif "gri" in name and "cp_lv" in name:
                lv_paths.append(full_path)

            # SRI products — sigma-naught radiometric images
            # these give absolute backscatter S₀ per pixel
            elif "sri" in name and "cp_lh" in name:
                sri_lh_paths.append(full_path)
            elif "sri" in name and "cp_lv" in name:
                sri_lv_paths.append(full_path)
            elif "sri_ma" in name:
                sri_ma_paths.append(full_path)
            else:
                unrecognized.append(full_path)

    print(f"Found {len(lh_paths)} GRI-LH file(s)")
    print(f"Found {len(lv_paths)} GRI-LV file(s)")
    print(f"Found {len(incidence_paths)} incidence file(s)")
    print(f"Found {len(sri_lh_paths)} SRI-LH sigma-naught file(s)")
    print(f"Found {len(sri_lv_paths)} SRI-LV sigma-naught file(s)")
    print(f"Found {len(sri_ma_paths)} SRI-MA combined sigma-naught file(s)")

    if not (len(lh_paths) == len(lv_paths) == len(incidence_paths)):
        raise ValueError(
            f"Mismatched GRI file counts — lh:{len(lh_paths)}, "
            f"lv:{len(lv_paths)}, incidence:{len(incidence_paths)}"
        )

    return lh_paths, lv_paths, incidence_paths, sri_lh_paths, sri_lv_paths, sri_ma_paths


def export_for_cpp(result, output_bin_path, target_crs_wkt=None):
    """Saves the ice detection results into a binary file.Prioritizes physical Ice Fraction or CPR over statistical z-score"""
    # 1. Determine the best data layer to export
    if "f_ice_mg" in result:
        export_data = result["f_ice_mg"].astype(np.float32)
    elif "cpr" in result:
        export_data = result["cpr"].astype(np.float32)
    else:
        export_data = result["confidence"].astype(np.float32)

    crs        = result["crs"]
    transform  = result["transform"]
    bounds     = result["bounds"]
    height, width = export_data.shape

    if target_crs_wkt is not None and crs is not None:
        moon_radius = 1737400.0
        deg_to_m    = np.pi / 180.0 * moon_radius

        correct_left   = bounds.left   * deg_to_m
        correct_right  = bounds.right  * deg_to_m
        correct_bottom = bounds.bottom * deg_to_m
        correct_top    = bounds.top    * deg_to_m

        bounds = rs.coords.BoundingBox(correct_left, correct_bottom,
                                       correct_right, correct_top)

        print(f"   Converted bounds to equirectangular meters:")
        print(f"   X (Lon): [{correct_left:.1f} to {correct_right:.1f}]")
        print(f"   Y (Lat): [{correct_bottom:.1f} to {correct_top:.1f}]")
        
    # ---SERIALIZATION DATA EXPORT---
    with open(output_bin_path, "wb") as f:
        f.write(struct.pack("II", width, height))
        f.write(struct.pack("dddd", bounds.left, bounds.right, bounds.bottom, bounds.top))
        
        # stream the selected high-quality data array
        f.write(export_data.tobytes())
        
    print(f"---Successfully exported synchronized binary layout to {output_bin_path}---\n")


'''this is where the functioning code starts, before this is all the function being used in the following code'''

ZIP_SOURCE_DIR = r"./data/DFSAR"
PROCESSING_DIR = r"./data/DFSAR"

#so now we just place the zip files inside DFSAR inside of data. and we get clean data without any hassle.
extract_zip_payloads(ZIP_SOURCE_DIR, PROCESSING_DIR)

#it looks for the lh,lv and icidence file in the folder named data. this makes it much easier to enter the data
lh_paths, lv_paths, incidence_paths, sri_lh_paths, sri_lv_paths, sri_ma_paths = discover_scene_files(PROCESSING_DIR)

print("---process all the data---")
result = addAllnProcess(
    lh_paths,
    lv_paths,
    incidence_paths,
)

print(f"\nMerged DFSAR scene bounds:")
print(f"  lon: [{result['bounds'].left:.3f}° to {result['bounds'].right:.3f}°]")
print(f"  lat: [{result['bounds'].bottom:.3f}° to {result['bounds'].top:.3f}°]")

# if SRI files available, compute direct S₀-based CPR and ice fraction
if sri_lh_paths and sri_lv_paths:
    print("\nSRI sigma-naught files found — computing direct CPR...")
    cpr, sigma_lh, sigma_lv = compute_sigma_naught_cpr(
        sri_lh_paths[0], sri_lv_paths[0],
        search_folder=PROCESSING_DIR
    )

    # load incidence angle
    with rs.open(incidence_paths[0]) as src:
        incidence = src.read(1).astype(np.float64)

    if cpr.shape == incidence.shape:
        print("\nComputing Maxwell Garnett ice fractions from S₀...")
        f_ice = sigma_naught_to_ice_fraction(sigma_lh, sigma_lv, incidence)

        # ice mask from proper CPR threshold (no z-score needed)
        ice_mask_direct = (cpr > 1.0) & (~np.isnan(cpr))
        print(f"Direct CPR ice candidates (CPR>1.0): {ice_mask_direct.sum():,}")

        # add to result for export
        result["cpr"]            = cpr
        result["f_ice_mg"]       = f_ice
        result["ice_mask_cpr"]   = ice_mask_direct
    else:
        print(f"Shape mismatch: CPR={cpr.shape} incidence={incidence.shape}")
        print("SRI and GRI have different dimensions — using z-score pipeline only")


#IMPORTANT
# LOLA DEM uses simple cylindrical (equirectangular) projection
# centered on 0° lat, 0° lon, Moon sphere radius 1737400m
# this should match exactly — all coordinate fusion depends on it

PSR_CRS_WKT = "+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +x_0=0 +y_0=0 +a=1737400 +b=1737400 +units=m +no_defs +axis=enu"

cpp_output_destination = "./lunar_grid_weights.bin"

# this is the binary file in which the data will be stored to be sent for computation on C++[faster work]
# eexport the data
export_for_cpp(
    result,
    output_bin_path=cpp_output_destination,
    target_crs_wkt=PSR_CRS_WKT,
)


print("")
print("---[code processed without any issues]---")