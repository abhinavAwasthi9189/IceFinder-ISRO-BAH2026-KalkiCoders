import numpy as np
import rasterio as rs
import struct
import os
from scipy import ndimage
from scipy.ndimage import label, binary_erosion, binary_dilation, zoom


MAX_SUN_ELEV_RAD = np.radians(1.54)  # moon's axial tilt

def load_dem(dem_path):
    with rs.open(dem_path) as src:
        transform = src.transform
        crs = src.crs
        global_height = src.height
        global_width = src.width

        # crop to the bottom 1/18th of the map (-80 to -90 degrees)
        start_row = int(global_height * 17 / 18)
        height_window = global_height - start_row
        window = rs.windows.Window(0, start_row, global_width, height_window)

        #read ONLY the cropped data into memory
        dem = src.read(1, window=window).astype(np.float32)
        
        #Update the affine transform matrix for the new crop window
        cropped_transform = rs.windows.transform(window, transform)
        
        # Recalculate the true geospatial bounding box for just this crop
        cropped_bounds = rs.windows.bounds(window, transform)

        # Calculate exact pixel scaling factor
        pixel_size_x = abs(cropped_transform.a)
        pixel_size_y = abs(cropped_transform.e)
        pixel_size = (pixel_size_x + pixel_size_y) / 2.0
        
    return dem, pixel_size, cropped_transform, crs, cropped_bounds

def compute_horizon_mask_one_azimuth(dem, pixel_size, dr, dc, max_steps=500):
    """For one azimuth direction (dr, dc), computes for every pixel whether the terrain horizon in that direction exceeds maximum possible Sun elevation angle.

    Returns a 2D boolean array:
        True  = terrain blocks Sun from this direction (shadowed from this az) | False = Sun CAN reach this pixel from this direction"""
    rows, cols = dem.shape

    # max horizon angle seen so far, for every pixel simultaneously
    # initialized to a very negative number
    max_horizon = np.full((rows, cols), -np.inf, dtype=np.float32)  # float32 not float64 — halves memory

    # integer step increments for array indexing
    idr = int(round(dr))
    idc = int(round(dc))

    for step in range(1, max_steps + 1):
        # compute the valid slice bounds — no wrap-around, no copy
        r_src_start = max(0, -step * idr)
        r_src_end   = min(rows, rows - step * idr)
        c_src_start = max(0, -step * idc)
        c_src_end   = min(cols, cols - step * idc)

        r_dst_start = max(0, step * idr)
        r_dst_end   = min(rows, rows + step * idr)
        c_dst_start = max(0, step * idc)
        c_dst_end   = min(cols, cols + step * idc)

        # make sure slices are same shape
        h = min(r_src_end - r_src_start, r_dst_end - r_dst_start)
        w = min(c_src_end - c_src_start, c_dst_end - c_dst_start)

        if h <= 0 or w <= 0:
            break

        src_patch = dem[r_src_start:r_src_start+h, c_src_start:c_src_start+w]
        dst_patch = dem[r_dst_start:r_dst_start+h, c_dst_start:c_dst_start+w]

        # distance in meters from each pixel to this step's terrain point
        ground_dist = step * pixel_size

        # angle from each pixel up to the terrain point at this step
        height_diff = src_patch - dst_patch
        angle = np.arctan2(height_diff, ground_dist)

        # update max horizon — keep the steepest angle seen so far
        np.maximum(
            max_horizon[r_dst_start:r_dst_start+h, c_dst_start:c_dst_start+w],
            angle,
            out=max_horizon[r_dst_start:r_dst_start+h, c_dst_start:c_dst_start+w]
        )

    # pixels where max horizon exceeds sun's max elevation are shadowed
    # from this azimuth — sun cannot reach them from this direction
    return max_horizon > MAX_SUN_ELEV_RAD

def compute_sky_view_factor(dem, pixel_size, azimuth_step_deg=45, max_steps=150):
    """For every pixel, computes the sky-view factor — what fraction of the upper hemisphere is visible (not blocked by surrounding terrain).
    A value near 1.0 means open terrain, lots of sky visible.
    A value near 0.0 means deeply enclosed — almost no sky visible.
    
    Pixels with low sky-view factor receive minimal scattered light even if they're already PSRs — these are doubly shadowed candidates.
    
    Reuses the same horizon angle logic as the PSR mask, so no extra DEM processing needed — just accumulate the horizon angles differently"""
    rows, cols = dem.shape
    azimuths = np.arange(0, 360, azimuth_step_deg)
    n_az = len(azimuths)
    
    # sum of cos(horizon_angle) across all azimuths per pixel
    svf_sum = np.zeros((rows, cols), dtype=np.float32)

    for az_deg in azimuths:
        az_rad = np.radians(az_deg)
        dr = -np.cos(az_rad)
        dc = np.sin(az_rad)

        idr = int(round(dr))
        idc = int(round(dc))

        max_horizon = np.full((rows, cols), 0.0, dtype=np.float32)

        for step in range(1, max_steps + 1):
            r_src_start = max(0, -step * idr)
            r_src_end   = min(rows, rows - step * idr)
            c_src_start = max(0, -step * idc)
            c_src_end   = min(cols, cols - step * idc)

            r_dst_start = max(0, step * idr)
            r_dst_end   = min(rows, rows + step * idr)
            c_dst_start = max(0, step * idc)
            c_dst_end   = min(cols, cols + step * idc)

            h = min(r_src_end - r_src_start, r_dst_end - r_dst_start)
            w = min(c_src_end - c_src_start, c_dst_end - c_dst_start)

            if h <= 0 or w <= 0:
                break

            src_patch = dem[r_src_start:r_src_start+h, c_src_start:c_src_start+w]
            dst_patch = dem[r_dst_start:r_dst_start+h, c_dst_start:c_dst_start+w]

            ground_dist = step * pixel_size
            height_diff = src_patch - dst_patch
            angle = np.arctan2(height_diff, ground_dist).astype(np.float32)

            np.maximum(
                max_horizon[r_dst_start:r_dst_start+h, c_dst_start:c_dst_start+w],
                angle,
                out=max_horizon[r_dst_start:r_dst_start+h, c_dst_start:c_dst_start+w]
            )

        horizon_clipped = np.maximum(max_horizon, 0.0)
        svf_sum += np.cos(horizon_clipped)

    # normalize by number of azimuths to get fraction 0.0-1.0
    sky_view_factor = svf_sum / n_az
    return sky_view_factor

def compute_psr_mask(dem, pixel_size, azimuth_step_deg=5, max_steps=500):
    """Computes the full PSR mask by checking all azimuth directions.A pixel is a PSR only if it is shadowed from EVERY azimuth direction.
        start assuming everything is PSR (all True),then for each azimuth, any pixel the Sun CAN reach gets cleared to False. After all azimuths, remaining True pixels are genuine PSRs."""
    rows, cols = dem.shape
    # assuming everything is permanently shadowed
    psr_mask = np.ones((rows, cols), dtype=bool)

    azimuths = np.arange(0, 360, azimuth_step_deg)
    total = len(azimuths)

    for i, az_deg in enumerate(azimuths):
        az_rad = np.radians(az_deg)
        # convert azimuth to grid direction
        # azimuth 0=north=-row, 90=east=+col
        dr = -np.cos(az_rad)
        dc = np.sin(az_rad)

        shadowed_from_this_az = compute_horizon_mask_one_azimuth(
            dem, pixel_size, dr, dc, max_steps
        )

        psr_mask &= shadowed_from_this_az

        print(f"Azimuth {az_deg:6.1f}° ({i+1}/{total}) — "
              f"PSR candidates remaining: {psr_mask.sum():,}")

        # early exit: if no PSR candidates remain, stop
        if psr_mask.sum() == 0:
            print("No PSR pixels remain — stopping early.")
            break

    return psr_mask

def apply_edge_mask(psr_mask, max_steps):
    """Zeros out pixels within max_steps of any DEM edge,since np.roll wraps terrain artificially at boundaries,producing fake shadow results there"""
    border = max_steps
    cleaned = psr_mask.copy()
    cleaned[:border, :]  = False   # top edge
    cleaned[-border:, :] = False   # bottom edge
    cleaned[:, :border]  = False   # left edge
    cleaned[:, -border:] = False   # right edge
    return cleaned

def get_depth_range(radius_m):
    """returns (min_depth, max_depth) in meters for a crater of given radius, based on empirical crater scaling laws for degraded polar craters. derived from Pike 1977 scaling, adjusted for lunar south polar degradation.
    
    We use degraded depth minimums since PSR craters are ancient and eroded.
    We use fresh depth maximums since some craters may be relatively young"""
    diameter_m = radius_m * 2

    if diameter_m <= 1500:          # scale 1: ~1km diameter, simple bowl
        d_min, d_max = 150, 220
    elif diameter_m <= 3500:        # scale 2: ~2.4km diameter, simple bowl
        d_min, d_max = 350, 520
    elif diameter_m <= 10000:       # scale 3: ~6km diameter, large simple bowl
        d_min, d_max = 850, 1300
    elif diameter_m <= 25000:       # scale 4: ~15km diameter, simple-complex transition
        d_min, d_max = 1800, 3100
    elif diameter_m <= 60000:       # scale 5: ~40km diameter, complex basin flat floor
        d_min, d_max = 2400, 3600
    else:                           # scale 6: ~100km diameter, giant complex basin
        d_min, d_max = 3000, 4500

    # DEM resolution tolerance:
    # scale 1 (1km diameter): 118m pixels means ~8 pixels across crater
    # deepest point likely captured — use 0.75 tolerance not 0.6
    # scale 2+ : well sampled, use 0.85
    dem_tolerance = 0.75 if diameter_m <= 1500 else 0.85
    return d_min * dem_tolerance, d_max

def find_craters_in_dem(dem, pixel_size, min_radius_m=500, max_radius_m=25000):
    """Highly optimized window-pass crater extractor designed to bypass memory locks and array-traps on large global scale raster chunks"""
    rows, cols = dem.shape
    min_radius_px = max(3, int(min_radius_m / pixel_size))
    max_radius_px = max(min_radius_px + 2, int(max_radius_m / pixel_size))

    # edge border must be at least as large as the largest crater radius
    # we scan for — otherwise minimum_filter boundary padding creates
    # artificial local minima near the array edge
    edge_border = max(max_radius_px + 10, 50)

    scale_radii = np.unique(np.geomspace(
        min_radius_px, max_radius_px, num=6, dtype=int
    ))

    print("...Scanning terrain grid blocks for regional structural minima...")
    print(f">>>Scanning at radius scales: {scale_radii} pixels")
    print(f"   = {[round(r * pixel_size / 1000, 1) for r in scale_radii]} km")

    all_raw = []

    for radius_px in scale_radii:
        if radius_px < 3:
            continue

        local_min = ndimage.minimum_filter(dem, size=(radius_px * 2, radius_px * 2))
        is_local_min = (dem == local_min)

        y_indices, x_indices = np.where(is_local_min)
        scale_count = 0

        for r, c in zip(y_indices, x_indices):
            # strict boundary check including edge_border exclusion zone
            if (r < edge_border or r >= rows - edge_border or
                c < edge_border or c >= cols - edge_border or
                r - radius_px < 0 or r + radius_px >= rows or
                c - radius_px < 0 or c + radius_px >= cols):
                continue

            floor_elev = float(dem[r, c])

            rim_samples = [
                dem[r - radius_px, c],
                dem[r + radius_px, c],
                dem[r, c - radius_px],
                dem[r, c + radius_px],
            ]
            rim_elev = float(np.mean(rim_samples))
            depth = rim_elev - floor_elev

            # use empirically grounded depth range from crater scaling laws
            radius_m = radius_px * pixel_size
            min_depth, max_depth = get_depth_range(radius_m)

            if min_depth < depth < max_depth:
                all_raw.append({
                    "center_row": int(r),
                    "center_col": int(c),
                    "floor_elev": floor_elev,
                    "rim_elev": rim_elev,
                    "radius_pixels": int(radius_px),
                    "radius_meters": radius_m,
                    "depth_meters": depth,
                })
                scale_count += 1

        print(f"   Scale {radius_px}px ({radius_px*pixel_size/1000:.1f}km): "
              f"{scale_count} raw candidates")

    print(f"...Raw intersections captured. Filtering duplicates from {len(all_raw)} discoveries...")
    unique_craters = []
    seen_zones = set()

    all_raw.sort(key=lambda x: x["radius_pixels"], reverse=True)

    for crater in all_raw:
        rp = crater["radius_pixels"]
        coord_key = (
            crater["center_row"] // max(1, rp),
            crater["center_col"] // max(1, rp),
            rp
        )
        if coord_key not in seen_zones:
            seen_zones.add(coord_key)
            unique_craters.append(crater)

    small  = [c for c in unique_craters if c["radius_pixels"] <= 5]
    medium = [c for c in unique_craters if 5 < c["radius_pixels"] <= 20]
    large  = [c for c in unique_craters if c["radius_pixels"] > 20]
    print(f"++Found {len(unique_craters)} verified distinct crater candidates++")
    print(f"   By size — small(<=5px): {len(small)}, "
          f"medium(5-20px): {len(medium)}, large(>20px): {len(large)}")

    return unique_craters

def find_doubly_shadowed_craters(dem, psr_mask, svf, pixel_size, min_psr_fraction=0.7):
    '''Highly optimized nested checking engine using local sub-grid windows'''
    all_craters = find_craters_in_dem(dem, pixel_size)
    rows, cols = dem.shape

    print("...Filtering craters by floor PSR fraction (Windowed Mode)...")
    psr_craters = []

    for crater in all_craters:
        cr = crater["center_row"]
        cc = crater["center_col"]
        rp = crater["radius_pixels"]

        # Only look at immediate neighborhood rows/cols
        r_min, r_max = max(0, cr - rp), min(rows, cr + rp + 1)
        c_min, c_max = max(0, cc - rp), min(cols, cc + rp + 1)

        # Build tiny localized grid meshes
        Y, X = np.ogrid[r_min:r_max, c_min:c_max]
        local_floor_mask = ((Y - cr)**2 + (X - cc)**2) <= rp**2

        local_psr = psr_mask[r_min:r_max, c_min:c_max]

        floor_pixels = local_floor_mask.sum()
        if floor_pixels == 0:
            continue

        psr_pixels_in_floor = (local_psr & local_floor_mask).sum()
        psr_fraction = psr_pixels_in_floor / floor_pixels

        '''large craters need less PSR coverage to qualify because a large outer crater only needs to CONTAIN PSR regions, not be entirely shadowed itself.
         Small inner craters need high PSR fraction since they are the actual ice targets.
        TIGHTENED TIERED THRESHOLD after fixing crater quality upstream'''
        if rp <= 5:
            threshold = min_psr_fraction        # small inner crater: strict 0.70
        elif rp <= 20:
            threshold = min_psr_fraction * 0.4  # medium: 0.28
        else:
            threshold = min_psr_fraction * 0.15 # large outer: 0.105

        # compute mean sky-view factor over crater floor
        local_svf = svf[r_min:r_max, c_min:c_max]
        mean_svf = float(np.mean(local_svf[local_floor_mask]))
        
        if psr_fraction >= threshold:
            crater["psr_fraction"] = psr_fraction
            crater["psr_pixel_count"] = int(psr_pixels_in_floor)
            crater["mean_svf"] = mean_svf  # lower = more enclosed = less scattered light
            psr_craters.append(crater)

    # print breakdown by size tier so we can verify all tiers are populated
    small  = [c for c in psr_craters if c["radius_pixels"] <= 5]
    medium = [c for c in psr_craters if 5 < c["radius_pixels"] <= 20]
    large  = [c for c in psr_craters if c["radius_pixels"] > 20]
    print(f"PSR craters by tier — small(<=5px): {len(small)}, "
          f"medium(5-20px): {len(medium)}, large(>20px): {len(large)}")

    # Fast proximity check for nesting
    print("...Computing double-shadowing nesting intersections...")
    doubly_shadowed = []

    psr_craters.sort(key=lambda x: x["radius_pixels"])

    for i, inner in enumerate(psr_craters):
        
        # minimum inner crater size for doubly shadowed classification:
        # it must be at least 1km radius to be rover-traversable and large enough to have meaningful ice volume
        if inner["radius_meters"] < 1000:
            continue
            
        for outer in psr_craters[i+1:]:

            # inner must be less than 30% the radius of outer
            if inner["radius_pixels"] >= outer["radius_pixels"] * 0.3:
                continue

            if outer["radius_pixels"] - inner["radius_pixels"] < 3:
                continue

            dist = np.sqrt((inner["center_row"] - outer["center_row"])**2 +
                           (inner["center_col"] - outer["center_col"])**2)

            if dist < outer["radius_pixels"]:
                entry = inner.copy()
                entry["outer_crater"] = outer
                entry["nesting_distance_px"] = float(dist)
                entry["is_doubly_shadowed"] = True
                doubly_shadowed.append(entry)
                break

    # sort by sky-view factor ascending — lowest SVF = most enclosed = least indirect illumination = best ice preservation conditions
    doubly_shadowed.sort(key=lambda x: x.get("mean_svf", 1.0))

    print(f"Found {len(doubly_shadowed)} doubly shadowed crater candidates")
    for ds in doubly_shadowed[:5]:
        print(f"   Center: ({ds['center_row']}, {ds['center_col']}) | "
              f"PSR: {ds['psr_fraction']*100:.1f}% | "
              f"SVF: {ds.get('mean_svf', -1):.3f} | "
              f"Depth: {ds['depth_meters']:.0f}m | "
              f"Radius: {ds['radius_meters']/1000:.1f}km")

    return doubly_shadowed

def run_psr_analysis(dem_path, azimuth_step_deg=5, max_steps=500):
    """Full PSR + doubly shadowed crater pipeline.Give it the LOLA DEM path, get back everything"""

    print("...Loading DEM...")
    dem, pixel_size, transform, crs, bounds = load_dem(dem_path)
    print(f"LOLA CRS: {crs}")
    print(f"LOLA bounds: {bounds}")

    # NO zoom/downsampling — slicing fix lets it run at full 118m resolution
    print(f"DEM shape: {dem.shape}, pixel size: {pixel_size:.1f}m")
    print(f"Elevation range: {dem.min():.0f}m to {dem.max():.0f}m")

    print("\nComputing PSR mask...")
    psr_mask = compute_psr_mask(dem, pixel_size, azimuth_step_deg, max_steps)
    # NO apply_edge_mask needed — slicing handles boundaries correctly
    print(f"PSR pixels: {psr_mask.sum():,} of {psr_mask.size:,} "
          f"({100*psr_mask.mean():.1f}%)")

    print("\nComputing sky-view factor (scattered light proxy)...")
    svf = compute_sky_view_factor(dem, pixel_size, azimuth_step_deg, max_steps)
    print(f"SVF range: {svf.min():.3f} to {svf.max():.3f}")
    print(f"Deeply enclosed pixels (SVF<0.1): {(svf < 0.1).sum():,}")
    
    print("\nFinding doubly shadowed craters...")
    doubly_shadowed = find_doubly_shadowed_craters(dem, psr_mask, svf, pixel_size)

    return {
        "dem": dem,
        "psr_mask": psr_mask,
        "sky_view_factor": svf,   
        "doubly_shadowed_craters": doubly_shadowed,
        "pixel_size": pixel_size,
        "transform": transform,
        "crs": crs,
        "bounds": bounds,
    }

def export_psr_and_dem_for_cpp(psr_result, output_bin_path):
    """Saves computed PSR masks and raw DEM elevations into a multi-channel binary matrix file optimized for high-performance C++ resampling nodes.

    Layout Format:
    +Bytes 0-3   : Width (uint32_t)
    +Bytes 4-7   : Height (uint32_t)
    +Bytes 8-15  : minX / left   (double)
    +Bytes 16-23 : maxX / right  (double)
    +Bytes 24-31 : minY / bottom (double)
    +Bytes 32-39 : maxY / top    (double)
    +Remaining   : Alternating pairs of [PSR_Weight, Elevation] as float32"""
    dem       = psr_result["dem"]
    psr_mask  = psr_result["psr_mask"]
    doubly    = psr_result["doubly_shadowed_craters"]
    transform = psr_result["transform"]

    height, width = dem.shape

    # compute bounds directly from the affine transform
    # transform * (col, row) gives (world_x, world_y)
    left,  top    = transform * (0,     0)   
    right, bottom = transform * (width, height)  

    print(f"   Bounds from transform:")
    print(f"   X (lon meters): [{left:.1f} to {right:.1f}]")
    print(f"   Y (lat meters): [{bottom:.1f} to {top:.1f}]")
    print(f"   In degrees — lat: [{bottom/30212:.2f}° to {top/30212:.2f}°]")

    psr_weights = np.ones((height, width), dtype=np.float32)
    psr_weights[psr_mask] = 0.5  # PSR pixels

    # mark doubly shadowed crater floors
    for crater in doubly:
        cr = crater["center_row"]
        cc = crater["center_col"]
        rp = crater["radius_pixels"]
        Y, X = np.ogrid[:height, :width]
        floor_mask = ((Y - cr)**2 + (X - cc)**2) <= rp**2
        psr_weights[floor_mask & psr_mask] = 0.1

    print(f"   Normal terrain pixels (1.0): {(psr_weights == 1.0).sum():,}")
    print(f"   PSR pixels (0.5):            {(np.abs(psr_weights - 0.5) < 0.01).sum():,}")
    print(f"   Doubly shadowed pixels (0.1):{(np.abs(psr_weights - 0.1) < 0.01).sum():,}")

    # interleave [weight, elevation] pairs — vectorized
    interleaved = np.empty((height, width, 2), dtype=np.float32)
    interleaved[..., 0] = psr_weights
    interleaved[..., 1] = dem.astype(np.float32)

    with open(output_bin_path, "wb") as f:
        # header: width, height, minX, maxX, minY, maxY
        f.write(struct.pack("IIdddd", width, height, left, right, bottom, top))
        f.write(interleaved.tobytes())

    file_size_mb = os.path.getsize(output_bin_path) / 1e6
    print(f"PSR binary exported: {width}x{height} → {output_bin_path} ({file_size_mb:.1f} MB)")

'''functioning code starts here before it are all functions'''

psr_binary_destination = "./lunar_psr_dem_grid.bin"

#get the whole data in just one step....
psr_result = run_psr_analysis(r".\data\PSR\Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif",
    azimuth_step_deg=45,  # Widened from 5° (Cuts loop iterations from 72 down to 8!)
    max_steps=150         # Reduced from 500 (Looks up to ~17km away instead of 60km)
)

# this ius just here so that we can find what locations does the doubly craters are at and be able to get the proper DFSAR files.
print("\nDoubly shadowed crater longitudes:")
for ds in psr_result["doubly_shadowed_craters"]:
    cr = ds["center_row"]
    cc = ds["center_col"]
    
    bounds = psr_result["bounds"]
    left, bottom, right, top = bounds
    
    pixel_size = psr_result["pixel_size"]
    
    # Use the unpacked variables
    world_x = left + (cc + 0.5) * pixel_size
    world_y = top  - (cr + 0.5) * pixel_size
    
    moon_radius = 1737400.0
    deg_to_m = np.pi / 180.0 * moon_radius
    
    lon = world_x / deg_to_m
    lat = world_y / deg_to_m
    
    print(f"  lat={lat:.3f}° lon={lon:.3f}° | "
          f"depth={ds['depth_meters']:.0f}m | "
          f"radius={ds['radius_meters']/1000:.1f}km | "
          f"PSR={ds['psr_fraction']*100:.0f}%")

#export the interrleaved layers
export_psr_and_dem_for_cpp(psr_result, output_bin_path=psr_binary_destination)

print("")
print("---[PSR/DEM pipeline fully processed without any issues]---")