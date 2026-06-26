import os
import numpy as np
import rasterio as rs
from rasterio.enums import Resampling
from rasterio.transform import rowcol
from scipy.ndimage import sobel
import heapq
import struct
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
from matplotlib.widgets import Cursor, Button
import warnings
import logging

# ---- Clean Presentation Layer Filters ----
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('matplotlib').setLevel(logging.ERROR)


# ---- Mission cache: persists start/end/ice-estimate/waypoints per crater target ----
_CACHE_HEADER_FMT = "<ddfiiiiI"
_CACHE_HEADER_SIZE = struct.calcsize(_CACHE_HEADER_FMT)


def load_mission_cache(path):
    """Loads every cached mission from the cache bin file."""
    records = []
    if not os.path.exists(path):
        return records

    with open(path, "rb") as f:
        while True:
            header = f.read(_CACHE_HEADER_SIZE)
            if len(header) < _CACHE_HEADER_SIZE:
                break 

            (target_lat, target_lon, ice_vol_mid,
             landing_row, landing_col,
             target_row, target_col,
             num_waypoints) = struct.unpack(_CACHE_HEADER_FMT, header)

            wp_bytes = f.read(num_waypoints * 2 * 4)  
            if len(wp_bytes) < num_waypoints * 2 * 4:
                print(" **Mission cache truncated mid-record — stopping read**")
                break

            flat = struct.unpack(f"<{num_waypoints * 2}i", wp_bytes)
            waypoints = list(zip(flat[0::2], flat[1::2]))

            records.append({
                "target_lat": target_lat,
                "target_lon": target_lon,
                "ice_vol_mid": ice_vol_mid,
                "landing_pixel": (landing_row, landing_col),
                "target_pixel": (target_row, target_col),
                "waypoints": waypoints,
            })
    return records


def append_mission_record(path, target_lat, target_lon, ice_vol_mid,landing_pixel, target_pixel, waypoints):
    '''appends one freshly-computed mission to the cache bin file'''

    header = struct.pack(
        _CACHE_HEADER_FMT,
        target_lat, target_lon, ice_vol_mid,
        landing_pixel[0], landing_pixel[1],
        target_pixel[0], target_pixel[1],
        len(waypoints),
    )
    flat = []
    for r, c in waypoints:
        flat.append(int(r))
        flat.append(int(c))
    wp_bytes = struct.pack(f"<{len(flat)}i", *flat) if flat else b""

    with open(path, "ab") as f:
        f.write(header)
        f.write(wp_bytes)


def find_cached_record(records, target_lat, target_lon, tol=1e-6):
    '''Looks for an existing cached mission matching this target's lat/lon'''
    for rec in records:
        if abs(rec["target_lat"] - target_lat) < tol and abs(rec["target_lon"] - target_lon) < tol:
            return rec
    return None


def load_ice_mask_binary(path, expected_height, expected_width):
    """Parses lunar_ice_mask.bin to load verified sensor-fusion ice locations."""
    if not os.path.exists(path):
        print(f" **Ice mask file not found: {path}. Rendering map without verified ice overlay**")
        return np.zeros((expected_height, expected_width), dtype=np.uint8)
    
    with open(path, "rb") as f:
        header = f.read(40)
        if len(header) < 40:
            return np.zeros((expected_height, expected_width), dtype=np.uint8)
        
        width, height = struct.unpack("II", header[0:8])
        raw = np.frombuffer(f.read(), dtype=np.uint8)
        
        if raw.size < width * height:
            print(" **Ice mask data stream truncated**")
            return np.zeros((expected_height, expected_width), dtype=np.uint8)
            
        mask = raw[:width * height].reshape((height, width))
        return mask


class LunarLandingSiteSelector:
    def __init__(self, dem_path, illumination_path, resolution_m=118.5):
        self.pixel_scale = resolution_m

        print("Opening and cropping DEM Map (-80 to -90 degrees)...")
        with rs.open(dem_path) as src:
            global_height = src.height
            global_width = src.width
            self.target_crs = src.crs

            start_row = int(global_height * 17 / 18)
            height_window = global_height - start_row
            dem_window = rs.windows.Window(0, start_row, global_width, height_window)

            self.dem_data = src.read(1, window=dem_window).astype(np.float32)
            self.cropped_transform = rs.windows.transform(dem_window, src.transform)

        print(f" => Cropped DEM Shape: {self.dem_data.shape}")

        print("...Opening and resampling Illumination Map to match DEM grid...")
        with rs.open(illumination_path) as src_illum:
            target_h, target_w = self.dem_data.shape
            self.illum_data = np.empty((target_h, target_w), dtype=np.float32)

            from rasterio.warp import reproject

            reproject(
                source=rs.band(src_illum, 1),
                destination=self.illum_data,
                src_transform=src_illum.transform,
                src_crs=src_illum.crs,
                dst_transform=self.cropped_transform,
                dst_crs=self.target_crs,
                resampling=rs.warp.Resampling.bilinear,
            )
        print(" -> Illumination map reprojected and matched to tracking footprint.")

    def parse_cpp_binary(self, binary_path):
        records = []
        record_format = "ddIIffffB"
        record_size = struct.calcsize(record_format)

        if not os.path.exists(binary_path):
            print(f"Warning: Binary file path not found: {binary_path}")
            return records

        with open(binary_path, "rb") as f:
            n_doubly = struct.unpack("I", f.read(4))[0]
            n_normal = struct.unpack("I", f.read(4))[0]
            total_records = n_doubly + n_normal

            for i in range(total_records):
                data = f.read(record_size)
                if not data:
                    break
                unpacked = struct.unpack(record_format, data)

                records.append({
                    "lat": unpacked[0],
                    "lon": unpacked[1],
                    "pixelCount": unpacked[2],
                    "icePxCount": unpacked[3],
                    "volMid": unpacked[4],
                    "category": unpacked[8],
                    "is_doubly_shadowed": i < n_doubly,
                })

        print(f" -> Parsed {len(records)} crater targets from binary "
              f"({n_doubly} doubly-shadowed, {n_normal} normal PSR).")
        return records

    def calculate_slopes_and_roughness(self, sub_dem):
        dx = sobel(sub_dem, axis=1) / (8.0 * self.pixel_scale)
        dy = sobel(sub_dem, axis=0) / (8.0 * self.pixel_scale)

        slope_radians = np.arctan(np.sqrt(dx ** 2 + dy ** 2))
        slope_degrees = np.degrees(slope_radians)

        dx2 = sobel(dx, axis=1) / (8.0 * self.pixel_scale)
        dy2 = sobel(dy, axis=0) / (8.0 * self.pixel_scale)
        roughness = np.sqrt(dx2 ** 2 + dy2 ** 2)

        return slope_degrees, roughness

    def evaluate_window(self, center_r, center_c, radius):
        h, w = self.dem_data.shape

        if center_r < 0 or center_r >= h or center_c < 0 or center_c >= w:
            print(f" **Target coordinate mapped to index [{center_r}, {center_c}], which is outside our cropped array layout**")
            return None

        r_min, r_max = max(0, center_r - radius), min(h, center_r + radius + 1)
        c_min, c_max = max(0, center_c - radius), min(w, center_c + radius + 1)

        sub_dem = self.dem_data[r_min:r_max, c_min:c_max]
        sub_illum = self.illum_data[r_min:r_max, c_min:c_max]

        if sub_dem.size == 0 or sub_illum.size == 0:
            return None

        slopes, roughness = self.calculate_slopes_and_roughness(sub_dem)

        if roughness.size == 0:
            return None

        roughness_threshold = np.percentile(roughness, 75)

        r_indices, c_indices = np.ogrid[r_min:r_max, c_min:c_max]
        distances_m = np.sqrt((r_indices - center_r) ** 2 + (c_indices - center_c) ** 2) * self.pixel_scale

        valid_mask = (
            (slopes < 15.0) &
            (roughness < roughness_threshold) &
            (sub_illum >= 0.70) &
            (distances_m < 10000.0)
        )

        valid_indices = np.argwhere(valid_mask)
        if len(valid_indices) == 0:
            return None

        best_score = -float("inf")
        best_local_coords = None

        for idx in valid_indices:
            vr, vc = idx[0], idx[1]
            illum_val = sub_illum[vr, vc]
            dist_val = distances_m[vr, vc]

            score = (illum_val * 100.0) - (dist_val / 100.0)

            if score > best_score:
                best_score = score
                best_local_coords = (r_min + vr, c_min + vc)

        return best_local_coords

    def find_landing_site(self, target_lat, target_lon, initial_radius=100, max_radius=1000):
        print(f"...Transforming Lat={target_lat:.3f}, Lon={target_lon:.3f} to native Lunar map projection...")

        LUNAR_RADIUS = 1737400.0
        native_x = LUNAR_RADIUS * np.radians(target_lon)
        native_y = LUNAR_RADIUS * np.radians(target_lat)

        center_r, center_c = rowcol(self.cropped_transform, native_x, native_y)
        target_pixel = (int(center_r), int(center_c))

        h, w = self.dem_data.shape
        print(f" -> Mapped Target Crater Grid Indices: Row={target_pixel[0]}, Col={target_pixel[1]}")

        if center_r < 0 or center_r >= h or center_c < 0 or center_c >= w:
            print(f"\n**Target Index [{target_pixel[0]}, {target_pixel[1]}] is outside the cropped layout limits**")
            return None, target_pixel

        current_radius = initial_radius
        while current_radius <= max_radius:
            print(f"...Searching for candidate locations within a radius of {current_radius} pixels...")
            landing_pixel = self.evaluate_window(center_r, center_c, current_radius)

            if landing_pixel is not None:
                safe_landing_pixel = (int(landing_pixel[0]), int(landing_pixel[1]))
                return safe_landing_pixel, target_pixel

            current_radius += 100

        return None, target_pixel


class LunarPathfinder:
    def __init__(self, dem_matrix, psr_binary_path):
        self.dem = dem_matrix
        self.height, self.width = dem_matrix.shape
        self.pixel_scale = 118.5
        self.psr_mask = self._load_psr_binary(psr_binary_path)

    def _load_psr_binary(self, path):
        print("...Parsing PSR Binary Data Grid...")
        try:
            with open(path, "rb") as f:
                header = f.read(40)
                if len(header) < 40:
                    print(" **PSR binary header truncated. Initializing zero mask**")
                    return np.zeros((self.height, self.width), dtype=np.uint8)

                width, height = struct.unpack("II", header[0:8])
                raw = np.frombuffer(f.read(), dtype=np.float32)
                expected_floats = width * height * 2

                if raw.size < expected_floats:
                    return np.zeros((self.height, self.width), dtype=np.uint8)

                interleaved = raw[:expected_floats].reshape((height, width, 2))
                weight = interleaved[:, :, 0]
                mask = (weight <= 0.9).astype(np.uint8)

                print(f" -> PSR Mask successfully parsed! Found {np.sum(mask == 1)} shadow/double-shadow pixels")
                return mask
        except FileNotFoundError:
            print(" **PSR binary grid file not found! Defaulting to empty shadow mask**")
            return np.zeros((self.height, self.width), dtype=np.uint8)

    def _calculate_edge_cost(self, r1, c1, r2, c2):
        is_diagonal = (r1 != r2) and (c1 != c2)
        distance = self.pixel_scale * (np.sqrt(2) if is_diagonal else 1.0)

        elevation_delta = abs(self.dem[r2, c2] - self.dem[r1, c1])
        slope_angle = np.degrees(np.arctan2(elevation_delta, distance))

        if slope_angle > 15.0:
            return float("inf")

        slope_cost_multiplier = 1.0 + (slope_angle / 15.0) ** 2
        psr_multiplier = 15.0 if self.psr_mask[r2, c2] == 1 else 1.0

        return distance * slope_cost_multiplier * psr_multiplier

    def _heuristic(self, r, c, target_r, target_c):
        dr = abs(r - target_r)
        dc = abs(c - target_c)
        return self.pixel_scale * ((dr + dc) + (np.sqrt(2) - 2) * min(dr, dc))

    def compute_astar_path(self, start_coords, goal_coords, max_nodes=300000):
        start_r, start_c = start_coords
        goal_r, goal_c = goal_coords

        open_set = []
        heapq.heappush(open_set, (0.0, 0.0, start_r, start_c))

        came_from = {}
        g_score = {(start_r, start_c): 0.0}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        print(f"...Pathfinding initialized from {start_coords} to {goal_coords}...")

        nodes_expanded = 0

        #define a bounding box to prevent runaway searches across the entire moon
        search_radius = max(abs(start_r - goal_r), abs(start_c - goal_c)) * 2
        search_radius = max(search_radius, 500) # At least 500px leeway
        
        min_r = max(0, min(start_r, goal_r) - search_radius)
        max_r = min(self.height - 1, max(start_r, goal_r) + search_radius)
        min_c = max(0, min(start_c, goal_c) - search_radius)
        max_c = min(self.width - 1, max(start_c, goal_c) + search_radius)

        while open_set:
            _, current_g, r, c = heapq.heappop(open_set)
            
            nodes_expanded += 1
            if nodes_expanded > max_nodes:
                print(f" **Pathfinding aborted: Exceeded maximum node limit ({max_nodes}). Target is likely unreachable**")
                return None

            if (r, c) == (goal_r, goal_c):
                print(f" ---Target Path Successfully Resolved! (Checked {nodes_expanded} nodes)---")
                return self._reconstruct_path(came_from, (r, c))

            for dr, dc in directions:
                nr, nc = r + dr, c + dc

                # check if out of global bounds OR outside our local search box
                if not (min_r <= nr <= max_r and min_c <= nc <= max_c):
                    continue

                edge_cost = self._calculate_edge_cost(r, c, nr, nc)
                if edge_cost == float("inf"):
                    continue

                tentative_g = current_g + edge_cost

                if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._heuristic(nr, nc, goal_r, goal_c)
                    came_from[(nr, nc)] = (r, c)
                    heapq.heappush(open_set, (f_score, tentative_g, nr, nc))

        print(f" **No viable route found (blocked completely by terrain constraints). Checked {nodes_expanded} nodes**")
        return None

    def _reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


def generate_isro_navigation_screen(dem_matrix, psr_mask, ice_mask, missions, transform, start_index=0):
    print("\nInitializing Flight Telemetry Dashboard...")
    height, width = dem_matrix.shape
    LUNAR_RADIUS = 1737400.0

    def pixel_to_latlon(col, row):
        native_x, native_y = transform * (col, row)
        lon = np.degrees(native_x / LUNAR_RADIUS)
        lat = np.degrees(native_y / LUNAR_RADIUS)
        return (lat, (lon + 180) % 360 - 180)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9), facecolor="#060606")

    fig_manager = plt.get_current_fig_manager()
    if hasattr(fig_manager, "window") and hasattr(fig_manager.window, "title"):
        fig_manager.window.title("ISRO MISSION MANAGEMENT - LUNAR NAVIGATION")
    elif hasattr(fig_manager, "set_window_title"):
        fig_manager.set_window_title("ISRO MISSION MANAGEMENT - LUNAR NAVIGATION")

    state = {"idx": start_index % len(missions), "cid": None}

    def render_mission(idx):
        mission = missions[idx]
        landing_pixel = mission["landing_pixel"]
        target_pixel = mission["target_pixel"]
        mission_route = mission["waypoints"]

        if state["cid"] is not None:
            fig.canvas.mpl_disconnect(state["cid"])

        fig.clf()

        path_array = np.array(mission_route)
        min_r, max_r = path_array[:, 0].min(), path_array[:, 0].max()
        min_c, max_c = path_array[:, 1].min(), path_array[:, 1].max()

        center_r = int((min_r + max_r) / 2)
        center_c = int((min_c + max_c) / 2)
        
        # Wide regional zoom tracker 
        view_half_size = 500

        crop_min_r, crop_max_r = max(0, center_r - view_half_size), min(height, center_r + view_half_size)
        crop_min_c, crop_max_c = max(0, center_c - view_half_size), min(width, center_c + view_half_size)

        localized_dem = dem_matrix[crop_min_r:crop_max_r, crop_min_c:crop_max_c]
        localized_psr = psr_mask[crop_min_r:crop_max_r, crop_min_c:crop_max_c]
        localized_ice = ice_mask[crop_min_r:crop_max_r, crop_min_c:crop_max_c]

        # Crater Permanent Shadow Region Overlay (Subtle Dark Tonal Layer)
        psr_overlay = np.zeros((*localized_psr.shape, 4), dtype=np.float32)
        psr_overlay[..., 0] = 0.1
        psr_overlay[..., 1] = 0.1
        psr_overlay[..., 2] = 0.2
        psr_overlay[..., 3] = np.where(localized_psr == 1, 0.35, 0.0)

        #Confirmed Ice Reserves Overlay (Vivid Translucent Blue Layer)
        ice_overlay = np.zeros((*localized_ice.shape, 4), dtype=np.float32)
        ice_overlay[..., 0] = 0.0  # R
        ice_overlay[..., 1] = 0.5  # G
        ice_overlay[..., 2] = 1.0  # B -> Vibrant electric blue
        ice_overlay[..., 3] = np.where(localized_ice == 1, 0.65, 0.0)

        total_ice_pixels = np.sum(localized_ice > 0)
        estimated_ice_area_km2 = (total_ice_pixels * (118.5 ** 2)) / 1e6

        land_lat, land_lon = pixel_to_latlon(landing_pixel[1], landing_pixel[0])
        targ_lat, targ_lon = pixel_to_latlon(target_pixel[1], target_pixel[0])

        ax_map = fig.add_axes([0.30, 0.12, 0.56, 0.76])

        sb_x = 0.03
        fig.text(sb_x, 0.93, "ISRO MISSION PANEL", fontsize=14, color="white", fontweight="bold", alpha=0.9)
        fig.text(sb_x, 0.87, "ACTIVE TARGET AREA:", fontsize=10, color="#aaaaaa")
        fig.text(sb_x, 0.84, f"crater_south_{abs(targ_lat):.1f}", fontsize=13, color="white", fontweight="bold")

        fig.text(sb_x, 0.78, "ROUTE TRACK LIMITS:", fontsize=10, color="#aaaaaa")
        fig.text(sb_x, 0.75, f"Optimized Trajectory ({len(mission_route)} WP)", fontsize=13, color="#38bdf8", fontweight="bold")

        fig.text(sb_x, 0.67, "+SAFE LANDING COORDS:", fontsize=10, color="#4ade80", fontweight="bold")
        fig.text(sb_x, 0.64, f"Matrix: [{landing_pixel[0]}, {landing_pixel[1]}]", fontsize=11, color="white")
        fig.text(sb_x, 0.59, f"Lat : {land_lat:.5f}°\nLon : {land_lon:.5f}°", fontsize=11, color="#4ade80")

        fig.text(sb_x, 0.51, "+CRATER TARGET OBJECTIVE:", fontsize=10, color="#f87171", fontweight="bold")
        fig.text(sb_x, 0.48, f"Matrix: [{target_pixel[0]}, {target_pixel[1]}]", fontsize=11, color="white")
        fig.text(sb_x, 0.43, f"Lat : {targ_lat:.5f}°\nLon : {targ_lon:.5f}°", fontsize=11, color="#f87171")

        fig.text(sb_x, 0.35, "+DETECTED ICE SHEET METRICS:", fontsize=10, color="#38bdf8", fontweight="bold")
        fig.text(sb_x, 0.32, f"Total Regional Ice: {total_ice_pixels} Px", fontsize=11, color="white")
        fig.text(sb_x, 0.29, f"Surface Ice Area  : {estimated_ice_area_km2:.3f} km²", fontsize=11, color="#38bdf8")
        fig.text(sb_x, 0.25, f"Target Vol Est    : {mission['ice_vol_mid']:,.0f} m³", fontsize=11, color="#38bdf8")

        fig.text(sb_x, 0.18, "+LIVE SENSOR TRACKER:", fontsize=10, color="#22d3ee", fontweight="bold")
        hud_text = fig.text(sb_x, 0.11, "LAT: ---.----°\nLON: ---.----°", fontsize=12, color="#22d3ee", fontweight="bold",
                             bbox=dict(boxstyle="round,pad=0.4", fc="#111111", ec="#22d3ee", alpha=0.6))

        fig.text(0.58, 0.94, "LUNAR SOUTH POLE DESCENT ARCHITECTURE", fontsize=18, color="white", fontweight="bold", ha="center")
        fig.text(0.58, 0.04, "Data Core: Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif", fontsize=9, color="#666666", ha="center")
        fig.text(0.58, 0.965, f"Target {idx + 1} / {len(missions)}", fontsize=11, color="#888888", ha="center")

        local_vmin, local_vmax = np.min(localized_dem), np.max(localized_dem)

        img = ax_map.imshow(localized_dem, cmap="gray", vmin=local_vmin, vmax=local_vmax,
                     extent=[crop_min_c, crop_max_c, crop_max_r, crop_min_r],
                     aspect="equal", interpolation="lanczos", origin="upper", zorder=1)

        # Draw structural shadows underneath
        ax_map.imshow(psr_overlay, extent=[crop_min_c, crop_max_c, crop_max_r, crop_min_r], aspect="equal", origin="upper", zorder=2)
        
        # Draw translucent blue verified ice grid on top
        ax_map.imshow(ice_overlay, extent=[crop_min_c, crop_max_c, crop_max_r, crop_min_r], aspect="equal", origin="upper", zorder=3)
        
        ax_map.plot(path_array[:, 1], path_array[:, 0], color="#ef4444", linewidth=3, zorder=4)
        ax_map.scatter(landing_pixel[1], landing_pixel[0], color="#4ade80", s=35, edgecolors="black", zorder=5)
        ax_map.scatter(target_pixel[1], target_pixel[0], color="#f87171", s=35, edgecolors="black", zorder=5)

        lbl_style = path_effects.withStroke(linewidth=3, foreground="#060606")
        ax_map.text(landing_pixel[1] + 6, landing_pixel[0] - 6, "LANDING SITE", fontsize=10, color="#4ade80", fontweight="bold", path_effects=[lbl_style], zorder=6)
        ax_map.text(target_pixel[1] - 6, target_pixel[0] + 10, "ICE SITE", fontsize=10, color="#f87171", fontweight="bold", path_effects=[lbl_style], zorder=6, ha="right")

        ax_map.set_xlabel("Columns", color="#888888", labelpad=8)
        ax_map.set_ylabel("Rows", color="#888888", labelpad=8)
        ax_map.grid(True, linestyle=":", color="#333333", alpha=0.6, zorder=0)

        cb_axes = fig.add_axes([0.88, 0.12, 0.015, 0.76])
        cbar = fig.colorbar(img, cax=cb_axes)
        cbar.outline.set_edgecolor("#444444")

        cursor = Cursor(ax_map, useblit=False, color="#22d3ee", linewidth=0.8, linestyle="--")
        ax_map._cursor = cursor

        def on_mouse_move(event):
            if event.inaxes == ax_map and event.xdata is not None and event.ydata is not None:
                lat, lon = pixel_to_latlon(event.xdata, event.ydata)
                hud_text.set_text(f"LAT: {lat:.5f}°\nLON: {lon:.5f}°")
            else:
                hud_text.set_text("LAT: ---.----°\nLON: ---.----°")
            fig.canvas.draw_idle()

        state["cid"] = fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

        ax_prev = fig.add_axes([0.30, 0.005, 0.08, 0.045])
        ax_next = fig.add_axes([0.78, 0.005, 0.08, 0.045])
        btn_prev = Button(ax_prev, "◀ Prev", color="#1a1a1a", hovercolor="#333333")
        btn_next = Button(ax_next, "Next ▶", color="#1a1a1a", hovercolor="#333333")
        for btn in (btn_prev, btn_next):
            btn.label.set_color("white")
        ax_map._buttons = (btn_prev, btn_next)

        def go_prev(event):
            try:
                if hasattr(fig.canvas, 'widgetlock') and fig.canvas.widgetlock.locked():
                    fig.canvas.widgetlock.release(fig.canvas.widgetlock._owner)
            except Exception:
                pass
            state["idx"] = (state["idx"] - 1) % len(missions)
            render_mission(state["idx"])

        def go_next(event):
            try:
                if hasattr(fig.canvas, 'widgetlock') and fig.canvas.widgetlock.locked():
                    fig.canvas.widgetlock.release(fig.canvas.widgetlock._owner)
            except Exception:
                pass
            state["idx"] = (state["idx"] + 1) % len(missions)
            render_mission(state["idx"])

        btn_prev.on_clicked(go_prev)
        btn_next.on_clicked(go_next)
        fig.canvas.draw_idle()

    render_mission(state["idx"])
    print(" -> Screen active. Opening UI window...")
    plt.show()


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    #loading all the data needed from other bin and tif files.
    dem_file = os.path.normpath(os.path.join(SCRIPT_DIR, "./data/PSR/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"))
    illum_file = os.path.normpath(os.path.join(SCRIPT_DIR, "./data/ILLUMI/AVG_ILLUMINATION.tiff"))
    binary_data = os.path.normpath(os.path.join(SCRIPT_DIR, "lunar_psr_locations.bin"))
    psr_grid_file = os.path.normpath(os.path.join(SCRIPT_DIR, "lunar_psr_dem_grid.bin"))
    ice_mask_file = os.path.normpath(os.path.join(SCRIPT_DIR, "lunar_ice_mask.bin"))
    mission_cache_file = os.path.normpath(os.path.join(SCRIPT_DIR, "mission_cache.bin"))

    if not os.path.exists(dem_file) or not os.path.exists(illum_file):
        print("**Error: Verification failed for tracking files**")
    else:
        selector = LunarLandingSiteSelector(
            dem_path=dem_file,
            illumination_path=illum_file,
            resolution_m=118.5,
        )

        targets = selector.parse_cpp_binary(binary_data)

        if not targets:
            print("**No crater targets found in binary — nothing to plan**")
        else:
            pathfinder = LunarPathfinder(
                dem_matrix=selector.dem_data,
                psr_binary_path=psr_grid_file,
            )

            print("Parsing Verified Ice Mask Binary Grid...")
            ice_mask = load_ice_mask_binary(ice_mask_file, selector.dem_data.shape[0], selector.dem_data.shape[1])

            cached_records = load_mission_cache(mission_cache_file)
            print(f"=>Mission cache: {len(cached_records)} record(s) loaded from {mission_cache_file}")

            missions = []

            for i, target in enumerate(targets):
                print(f"\n--- Target {i + 1}/{len(targets)}: lat={target['lat']:.3f}, lon={target['lon']:.3f} ---")

                cached = find_cached_record(cached_records, target["lat"], target["lon"])
                if cached is not None:
                    print("-- Found in cache — skipping recompute--")
                    missions.append(cached)
                    continue

                print(" ...Not cached — computing landing site + path...")
                landing_pixel, target_pixel = selector.find_landing_site(target["lat"], target["lon"])

                if landing_pixel is None:
                    print(" **No safe landing site found — skipping this target**")
                    continue

                mission_route = pathfinder.compute_astar_path(landing_pixel, target_pixel)

                if mission_route is None:
                    print(" **No viable path found — skipping this target**")
                    continue

                append_mission_record(
                    mission_cache_file,
                    target["lat"], target["lon"], target["volMid"],
                    landing_pixel, target_pixel, mission_route,
                )

                missions.append({
                    "target_lat": target["lat"],
                    "target_lon": target["lon"],
                    "ice_vol_mid": target["volMid"],
                    "landing_pixel": landing_pixel,
                    "target_pixel": target_pixel,
                    "waypoints": mission_route,
                })

            if missions:
                print(f"\n---{len(missions)} mission(s) ready — opening navigation dashboard---")
                generate_isro_navigation_screen(
                    dem_matrix=selector.dem_data,
                    psr_mask=pathfinder.psr_mask,
                    ice_mask=ice_mask,
                    missions=missions,
                    transform=selector.cropped_transform,
                )
            else:
                print("\n**Mission Aborted: no targets produced a viable landing+path**")