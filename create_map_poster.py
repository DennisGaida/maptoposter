
# create_map_poster.py (unified) — Buildings + Sun + POIs + Color overrides
# Includes: rectangular bbox (landscape fill) + OSMnx bbox-compat wrappers
# Adds: Overpass/OSMnx tuning via env/CLI; clean exit codes (0 on success), no side-effects

import os, time, json, argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import osmnx as ox
from shapely import affinity
from shapely.geometry import box as shp_box

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

QUALITY_PRESETS = {
    'standard': {'figsize': (12, 16), 'dpi': 300},
    'high':     {'figsize': (16, 21), 'dpi': 400},
    'ultra':    {'figsize': (18, 24), 'dpi': 600},
}

FONTS = None
THEME = None

# --- OSMnx runtime settings (overridable via env and CLI) ---
try:
    _DEFAULT_MAX_AREA = 50_000 * 50_000  # ≈ 50km x 50km in m²
    ox.settings.max_query_area_size = int(os.getenv("MAPTOP_MAX_QUERY_AREA_SIZE", _DEFAULT_MAX_AREA))
    if hasattr(ox.settings, "overpass_memory"):
        ox.settings.overpass_memory = int(os.getenv("MAPTOP_OVERPASS_MEMORY", "4096"))
    ox.settings.requests_timeout = int(os.getenv("MAPTOP_REQUESTS_TIMEOUT", "180"))
    ox.settings.log_console = bool(int(os.getenv("MAPTOP_LOG_CONSOLE", "1")))
except Exception:
    pass

# --- OSMnx bbox compatibility helpers ---
def _graph_from_bbox_compat(n, s, e, w, **kwargs):
    try:
        return ox.graph_from_bbox(n, s, e, w, **kwargs)
    except TypeError:
        try:
            return ox.graph_from_bbox(north=n, south=s, east=e, west=w, **kwargs)
        except TypeError:
            try:
                return ox.graph_from_bbox((n, s, e, w), **kwargs)
            except TypeError:
                poly = shp_box(w, s, e, n)
                return ox.graph_from_polygon(poly, **kwargs)

def _features_from_bbox_compat(n, s, e, w, *, tags=None):
    try:
        return ox.features_from_bbox(n, s, e, w, tags=tags)
    except TypeError:
        try:
            return ox.features_from_bbox(north=n, south=s, east=e, west=w, tags=tags)
        except TypeError:
            try:
                return ox.features_from_bbox((n, s, e, w), tags=tags)
            except TypeError:
                poly = shp_box(w, s, e, n)
                return ox.features_from_polygon(poly, tags=tags)


def load_fonts():
    global FONTS
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }
    for p in fonts.values():
        if not os.path.exists(p):
            print(f"⚠ Font not found: {p}")
            FONTS = None
            return
    FONTS = fonts


def generate_output_filename(city, theme_name, *, distance, quality, landscape, project_metric,
                              buildings=False, landmarks_only=False, extrude_depth=12, min_levels=5,
                              sun_azimuth=45, sun_elevation=45, pois=False, poi_tags_custom=False):
    os.makedirs(POSTERS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = (city or "coords").lower().replace(' ', '_')
    orient = 'l' if landscape else 'p'
    pm_flag = 't' if project_metric else 'f'
    b_flag = 'lm' if (buildings and landmarks_only) else ('all' if buildings else 'none')
    pois_flag = 'c' if (pois and poi_tags_custom) else ('t' if pois else 'f')
    fname = (f"{city_slug}_{theme_name}_{orient}_q{quality}_d{distance}_pm_{pm_flag}"
             f"_b{b_flag}_x{int(extrude_depth)}_ml{int(min_levels)}_sa{int(round(sun_azimuth))}"
             f"_se{int(round(sun_elevation))}_pois{pois_flag}_{ts}.png")
    return os.path.join(POSTERS_DIR, fname)


def get_available_themes():
    os.makedirs(THEMES_DIR, exist_ok=True)
    return [f[:-5] for f in sorted(os.listdir(THEMES_DIR)) if f.endswith('.json')]


def load_theme(theme_name="feature_based"):
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default feature_based theme.")
        return {"name":"Feature-Based Shading","bg":"#FFFFFF","text":"#000000","gradient_color":"#FFFFFF",
                "water":"#C0C0C0","parks":"#F0F0F0","road_motorway":"#0A0A0A","road_primary":"#1A1A1A",
                "road_secondary":"#2A2A2A","road_tertiary":"#3A3A3A","road_residential":"#4A4A4A","road_default":"#3A3A3A",
                "building_fill":"#1f1f1f","building_shadow":"#000000","poi_color":"#D33F49","poi_label_color":"#000000"}
    with open(theme_file, 'r', encoding='utf-8') as f:
        theme = json.load(f)
    theme.setdefault("building_fill", "#1f1f1f")
    theme.setdefault("building_shadow", "#000000")
    theme.setdefault("poi_color", "#D33F49")
    theme.setdefault("poi_label_color", theme.get("text", "#000000"))
    print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
    if 'description' in theme: print(f"  {theme['description']}")
    return theme


def create_gradient_fade(ax, color, location='bottom', zorder=10):
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    rgb = mcolors.to_rgb(color)
    my = np.zeros((256, 4))
    my[:,0], my[:,1], my[:,2] = rgb
    if location == 'bottom':
        my[:,3] = np.linspace(1, 0, 256); y0, y1 = 0, 0.25
    else:
        my[:,3] = np.linspace(0, 1, 256); y0, y1 = 0.75, 1.0
    cmap = mcolors.ListedColormap(my)
    xlim = ax.get_xlim(); ylim = ax.get_ylim(); yr = ylim[1]-ylim[0]
    yb = ylim[0] + yr*y0; yt = ylim[0] + yr*y1
    ax.imshow(vals, extent=[xlim[0], xlim[1], yb, yt], aspect='auto', cmap=cmap, zorder=zorder, origin='lower')


def get_edge_colors_by_type(G):
    colors = []
    for _, _, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list): highway = highway[0] if highway else 'unclassified'
        if highway in ['motorway','motorway_link']: c = THEME['road_motorway']
        elif highway in ['trunk','trunk_link','primary','primary_link']: c = THEME['road_primary']
        elif highway in ['secondary','secondary_link']: c = THEME['road_secondary']
        elif highway in ['tertiary','tertiary_link']: c = THEME['road_tertiary']
        elif highway in ['residential','living_street','unclassified']: c = THEME['road_residential']
        else: c = THEME['road_default']
        colors.append(c)
    return colors


def get_edge_widths_by_type(G):
    widths = []
    for _, _, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list): highway = highway[0] if highway else 'unclassified'
        if highway in ['motorway','motorway_link']: w = 1.2
        elif highway in ['trunk','trunk_link','primary','primary_link']: w = 1.0
        elif highway in ['secondary','secondary_link']: w = 0.8
        elif highway in ['tertiary','tertiary_link']: w = 0.6
        else: w = 0.4
        widths.append(w)
    return widths

# --- bbox helpers ---

def _rect_bbox_from_distance(point, figsize, distance_m):
    lat, lon = point
    width_in, height_in = figsize
    aspect = width_in / height_in
    if aspect >= 1.0:
        half_x_m = float(distance_m) * aspect
        half_y_m = float(distance_m)
    else:
        half_x_m = float(distance_m)
        half_y_m = float(distance_m) / aspect
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat))
    dlat = half_y_m / meters_per_deg_lat
    dlon = half_x_m / max(meters_per_deg_lon, 1e-9)
    return lat + dlat, lat - dlat, lon + dlon, lon - dlon

# --- landmark & plotting helpers ---

def _clip(v, lo, hi): return max(lo, min(hi, v))

def _shadow_offset_from_azimuth(depth_m, azimuth_deg):
    rad = np.deg2rad(azimuth_deg)
    return float(depth_m*np.sin(rad)), float(-depth_m*np.cos(rad))

def _effective_shadow_depth(extrude_depth, sun_elevation_deg):
    elev = _clip(float(sun_elevation_deg), 1.0, 89.0)
    tanv = np.tan(np.deg2rad(elev))
    eff = extrude_depth * (1.0/max(tanv,0.2))
    return float(_clip(eff, extrude_depth*0.4, extrude_depth*4.0))

def _shadow_alpha_for_elevation(base_alpha, sun_elevation_deg):
    elev = _clip(float(sun_elevation_deg), 0.0, 90.0)
    return float(base_alpha * (0.5 + 0.5*(1.0 - elev/90.0)))


def _filter_landmarks(bld, min_levels=5):
    import numpy as _np
    cands = []
    for col in ['name','wikidata','brand']:
        if col in bld.columns: cands.append(bld[col].notna())
    if 'tourism' in bld.columns:
        cands.append(bld['tourism'].isin(['attraction','museum','gallery','viewpoint','theme_park','zoo']))
    if 'historic' in bld.columns: cands.append(bld['historic'].notna())
    if 'building:levels' in bld.columns:
        try:
            lvl = bld['building:levels'].astype(float)
            cands.append(lvl >= float(min_levels))
        except Exception: pass
    if cands:
        mask = _np.logical_or.reduce(cands)
        return bld[mask]
    return bld


def _plot_extruded(ax, gdf, extrude_depth=12, sun_azimuth=45, sun_elevation=45,
                   fill_color="#1f1f1f", shadow_color="#000000", alpha_fill=0.9, base_alpha_shadow=0.25, z_base=3,
                   shadow_alpha_override=None):
    if gdf is None or gdf.empty: return
    polys = gdf[gdf.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
    if polys.empty: return
    eff_depth = _effective_shadow_depth(float(extrude_depth), float(sun_elevation))
    dx, dy = _shadow_offset_from_azimuth(eff_depth, sun_azimuth)
    alpha_shadow = _shadow_alpha_for_elevation(base_alpha_shadow, sun_elevation) if shadow_alpha_override is None else float(shadow_alpha_override)
    shadow = polys.copy(); shadow['geometry'] = shadow['geometry'].apply(lambda g: affinity.translate(g, xoff=dx, yoff=dy))
    shadow.plot(ax=ax, facecolor=shadow_color, edgecolor='none', alpha=alpha_shadow, zorder=z_base)
    polys.plot(ax=ax, facecolor=fill_color, edgecolor='none', alpha=alpha_fill, zorder=z_base+1)


def _plot_flat(ax, gdf, fill_color="#1f1f1f", alpha_fill=0.8, z_base=3):
    if gdf is None or gdf.empty: return
    polys = gdf[gdf.geometry.type.isin(['Polygon','MultiPolygon'])]
    if polys.empty: return
    polys.plot(ax=ax, facecolor=fill_color, edgecolor='none', alpha=alpha_fill, zorder=z_base)

# --- POIs helpers ---

def _default_poi_tags():
    return {
        'tourism': ['attraction','museum','gallery','viewpoint'],
        'historic': True,
        'amenity': ['theatre','arts_centre','townhall','courthouse','library','university','place_of_worship'],
        'building': ['church','cathedral','castle','stadium','public','civic','city_hall','train_station'],
        'leisure': ['stadium'],
        'man_made': ['lighthouse','tower']
    }

def _parse_poi_tags(tag_str):
    if not tag_str: return None
    tags = {}
    for part in [p.strip() for p in tag_str.split(';') if p.strip()]:
        if '=' not in part: continue
        k, v = part.split('=', 1); k, v = k.strip(), v.strip()
        if v.lower() in ('true','yes','1'): tags[k] = True
        else:
            vals = [vv.strip() for vv in v.split(',') if vv.strip()]
            tags[k] = vals[0] if len(vals)==1 else vals
    return tags

# --- Main poster ---

def create_poster(city, country, point, dist, output_file, *, figsize=(12,16), dpi=300,
                  project_metric=True, landscape=False,
                  buildings=False, landmarks_only=False, extrude_depth=12, min_levels=5, building_opacity=0.9,
                  sun_azimuth=45, sun_elevation=45, shadow_alpha=None,
                  pois=False, poi_labels=False, poi_max=20, poi_size=28, poi_color=None, poi_tags=None,
                  building_color=None, shadow_color=None):
    print(f"\nGenerating map for {city}, {country}...")
    px = (int(round(figsize[0]*dpi)), int(round(figsize[1]*dpi)))
    print(f"Canvas: {figsize[0]}x{figsize[1]} inches @ {dpi} DPI -> {px[0]}x{px[1]} px  orientation={'landscape' if landscape else 'portrait'}")

    north, south, east, west = _rect_bbox_from_distance(point, figsize, dist)

    steps = 3 + (1 if buildings else 0) + (1 if pois else 0)
    from tqdm import tqdm as _tqdm
    with _tqdm(total=steps, desc="Fetching map data", unit="step",
               bar_format='{l_bar}{bar}{n_fmt}/{total_fmt}') as pbar:
        pbar.set_description("Downloading street network")
        G = _graph_from_bbox_compat(north, south, east, west, network_type='all')
        pbar.update(1); time.sleep(0.2)

        pbar.set_description("Downloading water features")
        try:
            water = _features_from_bbox_compat(north, south, east, west, tags={'natural':'water','waterway':'riverbank'})
        except Exception: water = None
        pbar.update(1); time.sleep(0.1)

        pbar.set_description("Downloading parks/green spaces")
        try:
            parks = _features_from_bbox_compat(north, south, east, west, tags={'leisure':'park','landuse':'grass'})
        except Exception: parks = None
        pbar.update(1)

        bld = None
        if buildings:
            pbar.set_description("Downloading buildings")
            try:
                bld = _features_from_bbox_compat(north, south, east, west, tags={'building': True})
            except Exception: bld = None
            pbar.update(1)

        pois_gdf = None
        poi_tags_custom = False
        if pois:
            pbar.set_description("Downloading POIs")
            tags = _parse_poi_tags(poi_tags) if poi_tags else _default_poi_tags()
            try:
                pois_gdf = _features_from_bbox_compat(north, south, east, west, tags=tags)
            except Exception: pois_gdf = None
            poi_tags_custom = bool(poi_tags)
            pbar.update(1)

    bbox_poly = shp_box(west, south, east, north)

    if project_metric:
        G = ox.project_graph(G)
        graph_crs = G.graph.get("crs")
        if water is not None and not water.empty: water = water.to_crs(graph_crs)
        if parks is not None and not parks.empty: parks = parks.to_crs(graph_crs)
        if bld is not None and not bld.empty:     bld   = bld.to_crs(graph_crs)
        if pois_gdf is not None and not pois_gdf.empty: pois_gdf = pois_gdf.to_crs(graph_crs)
    print("✓ All data downloaded successfully!")

    print("Rendering map...")
    fig, ax = plt.subplots(figsize=figsize, facecolor=THEME['bg'])
    ax.set_facecolor(THEME['bg']); ax.set_position([0,0,1,1])

    try:
        import geopandas as gpd
        bbox_gs = gpd.GeoSeries([bbox_poly], crs="EPSG:4326")
        if project_metric:
            bbox_gs = bbox_gs.to_crs(G.graph.get("crs"))
        minx, miny, maxx, maxy = bbox_gs.total_bounds
        ax.set_xlim([minx, maxx]); ax.set_ylim([miny, maxy])
    except Exception: pass

    if project_metric:
        ax.set_aspect('equal', adjustable='datalim')

    if water is not None and not water.empty:
        wp = water[water.geometry.type.isin(['Polygon','MultiPolygon'])]
        if not wp.empty: wp.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=1)

    if parks is not None and not parks.empty:
        pk = parks[parks.geometry.type.isin(['Polygon','MultiPolygon'])]
        if not pk.empty: pk.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=2)

    if buildings and bld is not None and not bld.empty:
        bp = bld[bld.geometry.type.isin(['Polygon','MultiPolygon'])].copy()
        if landmarks_only: bp = _filter_landmarks(bp, min_levels=min_levels)
        if project_metric:
            _plot_extruded(ax, bp,
                           extrude_depth=float(extrude_depth), sun_azimuth=float(sun_azimuth), sun_elevation=float(sun_elevation),
                           fill_color=(building_color or THEME.get('building_fill','#1f1f1f')),
                           shadow_color=(shadow_color or THEME.get('building_shadow','#000000')),
                           alpha_fill=float(building_opacity), base_alpha_shadow=0.25,
                           z_base=3, shadow_alpha_override=(None if shadow_alpha is None else float(shadow_alpha)))
        else:
            _plot_flat(ax, bp, fill_color=(building_color or THEME.get('building_fill','#1f1f1f')), alpha_fill=float(building_opacity), zorder=3)

    edge_colors = get_edge_colors_by_type(G)
    edge_widths = get_edge_widths_by_type(G)
    ox.plot_graph(G, ax=ax, bgcolor=THEME['bg'], node_size=0, node_color=None, node_zorder=0,
                  edge_color=edge_colors, edge_linewidth=edge_widths, show=False, close=False)

    if pois and pois_gdf is not None and not pois_gdf.empty:
        try:
            df = pois_gdf.copy(); pts = df[df.geometry.type=='Point']
            if not pts.empty:
                xs = pts.geometry.x.values; ys = pts.geometry.y.values
                c = THEME.get('poi_color','#D33F49') if (not poi_color) else poi_color
                ax.scatter(xs, ys, s=float(poi_size or 28.0), c=c, marker='^', edgecolors='none', zorder=12)
        except Exception: pass

    create_gradient_fade(ax, THEME['gradient_color'], 'bottom', 10)
    create_gradient_fade(ax, THEME['gradient_color'], 'top', 10)

    if FONTS:
        font_main = FontProperties(fname=FONTS['bold'], size=60)
        font_sub  = FontProperties(fname=FONTS['light'], size=22)
        font_crd  = FontProperties(fname=FONTS['regular'], size=14)
    else:
        font_main = FontProperties(family='monospace', weight='bold', size=60)
        font_sub  = FontProperties(family='monospace', weight='normal', size=22)
        font_crd  = FontProperties(family='monospace', size=14)

    spaced_city = " ".join(list((city or '').upper()))
    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes, color=THEME['text'], ha='center', fontproperties=font_main, zorder=11)
    ax.text(0.5, 0.10, (country or '').upper(), transform=ax.transAxes, color=THEME['text'], ha='center', fontproperties=font_sub, zorder=11)

    lat, lon = point
    coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    if lon < 0: coords = coords.replace("E","W")
    ax.text(0.5, 0.07, coords, transform=ax.transAxes, color=THEME['text'], alpha=0.7, ha='center', fontproperties=font_crd, zorder=11)
    ax.plot([0.4,0.6], [0.125,0.125], transform=ax.transAxes, color=THEME['text'], linewidth=1, zorder=11)

    print(f"Saving to {output_file}...")
    plt.savefig(output_file, dpi=dpi, facecolor=THEME['bg'])
    plt.close(); print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    print("""
City Map Poster Generator (3D buildings + POIs + sun elevation + color overrides)
Usage:
  python create_map_poster.py --city <city> --country <country> [options]
""")


def list_themes():
    themes = get_available_themes()
    if not themes: print("No themes found in 'themes/' directory."); return
    print("\nAvailable Themes:\n"+"-"*40)
    for name in themes:
        path = os.path.join(THEMES_DIR, f"{name}.json")
        try:
            with open(path, 'r', encoding='utf-8') as f: data = json.load(f)
            display_name = data.get('name', name)
        except Exception: display_name = name
        print(f"  {name}\n    {display_name}\n")


def list_quality_levels():
    print("\nAvailable Quality Levels:\n"+"-"*40)
    for level, specs in QUALITY_PRESETS.items():
        w,h = specs['figsize']; dpi = specs['dpi']
        print(f"  {level}\n    Portrait: {w}x{h} in @ {dpi} DPI\n    Landscape: {h}x{w} in @ {dpi} DPI\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters (3D-like buildings, POIs, sun controls)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--city','-c', type=str)
    parser.add_argument('--country','-C', type=str)
    parser.add_argument('--theme','-t', type=str, default='feature_based')
    parser.add_argument('--distance','-d', type=int, default=29000, help='Half-size of the short canvas side in meters')
    parser.add_argument('--quality','-q', type=str, default='standard', choices=list(QUALITY_PRESETS.keys()))
    parser.add_argument('--landscape','-L', action='store_true', default=False)
    parser.add_argument('--list-themes', action='store_true')
    parser.add_argument('--project-metric', action='store_true', default=False)
    parser.add_argument('--list-quality', action='store_true')
    parser.add_argument('--latitude','-la', type=float)
    parser.add_argument('--longitude','-lo', type=float)

    # Buildings
    parser.add_argument('--buildings', action='store_true', default=False)
    parser.add_argument('--landmarks-only', action='store_true', default=False)
    parser.add_argument('--extrude-depth', type=float, default=12.0)
    parser.add_argument('--min-levels', type=float, default=5)
    parser.add_argument('--building-opacity', type=float, default=0.9)
    parser.add_argument('--building-color', type=str, default=None)
    parser.add_argument('--shadow-color', type=str, default=None)

    # Sun
    parser.add_argument('--sun-azimuth', type=float, default=45.0)
    parser.add_argument('--sun-elevation', type=float, default=45.0)
    parser.add_argument('--shadow-alpha', type=float, default=None)

    # POIs
    parser.add_argument('--pois', action='store_true', default=False)
    parser.add_argument('--poi-labels', action='store_true', default=False)
    parser.add_argument('--poi-max', type=int, default=20)
    parser.add_argument('--poi-size', type=float, default=28.0)
    parser.add_argument('--poi-color', type=str, default=None)
    parser.add_argument('--poi-tags', type=str, default=None)

    # Overpass/OSMnx tuning overrides
    parser.add_argument('--overpass-max-area', type=float, default=None, help='Override OSMnx max_query_area_size (m^2)')
    parser.add_argument('--overpass-memory', type=int, default=None, help='Overpass memory (MB) if supported')
    parser.add_argument('--requests-timeout', type=int, default=None, help='HTTP/Overpass timeout (seconds)')

    args = parser.parse_args()

    if args.list_themes:
        list_themes(); raise SystemExit(0)
    if args.list_quality:
        list_quality_levels(); raise SystemExit(0)

    city_country = args.city and args.country
    latlon = (args.latitude is not None and args.longitude is not None)
    if not city_country and not latlon:
        print("Error: --city and --country or --latitude and --longitude are required.\n"); print_examples(); raise SystemExit(1)

    available = get_available_themes()
    if args.theme not in available:
        print(f"Error: Theme '{args.theme}' not found. Available: {', '.join(available)}"); raise SystemExit(1)

    # Apply CLI overrides to OSMnx settings
    try:
        if args.overpass_max_area is not None:
            ox.settings.max_query_area_size = int(args.overpass_max_area)
        if args.overpass_memory is not None and hasattr(ox.settings, 'overpass_memory'):
            ox.settings.overpass_memory = int(args.overpass_memory)
        if args.requests_timeout is not None:
            ox.settings.requests_timeout = int(args.requests_timeout)
    except Exception:
        pass

    preset = QUALITY_PRESETS[args.quality]
    base_w, base_h = preset['figsize']; dpi = preset['dpi']
    figsize = (base_h, base_w) if args.landscape else (base_w, base_h)

    THEME = load_theme(args.theme)
    load_fonts()

    try:
        if latlon:
            point = (args.latitude, args.longitude)
        else:
            from geopy.geocoders import Nominatim
            loc = Nominatim(user_agent='city_map_poster').geocode(f"{args.city}, {args.country}")
            point = (loc.latitude, loc.longitude)

        output_file = generate_output_filename(
            args.city or '', args.theme,
            distance=args.distance, quality=args.quality, landscape=args.landscape, project_metric=args.project_metric,
            buildings=args.buildings, landmarks_only=args.landmarks_only,
            extrude_depth=args.extrude_depth, min_levels=args.min_levels,
            sun_azimuth=args.sun_azimuth, sun_elevation=args.sun_elevation,
            pois=args.pois, poi_tags_custom=bool(args.poi_tags))

        create_poster(
            args.city or '', args.country or '', point, args.distance, output_file,
            figsize=figsize, dpi=dpi, project_metric=args.project_metric, landscape=args.landscape,
            buildings=args.buildings, landmarks_only=args.landmarks_only,
            extrude_depth=args.extrude_depth, min_levels=args.min_levels, building_opacity=args.building_opacity,
            sun_azimuth=args.sun_azimuth, sun_elevation=args.sun_elevation, shadow_alpha=args.shadow_alpha,
            pois=args.pois, poi_labels=args.poi_labels, poi_max=args.poi_max, poi_size=args.poi_size,
            poi_color=args.poi_color, poi_tags=args.poi_tags,
            building_color=args.building_color, shadow_color=args.shadow_color)
        raise SystemExit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback; traceback.print_exc(); raise SystemExit(1)
