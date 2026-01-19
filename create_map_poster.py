
import osmnx as ox
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.colors as mcolors
import numpy as np
from geopy.geocoders import Nominatim
from tqdm import tqdm
import time
import json
import os
from datetime import datetime
import argparse

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

# Quality presets: (figsize_width, figsize_height) PORTRAIT baseline (inches) and dpi
QUALITY_PRESETS = {
    'standard': {'figsize': (12, 16), 'dpi': 300, 'description': 'Standard – digital/small prints'},
    'high':     {'figsize': (16, 21), 'dpi': 400, 'description': 'High – medium/large prints'},
    'ultra':    {'figsize': (18, 24), 'dpi': 600, 'description': 'Ultra – professional/large format'},
}

def load_fonts():
    fonts = {
        'bold': os.path.join(FONTS_DIR, 'Roboto-Bold.ttf'),
        'regular': os.path.join(FONTS_DIR, 'Roboto-Regular.ttf'),
        'light': os.path.join(FONTS_DIR, 'Roboto-Light.ttf')
    }
    for _, path in fonts.items():
        if not os.path.exists(path):
            print(f"⚠ Font not found: {path}")
            return None
    return fonts

FONTS = load_fonts()

def generate_output_filename(city, theme_name, *, distance, quality, landscape, project_metric):
    """Create filename reflecting key options.
    Pattern: <city>_<theme>_<l|p>_q<quality>_d<DIST>_pm_<t|f>_<timestamp>.png
    Examples: hamburg_warm_beige_l_qstandard_d2000_pm_t_20260119_092155.png
              hamburg_noir_p_qhigh_d10000_pm_f_20260119_092201.png
    """
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug  = city.lower().replace(' ', '_')
    theme_slug = theme_name
    orient = 'l' if landscape else 'p'
    pm_flag = 't' if project_metric else 'f'
    fname = f"{city_slug}_{theme_slug}_{orient}_q{quality}_d{distance}_pm_{pm_flag}_{ts}.png"
    return os.path.join(POSTERS_DIR, fname)

def get_available_themes():
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []
    return [f[:-5] for f in sorted(os.listdir(THEMES_DIR)) if f.endswith('.json')]

def load_theme(theme_name="feature_based"):
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")
    if not os.path.exists(theme_file):
        print(f"⚠ Theme file '{theme_file}' not found. Using default feature_based theme.")
        return {
            "name": "Feature-Based Shading",
            "bg": "#FFFFFF",
            "text": "#000000",
            "gradient_color": "#FFFFFF",
            "water": "#C0C0C0",
            "parks": "#F0F0F0",
            "road_motorway": "#0A0A0A",
            "road_primary": "#1A1A1A",
            "road_secondary": "#2A2A2A",
            "road_tertiary": "#3A3A3A",
            "road_residential": "#4A4A4A",
            "road_default": "#3A3A3A"
        }
    with open(theme_file, 'r') as f:
        theme = json.load(f)
    print(f"✓ Loaded theme: {theme.get('name', theme_name)}")
    if 'description' in theme:
        print(f"  {theme['description']}")
    return theme

THEME = None

def create_gradient_fade(ax, color, location='bottom', zorder=10):
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))
    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]; my_colors[:, 1] = rgb[1]; my_colors[:, 2] = rgb[2]
    if location == 'bottom':
        my_colors[:, 3] = np.linspace(1, 0, 256); extent_y_start, extent_y_end = 0, 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256); extent_y_start, extent_y_end = 0.75, 1.0
    custom_cmap = mcolors.ListedColormap(my_colors)
    xlim = ax.get_xlim(); ylim = ax.get_ylim(); y_range = ylim[1] - ylim[0]
    y_bottom = ylim[0] + y_range * extent_y_start
    y_top    = ylim[0] + y_range * extent_y_end
    ax.imshow(gradient, extent=[xlim[0], xlim[1], y_bottom, y_top], aspect='auto', cmap=custom_cmap,
              zorder=zorder, origin='lower')

def get_edge_colors_by_type(G):
    edge_colors = []
    for _, _, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        if highway in ['motorway', 'motorway_link']:
            color = THEME['road_motorway']
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            color = THEME['road_primary']
        elif highway in ['secondary', 'secondary_link']:
            color = THEME['road_secondary']
        elif highway in ['tertiary', 'tertiary_link']:
            color = THEME['road_tertiary']
        elif highway in ['residential', 'living_street', 'unclassified']:
            color = THEME['road_residential']
        else:
            color = THEME['road_default']
        edge_colors.append(color)
    return edge_colors

def get_edge_widths_by_type(G):
    edge_widths = []
    for _, _, data in G.edges(data=True):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'
        if highway in ['motorway', 'motorway_link']:
            width = 1.2
        elif highway in ['trunk', 'trunk_link', 'primary', 'primary_link']:
            width = 1.0
        elif highway in ['secondary', 'secondary_link']:
            width = 0.8
        elif highway in ['tertiary', 'tertiary_link']:
            width = 0.6
        else:
            width = 0.4
        edge_widths.append(width)
    return edge_widths

def get_coordinates(city, country):
    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster")
    time.sleep(1)
    location = geolocator.geocode(f"{city}, {country}")
    if location:
        print(f"✓ Found: {location.address}")
        print(f"✓ Coordinates: {location.latitude}, {location.longitude}")
        return (location.latitude, location.longitude)
    raise ValueError(f"Could not find coordinates for {city}, {country}")

def create_poster(city, country, point, dist, output_file, *, figsize=(12, 16), dpi=300,
                  project_metric=True, landscape=False):
    print(f"\nGenerating map for {city}, {country}...")
    px = (int(round(figsize[0]*dpi)), int(round(figsize[1]*dpi)))
    print(f"Canvas: {figsize[0]}x{figsize[1]} inches @ {dpi} DPI -> {px[0]}x{px[1]} px | orientation={'landscape' if landscape else 'portrait'}")

    # 1. Fetch data
    with tqdm(total=3, desc="Fetching map data", unit="step",
              bar_format='{l_bar}{bar}{n_fmt}/{total_fmt}') as pbar:
        pbar.set_description("Downloading street network")
        G = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all')
        pbar.update(1)
        time.sleep(0.5)
        pbar.set_description("Downloading water features")
        try:
            water = ox.features_from_point(point, tags={'natural': 'water', 'waterway': 'riverbank'}, dist=dist)
        except Exception:
            water = None
        pbar.update(1)
        time.sleep(0.3)
        pbar.set_description("Downloading parks/green spaces")
        try:
            parks = ox.features_from_point(point, tags={'leisure': 'park', 'landuse': 'grass'}, dist=dist)
        except Exception:
            parks = None
        pbar.update(1)

    if project_metric:
        G = ox.project_graph(G)
        graph_crs = G.graph.get("crs")
        ax.set_aspect('equal', adjustable='datalim')
        if water is not None and not water.empty:
            water = water.to_crs(graph_crs)
        if parks is not None and not parks.empty:
            parks = parks.to_crs(graph_crs)
    print("✓ All data downloaded successfully!")

    # 2. Plot (single source of truth: figsize)
    print("Rendering map...")
    fig, ax = plt.subplots(figsize=figsize, facecolor=THEME['bg'])
    ax.set_facecolor(THEME['bg'])
    ax.set_position([0, 0, 1, 1])

    # 3. Layers
    if water is not None and not water.empty:
        water_polys = water[water.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not water_polys.empty:
            water_polys.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=1)
    if parks is not None and not parks.empty:
        parks_polys = parks[parks.geometry.type.isin(['Polygon', 'MultiPolygon'])]
        if not parks_polys.empty:
            parks_polys.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=2)

    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(G)
    edge_widths = get_edge_widths_by_type(G)
    ox.plot_graph(
        G, ax=ax, bgcolor=THEME['bg'], node_size=0, node_color=None, node_zorder=0,
        edge_color=edge_colors, edge_linewidth=edge_widths, show=False, close=False
    )

    create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)

    # Typography
    if FONTS:
        font_main = FontProperties(fname=FONTS['bold'], size=60)
        font_sub  = FontProperties(fname=FONTS['light'], size=22)
        font_coords = FontProperties(fname=FONTS['regular'], size=14)
    else:
        font_main = FontProperties(family='monospace', weight='bold', size=60)
        font_sub  = FontProperties(family='monospace', weight='normal', size=22)
        font_coords = FontProperties(family='monospace', size=14)

    spaced_city = " ".join(list(city.upper()))
    ax.text(0.5, 0.14, spaced_city, transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_main, zorder=11)
    ax.text(0.5, 0.10, country.upper(), transform=ax.transAxes,
            color=THEME['text'], ha='center', fontproperties=font_sub, zorder=11)

    lat, lon = point
    coords = f"{lat:.4f}° N / {lon:.4f}° E" if lat >= 0 else f"{abs(lat):.4f}° S / {lon:.4f}° E"
    if lon < 0:
        coords = coords.replace("E", "W")
    ax.text(0.5, 0.07, coords, transform=ax.transAxes,
            color=THEME['text'], alpha=0.7, ha='center', fontproperties=font_coords, zorder=11)

    ax.plot([0.4, 0.6], [0.125, 0.125], transform=ax.transAxes,
            color=THEME['text'], linewidth=1, zorder=11)

    print(f"Saving to {output_file}...")
    plt.savefig(output_file, dpi=dpi, facecolor=THEME['bg'])
    plt.close()
    print(f"✓ Done! Poster saved as {output_file}")


def print_examples():
    print("""
City Map Poster Generator
=========================
Usage:
  python create_map_poster.py --city <city> --country <country> [options]
Examples:
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000
  python create_map_poster.py --list-themes
Options:
  --city, -c         City name (required)
  --country, -C      Country name (required)
  --theme, -t        Theme ID (default: feature_based)
  --distance, -d     Map radius in meters (default: 29000)
  --quality, -q      Output quality: standard|high|ultra
  --landscape, -L    Landscape orientation (swaps inches)
  --project-metric   Reproject to local metric CRS
  --list-themes      List themes
  --list-quality     List quality presets
""")

def list_themes():
    themes = get_available_themes()
    if not themes:
        print("No themes found in 'themes/' directory.")
        return
    print("\nAvailable Themes:\n" + "-"*60)
    for name in themes:
        path = os.path.join(THEMES_DIR, f"{name}.json")
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            display_name = data.get('name', name)
        except Exception:
            display_name = name
        print(f"  {name}\n    {display_name}\n")

def list_quality_levels():
    print("\nAvailable Quality Levels:\n" + "-"*60)
    for level, specs in QUALITY_PRESETS.items():
        w, h = specs['figsize']; dpi = specs['dpi']
        print(f"  {level}\n    {specs['description']}\n    Portrait:  {w}x{h} in @ {dpi} DPI ({int(w*dpi)}x{int(h*dpi)} px)\n    Landscape: {h}x{w} in @ {dpi} DPI ({int(h*dpi)}x{int(w*dpi)} px)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster.py --list-themes
 """
    )
    parser.add_argument('--city', '-c', type=str, help='City name')
    parser.add_argument('--country', '-C', type=str, help='Country name')
    parser.add_argument('--theme', '-t', type=str, default='feature_based', help='Theme name (default: feature_based)')
    parser.add_argument('--distance', '-d', type=int, default=29000, help='Map radius in meters (default: 29000)')
    parser.add_argument('--quality', '-q', type=str, default='standard', choices=list(QUALITY_PRESETS.keys()),
                        help='Output quality level (default: standard)')
    parser.add_argument('--landscape', '-L', action='store_true', default=False, help='Render in landscape orientation')
    parser.add_argument('--list-themes', action='store_true', help='List all available themes')
    parser.add_argument('--project-metric', action='store_true', default=False, help='Reproject to a local metric CRS')
    parser.add_argument('--list-quality', action='store_true', help='List available output quality presets and exit')

    args = parser.parse_args()

    if len(os.sys.argv) == 1:
        print_examples(); os.sys.exit(0)

    if args.list_themes:
        list_themes(); os.sys.exit(0)
    if args.list_quality:
        list_quality_levels(); os.sys.exit(0)

    if not args.city or not args.country:
        print("Error: --city and --country are required.\n"); print_examples(); os.sys.exit(1)

    available = get_available_themes()
    if args.theme not in available:
        print(f"Error: Theme '{args.theme}' not found.\nAvailable: {', '.join(available)}"); os.sys.exit(1)

    print("="*50); print("City Map Poster Generator"); print("="*50)

    preset = QUALITY_PRESETS[args.quality]
    base_w, base_h = preset['figsize']; dpi = preset['dpi']
    figsize = (base_h, base_w) if args.landscape else (base_w, base_h)

    print(f"Quality: {args.quality.upper()} | Orientation: {'landscape' if args.landscape else 'portrait'} | Project-Metric: {args.project_metric}")

    THEME = load_theme(args.theme)
    try:
        coords = get_coordinates(args.city, args.country)
        output_file = generate_output_filename(
            args.city, args.theme,
            distance=args.distance, quality=args.quality,
            landscape=args.landscape, project_metric=args.project_metric,
        )
        create_poster(
            args.city, args.country, coords, args.distance, output_file,
            figsize=figsize, dpi=dpi, project_metric=args.project_metric,
            landscape=args.landscape,
        )
        print("\n" + "="*50); print("✓ Poster generation complete!"); print("="*50)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback; traceback.print_exc(); os.sys.exit(1)
