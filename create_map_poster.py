#!/usr/bin/env python3
"""
City Map Poster Generator

This module generates beautiful, minimalist map posters for any city in the world.
It fetches OpenStreetMap data using OSMnx, applies customizable themes, and creates
high-quality poster-ready images with roads, water features, parks, buildings, and POIs.

Integrates upstream features (caching, i18n, font management, output formats) with
custom PR features (buildings, POIs, sun controls, quality presets, landscape mode).
"""

import argparse
import asyncio
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
from geopandas import GeoDataFrame
from geopy.geocoders import Nominatim
from lat_lon_parser import parse
from matplotlib.font_manager import FontProperties
from networkx import MultiDiGraph
from shapely import affinity
from shapely.geometry import Point
from shapely.geometry import box as shp_box
from tqdm import tqdm

from font_management import load_fonts


class CacheError(Exception):
    """Raised when a cache operation fails."""


CACHE_DIR_PATH = os.environ.get("CACHE_DIR", "cache")
CACHE_DIR = Path(CACHE_DIR_PATH)
CACHE_DIR.mkdir(exist_ok=True)

THEMES_DIR = "themes"
FONTS_DIR = "fonts"
POSTERS_DIR = "posters"

FILE_ENCODING = "utf-8"

QUALITY_PRESETS = {
    'standard': {'figsize': (12, 16), 'dpi': 300},
    'high':     {'figsize': (16, 21), 'dpi': 400},
    'ultra':    {'figsize': (18, 24), 'dpi': 600},
}

FONTS = load_fonts()

# Load theme (set later via CLI)
THEME = dict[str, str]()

# --- OSMnx runtime settings (overridable via env and CLI) ---
try:
    _DEFAULT_MAX_AREA = 50_000 * 50_000  # ~50km x 50km in m^2
    ox.settings.max_query_area_size = int(os.getenv("MAPTOP_MAX_QUERY_AREA_SIZE", _DEFAULT_MAX_AREA))
    if hasattr(ox.settings, "overpass_memory"):
        ox.settings.overpass_memory = int(os.getenv("MAPTOP_OVERPASS_MEMORY", "4096"))
    ox.settings.requests_timeout = int(os.getenv("MAPTOP_REQUESTS_TIMEOUT", "180"))
    ox.settings.log_console = bool(int(os.getenv("MAPTOP_LOG_CONSOLE", "1")))
except Exception:
    pass


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> str:
    safe = key.replace(os.sep, "_")
    return os.path.join(CACHE_DIR, f"{safe}.pkl")


def cache_get(key: str):
    try:
        path = _cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CacheError(f"Cache read failed: {e}") from e


def cache_set(key: str, value):
    try:
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        path = _cache_path(key)
        with open(path, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        raise CacheError(f"Cache write failed: {e}") from e


# ---------------------------------------------------------------------------
# OSMnx bbox compatibility helpers (from PR integrations)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# i18n helpers (from main)
# ---------------------------------------------------------------------------

def is_latin_script(text):
    """
    Check if text is primarily Latin script.
    Used to determine if letter-spacing should be applied to city names.
    """
    if not text:
        return True

    latin_count = 0
    total_alpha = 0

    for char in text:
        if char.isalpha():
            total_alpha += 1
            if ord(char) < 0x250:
                latin_count += 1

    if total_alpha == 0:
        return True

    return (latin_count / total_alpha) > 0.8


# ---------------------------------------------------------------------------
# Filename / theme helpers
# ---------------------------------------------------------------------------

def generate_output_filename(city, theme_name, output_format="png"):
    """Generate unique output filename with city, theme, and datetime."""
    if not os.path.exists(POSTERS_DIR):
        os.makedirs(POSTERS_DIR)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_slug = city.lower().replace(" ", "_")
    ext = output_format.lower()
    filename = f"{city_slug}_{theme_name}_{timestamp}.{ext}"
    return os.path.join(POSTERS_DIR, filename)


def get_available_themes():
    """Scans the themes directory and returns a list of available theme names."""
    if not os.path.exists(THEMES_DIR):
        os.makedirs(THEMES_DIR)
        return []

    themes = []
    for file in sorted(os.listdir(THEMES_DIR)):
        if file.endswith(".json"):
            theme_name = file[:-5]
            themes.append(theme_name)
    return themes


def load_theme(theme_name="terracotta"):
    """Load theme from JSON file in themes directory."""
    theme_file = os.path.join(THEMES_DIR, f"{theme_name}.json")

    if not os.path.exists(theme_file):
        print(f"Warning: Theme file '{theme_file}' not found. Using default terracotta theme.")
        return {
            "name": "Terracotta",
            "description": "Mediterranean warmth - burnt orange and clay tones on cream",
            "bg": "#F5EDE4",
            "text": "#8B4513",
            "gradient_color": "#F5EDE4",
            "water": "#A8C4C4",
            "parks": "#E8E0D0",
            "road_motorway": "#A0522D",
            "road_primary": "#B8653A",
            "road_secondary": "#C9846A",
            "road_tertiary": "#D9A08A",
            "road_residential": "#E5C4B0",
            "road_default": "#D9A08A",
        }

    with open(theme_file, "r", encoding=FILE_ENCODING) as f:
        theme = json.load(f)

    # Ensure building/POI theme keys have defaults
    theme.setdefault("building_fill", "#1f1f1f")
    theme.setdefault("building_shadow", "#000000")
    theme.setdefault("poi_color", "#D33F49")
    theme.setdefault("poi_label_color", theme.get("text", "#000000"))

    print(f"Loaded theme: {theme.get('name', theme_name)}")
    if "description" in theme:
        print(f"  {theme['description']}")
    return theme


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def create_gradient_fade(ax, color, location="bottom", zorder=10):
    """Creates a fade effect at the top or bottom of the map."""
    vals = np.linspace(0, 1, 256).reshape(-1, 1)
    gradient = np.hstack((vals, vals))

    rgb = mcolors.to_rgb(color)
    my_colors = np.zeros((256, 4))
    my_colors[:, 0] = rgb[0]
    my_colors[:, 1] = rgb[1]
    my_colors[:, 2] = rgb[2]

    if location == "bottom":
        my_colors[:, 3] = np.linspace(1, 0, 256)
        extent_y_start = 0
        extent_y_end = 0.25
    else:
        my_colors[:, 3] = np.linspace(0, 1, 256)
        extent_y_start = 0.75
        extent_y_end = 1.0

    custom_cmap = mcolors.ListedColormap(my_colors)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]

    y_bottom = ylim[0] + y_range * extent_y_start
    y_top = ylim[0] + y_range * extent_y_end

    ax.imshow(
        gradient,
        extent=[xlim[0], xlim[1], y_bottom, y_top],
        aspect="auto",
        cmap=custom_cmap,
        zorder=zorder,
        origin="lower",
    )


def get_edge_colors_by_type(g):
    """Assigns colors to edges based on road type hierarchy."""
    edge_colors = []

    for _u, _v, data in g.edges(data=True):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'

        if highway in ["motorway", "motorway_link"]:
            color = THEME["road_motorway"]
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            color = THEME["road_primary"]
        elif highway in ["secondary", "secondary_link"]:
            color = THEME["road_secondary"]
        elif highway in ["tertiary", "tertiary_link"]:
            color = THEME["road_tertiary"]
        elif highway in ["residential", "living_street", "unclassified"]:
            color = THEME["road_residential"]
        else:
            color = THEME['road_default']

        edge_colors.append(color)

    return edge_colors


def get_edge_widths_by_type(g):
    """Assigns line widths to edges based on road type."""
    edge_widths = []

    for _u, _v, data in g.edges(data=True):
        highway = data.get('highway', 'unclassified')
        if isinstance(highway, list):
            highway = highway[0] if highway else 'unclassified'

        if highway in ["motorway", "motorway_link"]:
            width = 1.2
        elif highway in ["trunk", "trunk_link", "primary", "primary_link"]:
            width = 1.0
        elif highway in ["secondary", "secondary_link"]:
            width = 0.8
        elif highway in ["tertiary", "tertiary_link"]:
            width = 0.6
        else:
            width = 0.4

        edge_widths.append(width)

    return edge_widths


# ---------------------------------------------------------------------------
# Geocoding (from main — with caching + asyncio safety)
# ---------------------------------------------------------------------------

def get_coordinates(city, country):
    """Fetches coordinates for a given city and country using geopy, with caching."""
    coords_key = f"coords_{city.lower()}_{country.lower()}"
    cached = cache_get(coords_key)
    if cached:
        print(f"Using cached coordinates for {city}, {country}")
        return cached

    print("Looking up coordinates...")
    geolocator = Nominatim(user_agent="city_map_poster", timeout=10)
    time.sleep(1)

    try:
        location = geolocator.geocode(f"{city}, {country}")
    except Exception as e:
        raise ValueError(f"Geocoding failed for {city}, {country}: {e}") from e

    if asyncio.iscoroutine(location):
        try:
            location = asyncio.run(location)
        except RuntimeError as exc:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Geocoder returned a coroutine while an event loop is already running. "
                    "Run this script in a synchronous environment."
                ) from exc
            location = loop.run_until_complete(location)

    if location:
        addr = getattr(location, "address", None)
        if addr:
            print(f"Found: {addr}")
        else:
            print("Found location (address not available)")
        print(f"Coordinates: {location.latitude}, {location.longitude}")
        try:
            cache_set(coords_key, (location.latitude, location.longitude))
        except CacheError as e:
            print(e)
        return (location.latitude, location.longitude)

    raise ValueError(f"Could not find coordinates for {city}, {country}")


# ---------------------------------------------------------------------------
# Crop / projection helpers (from main)
# ---------------------------------------------------------------------------

def get_crop_limits(g_proj, center_lat_lon, fig, dist):
    """Crop inward to preserve aspect ratio while guaranteeing full coverage."""
    lat, lon = center_lat_lon

    center = (
        ox.projection.project_geometry(
            Point(lon, lat),
            crs="EPSG:4326",
            to_crs=g_proj.graph["crs"]
        )[0]
    )
    center_x, center_y = center.x, center.y

    fig_width, fig_height = fig.get_size_inches()
    aspect = fig_width / fig_height

    half_x = dist
    half_y = dist

    if aspect > 1:
        half_y = half_x / aspect
    else:
        half_x = half_y * aspect

    return (
        (center_x - half_x, center_x + half_x),
        (center_y - half_y, center_y + half_y),
    )


# ---------------------------------------------------------------------------
# Data fetching with caching (from main)
# ---------------------------------------------------------------------------

def fetch_graph(point, dist) -> MultiDiGraph | None:
    """Fetch street network graph from OpenStreetMap with caching."""
    lat, lon = point
    graph_key = f"graph_{lat}_{lon}_{dist}"
    cached = cache_get(graph_key)
    if cached is not None:
        print("Using cached street network")
        return cast(MultiDiGraph, cached)

    try:
        g = ox.graph_from_point(point, dist=dist, dist_type='bbox', network_type='all', truncate_by_edge=True)
        time.sleep(0.5)
        try:
            cache_set(graph_key, g)
        except CacheError as e:
            print(e)
        return g
    except Exception as e:
        print(f"OSMnx error while fetching graph: {e}")
        return None


def fetch_features(point, dist, tags, name) -> GeoDataFrame | None:
    """Fetch geographic features from OpenStreetMap with caching."""
    lat, lon = point
    tag_str = "_".join(tags.keys())
    features_key = f"{name}_{lat}_{lon}_{dist}_{tag_str}"
    cached = cache_get(features_key)
    if cached is not None:
        print(f"Using cached {name}")
        return cast(GeoDataFrame, cached)

    try:
        data = ox.features_from_point(point, tags=tags, dist=dist)
        time.sleep(0.3)
        try:
            cache_set(features_key, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching features: {e}")
        return None


def fetch_features_bbox(n, s, e, w, tags, name) -> GeoDataFrame | None:
    """Fetch features using bbox (for buildings/POIs in bbox mode) with caching."""
    features_key = f"{name}_{n}_{s}_{e}_{w}_{'_'.join(tags.keys())}"
    cached = cache_get(features_key)
    if cached is not None:
        print(f"Using cached {name}")
        return cast(GeoDataFrame, cached)

    try:
        data = _features_from_bbox_compat(n, s, e, w, tags=tags)
        time.sleep(0.3)
        try:
            cache_set(features_key, data)
        except CacheError as e:
            print(e)
        return data
    except Exception as e:
        print(f"OSMnx error while fetching {name}: {e}")
        return None


# ---------------------------------------------------------------------------
# Building rendering helpers (from PR integrations)
# ---------------------------------------------------------------------------

def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _shadow_offset_from_azimuth(depth_m, azimuth_deg):
    rad = np.deg2rad(azimuth_deg)
    return float(depth_m * np.sin(rad)), float(-depth_m * np.cos(rad))


def _effective_shadow_depth(extrude_depth, sun_elevation_deg):
    elev = _clip(float(sun_elevation_deg), 1.0, 89.0)
    tanv = np.tan(np.deg2rad(elev))
    eff = extrude_depth * (1.0 / max(tanv, 0.2))
    return float(_clip(eff, extrude_depth * 0.4, extrude_depth * 4.0))


def _shadow_alpha_for_elevation(base_alpha, sun_elevation_deg):
    elev = _clip(float(sun_elevation_deg), 0.0, 90.0)
    return float(base_alpha * (0.5 + 0.5 * (1.0 - elev / 90.0)))


def _filter_landmarks(bld, min_levels=5):
    cands = []
    for col in ['name', 'wikidata', 'brand']:
        if col in bld.columns:
            cands.append(bld[col].notna())
    if 'tourism' in bld.columns:
        cands.append(bld['tourism'].isin(['attraction', 'museum', 'gallery', 'viewpoint', 'theme_park', 'zoo']))
    if 'historic' in bld.columns:
        cands.append(bld['historic'].notna())
    if 'building:levels' in bld.columns:
        try:
            lvl = bld['building:levels'].astype(float)
            cands.append(lvl >= float(min_levels))
        except Exception:
            pass
    if cands:
        mask = np.logical_or.reduce(cands)
        return bld[mask]
    return bld


def _plot_extruded(ax, gdf, extrude_depth=12, sun_azimuth=45, sun_elevation=45,
                   fill_color="#1f1f1f", shadow_color="#000000", alpha_fill=0.9,
                   base_alpha_shadow=0.25, z_base=3, shadow_alpha_override=None):
    if gdf is None or gdf.empty:
        return
    polys = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    if polys.empty:
        return
    eff_depth = _effective_shadow_depth(float(extrude_depth), float(sun_elevation))
    dx, dy = _shadow_offset_from_azimuth(eff_depth, sun_azimuth)
    alpha_shadow = (
        _shadow_alpha_for_elevation(base_alpha_shadow, sun_elevation)
        if shadow_alpha_override is None
        else float(shadow_alpha_override)
    )
    shadow = polys.copy()
    shadow['geometry'] = shadow['geometry'].apply(lambda g: affinity.translate(g, xoff=dx, yoff=dy))
    shadow.plot(ax=ax, facecolor=shadow_color, edgecolor='none', alpha=alpha_shadow, zorder=z_base)
    polys.plot(ax=ax, facecolor=fill_color, edgecolor='none', alpha=alpha_fill, zorder=z_base + 1)


def _plot_flat(ax, gdf, fill_color="#1f1f1f", alpha_fill=0.8, z_base=3):
    if gdf is None or gdf.empty:
        return
    polys = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]
    if polys.empty:
        return
    polys.plot(ax=ax, facecolor=fill_color, edgecolor='none', alpha=alpha_fill, zorder=z_base)


# ---------------------------------------------------------------------------
# POI helpers (from PR integrations)
# ---------------------------------------------------------------------------

def _default_poi_tags():
    return {
        'tourism': ['attraction', 'museum', 'gallery', 'viewpoint'],
        'historic': True,
        'amenity': ['theatre', 'arts_centre', 'townhall', 'courthouse', 'library', 'university', 'place_of_worship'],
        'building': ['church', 'cathedral', 'castle', 'stadium', 'public', 'civic', 'city_hall', 'train_station'],
        'leisure': ['stadium'],
        'man_made': ['lighthouse', 'tower'],
    }


def _parse_poi_tags(tag_str):
    if not tag_str:
        return None
    tags = {}
    for part in [p.strip() for p in tag_str.split(';') if p.strip()]:
        if '=' not in part:
            continue
        k, v = part.split('=', 1)
        k, v = k.strip(), v.strip()
        if v.lower() in ('true', 'yes', '1'):
            tags[k] = True
        else:
            vals = [vv.strip() for vv in v.split(',') if vv.strip()]
            tags[k] = vals[0] if len(vals) == 1 else vals
    return tags


# ---------------------------------------------------------------------------
# Main poster creation
# ---------------------------------------------------------------------------

def create_poster(
    city,
    country,
    point,
    dist,
    output_file,
    output_format="png",
    width=12,
    height=16,
    country_label=None,
    name_label=None,
    display_city=None,
    display_country=None,
    fonts=None,
    # PR integration features
    buildings=False,
    landmarks_only=False,
    extrude_depth=12,
    min_levels=5,
    building_opacity=0.9,
    sun_azimuth=45,
    sun_elevation=45,
    shadow_alpha=None,
    pois=False,
    poi_labels=False,
    poi_max=20,
    poi_size=28,
    poi_color=None,
    poi_tags=None,
    building_color=None,
    shadow_color=None,
):
    # Handle display names for i18n support
    display_city = display_city or name_label or city
    display_country = display_country or country_label or country

    print(f"\nGenerating map for {city}, {country}...")

    # Progress bar for data fetching
    total_steps = 3 + (1 if buildings else 0) + (1 if pois else 0)
    with tqdm(
        total=total_steps,
        desc="Fetching map data",
        unit="step",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
    ) as pbar:
        # 1. Fetch Street Network
        pbar.set_description("Downloading street network")
        compensated_dist = dist * (max(height, width) / min(height, width)) / 4
        g = fetch_graph(point, compensated_dist)
        if g is None:
            raise RuntimeError("Failed to retrieve street network data.")
        pbar.update(1)

        # 2. Fetch Water Features
        pbar.set_description("Downloading water features")
        water = fetch_features(
            point,
            compensated_dist,
            tags={"natural": ["water", "bay", "strait"], "waterway": "riverbank"},
            name="water",
        )
        pbar.update(1)

        # 3. Fetch Parks
        pbar.set_description("Downloading parks/green spaces")
        parks = fetch_features(
            point,
            compensated_dist,
            tags={"leisure": "park", "landuse": "grass"},
            name="parks",
        )
        pbar.update(1)

        # 4. Fetch Buildings (PR integration)
        bld = None
        if buildings:
            pbar.set_description("Downloading buildings")
            bld = fetch_features(
                point,
                compensated_dist,
                tags={"building": True},
                name="buildings",
            )
            pbar.update(1)

        # 5. Fetch POIs (PR integration)
        pois_gdf = None
        if pois:
            pbar.set_description("Downloading POIs")
            tags = _parse_poi_tags(poi_tags) if poi_tags else _default_poi_tags()
            pois_gdf = fetch_features(
                point,
                compensated_dist,
                tags=tags,
                name="pois",
            )
            pbar.update(1)

    print("All data retrieved successfully!")

    # Setup Plot
    print("Rendering map...")
    fig, ax = plt.subplots(figsize=(width, height), facecolor=THEME["bg"])
    ax.set_facecolor(THEME["bg"])
    ax.set_position((0.0, 0.0, 1.0, 1.0))

    # Project graph to a metric CRS so distances and aspect are linear (meters)
    g_proj = ox.project_graph(g)
    graph_crs = g_proj.graph["crs"]

    # Plot Layers
    # Layer 1: Polygons (filter to only polygon/multipolygon geometries)
    if water is not None and not water.empty:
        water_polys = water[water.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not water_polys.empty:
            try:
                water_polys = ox.projection.project_gdf(water_polys)
            except Exception:
                water_polys = water_polys.to_crs(graph_crs)
            water_polys.plot(ax=ax, facecolor=THEME['water'], edgecolor='none', zorder=0.5)

    if parks is not None and not parks.empty:
        parks_polys = parks[parks.geometry.type.isin(["Polygon", "MultiPolygon"])]
        if not parks_polys.empty:
            try:
                parks_polys = ox.projection.project_gdf(parks_polys)
            except Exception:
                parks_polys = parks_polys.to_crs(graph_crs)
            parks_polys.plot(ax=ax, facecolor=THEME['parks'], edgecolor='none', zorder=0.8)

    # Layer 1.5: Buildings (PR integration)
    if buildings and bld is not None and not bld.empty:
        bp = bld[bld.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
        try:
            bp = ox.projection.project_gdf(bp)
        except Exception:
            bp = bp.to_crs(graph_crs)
        if landmarks_only:
            bp = _filter_landmarks(bp, min_levels=min_levels)
        _plot_extruded(
            ax, bp,
            extrude_depth=float(extrude_depth),
            sun_azimuth=float(sun_azimuth),
            sun_elevation=float(sun_elevation),
            fill_color=(building_color or THEME.get('building_fill', '#1f1f1f')),
            shadow_color=(shadow_color or THEME.get('building_shadow', '#000000')),
            alpha_fill=float(building_opacity),
            base_alpha_shadow=0.25,
            z_base=3,
            shadow_alpha_override=(None if shadow_alpha is None else float(shadow_alpha)),
        )

    # Layer 2: Roads with hierarchy coloring
    print("Applying road hierarchy colors...")
    edge_colors = get_edge_colors_by_type(g_proj)
    edge_widths = get_edge_widths_by_type(g_proj)

    # Determine cropping limits to maintain the poster aspect ratio
    crop_xlim, crop_ylim = get_crop_limits(g_proj, point, fig, compensated_dist)

    ox.plot_graph(
        g_proj, ax=ax, bgcolor=THEME['bg'],
        node_size=0,
        edge_color=edge_colors,
        edge_linewidth=edge_widths,
        show=False,
        close=False,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(crop_xlim)
    ax.set_ylim(crop_ylim)

    # Layer 2.5: POIs (PR integration)
    if pois and pois_gdf is not None and not pois_gdf.empty:
        try:
            pois_proj = pois_gdf.copy()
            try:
                pois_proj = ox.projection.project_gdf(pois_proj)
            except Exception:
                pois_proj = pois_proj.to_crs(graph_crs)
            pts = pois_proj[pois_proj.geometry.type == 'Point']
            if not pts.empty:
                xs = pts.geometry.x.values
                ys = pts.geometry.y.values
                c = poi_color or THEME.get('poi_color', '#D33F49')
                ax.scatter(xs, ys, s=float(poi_size or 28.0), c=c, marker='^', edgecolors='none', zorder=12)
        except Exception:
            pass

    # Layer 3: Gradients (Top and Bottom)
    create_gradient_fade(ax, THEME['gradient_color'], location='bottom', zorder=10)
    create_gradient_fade(ax, THEME['gradient_color'], location='top', zorder=10)

    # Calculate scale factor based on smaller dimension (reference 12 inches)
    scale_factor = min(height, width) / 12.0

    base_main = 60
    base_sub = 22
    base_coords = 14
    base_attr = 8

    # Typography - use custom fonts if provided, otherwise use default FONTS
    active_fonts = fonts or FONTS
    if active_fonts:
        font_sub = FontProperties(fname=active_fonts["light"], size=base_sub * scale_factor)
        font_coords = FontProperties(fname=active_fonts["regular"], size=base_coords * scale_factor)
        font_attr = FontProperties(fname=active_fonts["light"], size=base_attr * scale_factor)
    else:
        font_sub = FontProperties(family="monospace", weight="normal", size=base_sub * scale_factor)
        font_coords = FontProperties(family="monospace", size=base_coords * scale_factor)
        font_attr = FontProperties(family="monospace", size=base_attr * scale_factor)

    # Format city name based on script type (i18n from main)
    if is_latin_script(display_city):
        spaced_city = "  ".join(list(display_city.upper()))
    else:
        spaced_city = display_city

    # Dynamically adjust font size based on city name length (from main)
    base_adjusted_main = base_main * scale_factor
    city_char_count = len(display_city)

    if city_char_count > 10:
        length_factor = 10 / city_char_count
        adjusted_font_size = max(base_adjusted_main * length_factor, 10 * scale_factor)
    else:
        adjusted_font_size = base_adjusted_main

    if active_fonts:
        font_main_adjusted = FontProperties(fname=active_fonts["bold"], size=adjusted_font_size)
    else:
        font_main_adjusted = FontProperties(family="monospace", weight="bold", size=adjusted_font_size)

    # --- BOTTOM TEXT ---
    ax.text(
        0.5, 0.14, spaced_city,
        transform=ax.transAxes, color=THEME["text"],
        ha="center", fontproperties=font_main_adjusted, zorder=11,
    )

    ax.text(
        0.5, 0.10, display_country.upper(),
        transform=ax.transAxes, color=THEME["text"],
        ha="center", fontproperties=font_sub, zorder=11,
    )

    lat, lon = point
    coords = (
        f"{lat:.4f}\u00b0 N / {lon:.4f}\u00b0 E"
        if lat >= 0
        else f"{abs(lat):.4f}\u00b0 S / {lon:.4f}\u00b0 E"
    )
    if lon < 0:
        coords = coords.replace("E", "W")

    ax.text(
        0.5, 0.07, coords,
        transform=ax.transAxes, color=THEME["text"], alpha=0.7,
        ha="center", fontproperties=font_coords, zorder=11,
    )

    ax.plot(
        [0.4, 0.6], [0.125, 0.125],
        transform=ax.transAxes, color=THEME["text"],
        linewidth=1 * scale_factor, zorder=11,
    )

    # --- ATTRIBUTION (bottom right) ---
    ax.text(
        0.98, 0.02, "\u00a9 OpenStreetMap contributors",
        transform=ax.transAxes, color=THEME["text"], alpha=0.5,
        ha="right", va="bottom", fontproperties=font_attr, zorder=11,
    )

    # Save
    print(f"Saving to {output_file}...")

    fmt = output_format.lower()
    save_kwargs = dict(
        facecolor=THEME["bg"],
        bbox_inches="tight",
        pad_inches=0.05,
    )

    if fmt == "png":
        save_kwargs["dpi"] = 300

    plt.savefig(output_file, format=fmt, **save_kwargs)
    plt.close()
    print(f"Done! Poster saved as {output_file}")


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def print_examples():
    """Print usage examples."""
    print("""
City Map Poster Generator
=========================

Usage:
  python create_map_poster.py --city <city> --country <country> [options]

Examples:
  # Iconic grid patterns
  python create_map_poster.py -c "New York" -C "USA" -t noir -d 12000           # Manhattan grid
  python create_map_poster.py -c "Barcelona" -C "Spain" -t warm_beige -d 8000   # Eixample district grid

  # Waterfront & canals
  python create_map_poster.py -c "Venice" -C "Italy" -t blueprint -d 4000       # Canal network
  python create_map_poster.py -c "Amsterdam" -C "Netherlands" -t ocean -d 6000  # Concentric canals
  python create_map_poster.py -c "Dubai" -C "UAE" -t midnight_blue -d 15000     # Palm & coastline

  # Radial patterns
  python create_map_poster.py -c "Paris" -C "France" -t pastel_dream -d 10000   # Haussmann boulevards
  python create_map_poster.py -c "Moscow" -C "Russia" -t noir -d 12000          # Ring roads

  # Organic old cities
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 15000    # Dense organic streets
  python create_map_poster.py -c "Marrakech" -C "Morocco" -t terracotta -d 5000 # Medina maze
  python create_map_poster.py -c "Rome" -C "Italy" -t warm_beige -d 8000        # Ancient street layout

  # Coastal cities
  python create_map_poster.py -c "San Francisco" -C "USA" -t sunset -d 10000    # Peninsula grid
  python create_map_poster.py -c "Sydney" -C "Australia" -t ocean -d 12000      # Harbor city
  python create_map_poster.py -c "Mumbai" -C "India" -t contrast_zones -d 18000 # Coastal peninsula

  # River cities
  python create_map_poster.py -c "London" -C "UK" -t noir -d 15000              # Thames curves
  python create_map_poster.py -c "Budapest" -C "Hungary" -t copper_patina -d 8000  # Danube split

  # With buildings and POIs
  python create_map_poster.py -c "London" -C "UK" -t noir -d 8000 --buildings --pois
  python create_map_poster.py -c "Tokyo" -C "Japan" -t japanese_ink -d 6000 --buildings --landmarks-only

  # Override center coordinates
  python create_map_poster.py --city "New York" --country "USA" -lat 40.776676 -long -73.971321 -t noir

  # Multilingual
  python create_map_poster.py -c "Tokyo" -C "Japan" -dc "\u6771\u4eac" -dC "\u65e5\u672c" --font-family "Noto Sans JP"

  # List themes
  python create_map_poster.py --list-themes

  # Generate posters for every theme
  python create_map_poster.py -c "Tokyo" -C "Japan" --all-themes

Options:
  --city, -c        City name (required)
  --country, -C     Country name (required)
  --country-label   Override country text displayed on poster
  --theme, -t       Theme name (default: terracotta)
  --all-themes      Generate posters for all themes
  --distance, -d    Map radius in meters (default: 18000)
  --width, -W       Image width in inches (default: 12, max: 20)
  --height, -H      Image height in inches (default: 16, max: 20)
  --quality, -q     Quality preset: standard, high, ultra
  --buildings       Render building footprints
  --pois            Render points of interest
  --list-themes     List all available themes
  --list-quality    List available quality presets

Distance guide:
  4000-6000m   Small/dense cities (Venice, Amsterdam old center)
  8000-12000m  Medium cities, focused downtown (Paris, Barcelona)
  15000-20000m Large metros, full city view (Tokyo, Mumbai)

Available themes can be found in the 'themes/' directory.
Generated posters are saved to 'posters/' directory.
""")


def list_themes():
    """List all available themes with descriptions."""
    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        return

    print("\nAvailable Themes:")
    print("-" * 60)
    for theme_name in available_themes:
        theme_path = os.path.join(THEMES_DIR, f"{theme_name}.json")
        try:
            with open(theme_path, "r", encoding=FILE_ENCODING) as f:
                theme_data = json.load(f)
                display_name = theme_data.get('name', theme_name)
                description = theme_data.get('description', '')
        except (OSError, json.JSONDecodeError):
            display_name = theme_name
            description = ""
        print(f"  {theme_name}")
        print(f"    {display_name}")
        if description:
            print(f"    {description}")
        print()


def list_quality_levels():
    """List available quality presets."""
    print("\nAvailable Quality Levels:")
    print("-" * 40)
    for level, specs in QUALITY_PRESETS.items():
        w, h = specs['figsize']
        dpi = specs['dpi']
        print(f"  {level}")
        print(f"    Portrait: {w}x{h} in @ {dpi} DPI")
        print(f"    Landscape: {h}x{w} in @ {dpi} DPI")
        print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate beautiful map posters for any city",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_map_poster.py --city "New York" --country "USA"
  python create_map_poster.py --city "New York" --country "USA" -lat 40.776676 -long -73.971321 --theme neon_cyberpunk
  python create_map_poster.py --city Tokyo --country Japan --theme midnight_blue
  python create_map_poster.py --city Paris --country France --theme noir --distance 15000
  python create_map_poster.py --list-themes
        """,
    )

    # Core options
    parser.add_argument("--city", "-c", type=str, help="City name")
    parser.add_argument("--country", "-C", type=str, help="Country name")
    parser.add_argument(
        "--latitude", "-lat", dest="latitude", type=str,
        help="Override latitude center point",
    )
    parser.add_argument(
        "--longitude", "-long", dest="longitude", type=str,
        help="Override longitude center point",
    )
    parser.add_argument(
        "--country-label", dest="country_label", type=str,
        help="Override country text displayed on poster",
    )
    parser.add_argument(
        "--theme", "-t", type=str, default="terracotta",
        help="Theme name (default: terracotta)",
    )
    parser.add_argument(
        "--all-themes", "--All-themes", dest="all_themes", action="store_true",
        help="Generate posters for all themes",
    )
    parser.add_argument(
        "--distance", "-d", type=int, default=18000,
        help="Map radius in meters (default: 18000)",
    )
    parser.add_argument(
        "--width", "-W", type=float, default=12,
        help="Image width in inches (default: 12, max: 20)",
    )
    parser.add_argument(
        "--height", "-H", type=float, default=16,
        help="Image height in inches (default: 16, max: 20)",
    )
    parser.add_argument("--list-themes", action="store_true", help="List all available themes")

    # Quality presets (from PR integrations)
    parser.add_argument(
        "--quality", "-q", type=str, default=None,
        choices=list(QUALITY_PRESETS.keys()),
        help="Quality preset (overrides --width/--height/DPI)",
    )
    parser.add_argument("--list-quality", action="store_true", help="List quality presets")

    # i18n (from main)
    parser.add_argument(
        "--display-city", "-dc", type=str,
        help="Custom display name for city (for i18n support)",
    )
    parser.add_argument(
        "--display-country", "-dC", type=str,
        help="Custom display name for country (for i18n support)",
    )
    parser.add_argument(
        "--font-family", type=str,
        help='Google Fonts family name (e.g., "Noto Sans JP", "Open Sans")',
    )

    # Output format (from main)
    parser.add_argument(
        "--format", "-f", default="png", choices=["png", "svg", "pdf"],
        help="Output format for the poster (default: png)",
    )

    # Buildings (from PR integrations)
    parser.add_argument("--buildings", action="store_true", default=False, help="Render building footprints")
    parser.add_argument("--landmarks-only", action="store_true", default=False, help="Only render landmark buildings")
    parser.add_argument("--extrude-depth", type=float, default=12.0, help="Shadow extrusion depth")
    parser.add_argument("--min-levels", type=float, default=5, help="Min building levels for landmark filter")
    parser.add_argument("--building-opacity", type=float, default=0.9, help="Building fill opacity")
    parser.add_argument("--building-color", type=str, default=None, help="Override building fill color")
    parser.add_argument("--shadow-color", type=str, default=None, help="Override building shadow color")

    # Sun (from PR integrations)
    parser.add_argument("--sun-azimuth", type=float, default=45.0, help="Sun azimuth in degrees")
    parser.add_argument("--sun-elevation", type=float, default=45.0, help="Sun elevation in degrees")
    parser.add_argument("--shadow-alpha", type=float, default=None, help="Override shadow opacity")

    # POIs (from PR integrations)
    parser.add_argument("--pois", action="store_true", default=False, help="Render points of interest")
    parser.add_argument("--poi-labels", action="store_true", default=False, help="Show POI labels")
    parser.add_argument("--poi-max", type=int, default=20, help="Maximum number of POIs")
    parser.add_argument("--poi-size", type=float, default=28.0, help="POI marker size")
    parser.add_argument("--poi-color", type=str, default=None, help="Override POI color")
    parser.add_argument("--poi-tags", type=str, default=None, help="Custom POI tags (key=val;key=val)")

    # Overpass/OSMnx tuning (from PR integrations)
    parser.add_argument("--overpass-max-area", type=float, default=None, help="Override OSMnx max_query_area_size (m^2)")
    parser.add_argument("--overpass-memory", type=int, default=None, help="Overpass memory (MB) if supported")
    parser.add_argument("--requests-timeout", type=int, default=None, help="HTTP/Overpass timeout (seconds)")

    args = parser.parse_args()

    # If no arguments provided, show examples
    if len(sys.argv) == 1:
        print_examples()
        sys.exit(0)

    # List themes if requested
    if args.list_themes:
        list_themes()
        sys.exit(0)

    # List quality presets if requested
    if args.list_quality:
        list_quality_levels()
        sys.exit(0)

    # Validate required arguments
    if not args.city or not args.country:
        print("Error: --city and --country are required.\n")
        print_examples()
        sys.exit(1)

    # Enforce maximum dimensions
    if args.width > 20:
        print(f"Warning: Width {args.width} exceeds max of 20. Clamping to 20.")
        args.width = 20.0
    if args.height > 20:
        print(f"Warning: Height {args.height} exceeds max of 20. Clamping to 20.")
        args.height = 20.0

    # If quality preset specified, it overrides width/height
    if args.quality:
        preset = QUALITY_PRESETS[args.quality]
        args.width, args.height = preset['figsize']

    available_themes = get_available_themes()
    if not available_themes:
        print("No themes found in 'themes/' directory.")
        sys.exit(1)

    if args.all_themes:
        themes_to_generate = available_themes
    else:
        if args.theme not in available_themes:
            print(f"Error: Theme '{args.theme}' not found.")
            print(f"Available themes: {', '.join(available_themes)}")
            sys.exit(1)
        themes_to_generate = [args.theme]

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

    print("=" * 50)
    print("City Map Poster Generator")
    print("=" * 50)

    # Load custom fonts if specified
    custom_fonts = None
    if args.font_family:
        custom_fonts = load_fonts(args.font_family)
        if not custom_fonts:
            print(f"Warning: Failed to load '{args.font_family}', falling back to Roboto")

    # Get coordinates and generate poster
    try:
        if args.latitude and args.longitude:
            lat = parse(args.latitude)
            lon = parse(args.longitude)
            coords = [lat, lon]
            print(f"Coordinates: {', '.join([str(i) for i in coords])}")
        else:
            coords = get_coordinates(args.city, args.country)

        for theme_name in themes_to_generate:
            THEME = load_theme(theme_name)
            output_file = generate_output_filename(args.city, theme_name, args.format)
            create_poster(
                args.city,
                args.country,
                coords,
                args.distance,
                output_file,
                args.format,
                args.width,
                args.height,
                country_label=args.country_label,
                display_city=args.display_city,
                display_country=args.display_country,
                fonts=custom_fonts,
                # PR integration features
                buildings=args.buildings,
                landmarks_only=args.landmarks_only,
                extrude_depth=args.extrude_depth,
                min_levels=args.min_levels,
                building_opacity=args.building_opacity,
                sun_azimuth=args.sun_azimuth,
                sun_elevation=args.sun_elevation,
                shadow_alpha=args.shadow_alpha,
                pois=args.pois,
                poi_labels=args.poi_labels,
                poi_max=args.poi_max,
                poi_size=args.poi_size,
                poi_color=args.poi_color,
                poi_tags=args.poi_tags,
                building_color=args.building_color,
                shadow_color=args.shadow_color,
            )

        print("\n" + "=" * 50)
        print("Poster generation complete!")
        print("=" * 50)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
