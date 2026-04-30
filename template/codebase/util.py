import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon


def load_polygons(polygons_dir, polygons,
                  target_crs='EPSG:32610',
                  filter_release=False):
    """
    Load polygon geometries from .pol or .shp files into a GeoDataFrame.

    Args:
        polygons_dir:   Directory containing the polygon files.
        polygons:       List of polygon config dicts from config.yaml, each with
                        keys: 'name', 'file', and optionally 'pulse_size',
                        'release_interval', 'release'.
        target_crs:     Target CRS for all output geometries.  Defaults to
                        EPSG:32610 (WGS 84 / UTM zone 10N).  Override for
                        other regions (e.g. 'EPSG:32611' for UTM zone 11N).
        filter_release: If True, only load polygons where 'release' is True.

    Returns:
        GeoDataFrame with columns: geometry, name, pulse_size, release_interval.
    """
    all_pols = gpd.GeoDataFrame()

    regions = polygons
    if filter_release:
        regions = [r for r in polygons if r.get('release', True)]

    for region in regions:
        filepath = os.path.join(polygons_dir, region['file'])
        if not os.path.isfile(filepath):
            print(f'! File not found, skipping: {filepath}')
            continue

        try:
            if filepath.endswith('.pol'):
                gdf = read_pol_file(filepath)
            elif filepath.endswith('.shp'):
                gdf = gpd.read_file(filepath)
            else:
                print(f'! Unsupported file type, skipping: {filepath}')
                continue

            if gdf.crs is None:
                gdf = gdf.set_crs(target_crs)
            else:
                gdf = gdf.to_crs(target_crs)

            gdf['name']             = region['name']
            gdf['pulse_size']       = region.get('pulse_size', None)
            gdf['release_interval'] = region.get('release_interval', None)

            all_pols = pd.concat([all_pols, gdf], ignore_index=True)
        except Exception as exc:
            print(f'! Error loading {filepath}: {exc}')

    return all_pols


def read_pol_file(path):
    """
    Parse a Delft3D/MIKE .pol polygon file into a GeoDataFrame.

    Expected format:
        Line 1: file header (ignored)
        Line 2: polygon name
        Line 3: blank or metadata (ignored)
        Line 4: vertex count
        Lines 5+: space-separated x y coordinate pairs

    Args:
        path:  Path to the .pol file.

    Returns:
        Single-row GeoDataFrame with columns: geometry, name, vertex_count.
    """
    with open(path) as f:
        lines = f.readlines()

    name         = lines[1].strip()
    vertex_count = int(lines[3].strip())
    coords = [
        tuple(map(float, ln.split()))
        for ln in lines[4:]
        if len(ln.split()) == 2
    ]
    gdf = gpd.GeoDataFrame({
        'geometry':     [Polygon(coords)],
        'name':         [name],
        'vertex_count': [vertex_count],
    })
    return gdf


def read_pol_files_in_dir(directory):
    """
    Read all .pol files in a directory into a single GeoDataFrame.

    Args:
        directory:  Path to directory containing .pol files.

    Returns:
        GeoDataFrame with all polygons concatenated.
    """
    polygons = gpd.GeoDataFrame()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filepath.endswith('.pol'):
            gdf = read_pol_file(filepath)
            polygons = pd.concat([polygons, gdf], ignore_index=True)
    return polygons


def extract_polygon_coords(gdf):
    """
    Extract exterior coordinates from all polygons in a GeoDataFrame.

    Handles both Polygon and MultiPolygon geometries.  Returns coordinates
    as nested lists (not tuples) for OceanTracker compatibility.

    Args:
        gdf:  GeoDataFrame with polygon geometries.

    Returns:
        List of polygons, where each polygon is a list of [x, y] pairs.
    """
    coords = []
    for geom in gdf.geometry:
        if geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                coords.append([list(pt) for pt in poly.exterior.coords])
        elif geom.geom_type == 'Polygon':
            coords.append([list(pt) for pt in geom.exterior.coords])
    return coords
