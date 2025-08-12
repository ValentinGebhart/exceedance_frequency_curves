"""
coordinate functionalities
"""

import warnings

import numpy as np
import geopandas as gpd
import xarray as xr
from scipy.spatial import cKDTree
from shapely.geometry.point import Point


# from return_period_maps import ReturnPeriodMap


def _is_subgrid(geometry):
    """checks if there are at least two points with same lat
    and two points with same lon
    """
    all_coords = np.array([geom.coords[0] for geom in geometry])

    return (np.unique(all_coords[:, 0], return_counts=True)[1].max() > 1) & (
        np.unique(all_coords[:, 1], return_counts=True)[1].max() > 1
    )


def _infer_grid_parameter(geometry):

    if not _is_subgrid(geometry):
        raise ValueError("Geometry does not seem to be a grid.")

    all_coords = np.array([geom.coords[0] for geom in geometry])

    resolution_lon = np.min(np.diff(np.sort(np.unique(all_coords[:, 0]))))
    resolution_lat = np.min(np.diff(np.sort(np.unique(all_coords[:, 1]))))

    if np.round(resolution_lon - resolution_lat, decimals=3) != 0:
        warnings.warn("Grid is not square grid.")

    minlon, minlat, maxlon, maxlat = geometry.total_bounds

    return {
        "minlon": minlon,
        "minlat": minlat,
        "maxlon": maxlon,
        "maxlat": maxlat,
        "resolution_lat": resolution_lat,
        "resolution_lon": resolution_lon,
    }


def change_grid_resolution(geometry, scale_factor):
    grid_dict = _infer_grid_parameter(geometry)
    step_lon = scale_factor * grid_dict["resolution_lon"]
    step_lat = scale_factor * grid_dict["resolution_lat"]
    offset_lon = (scale_factor - 1) / 2 * grid_dict["resolution_lon"]
    offset_lat = (scale_factor - 1) / 2 * grid_dict["resolution_lat"]

    new_geometry = gpd.GeoSeries(
        [
            Point(x, y)
            for x in np.arange(
                grid_dict["minlon"] + offset_lon,
                grid_dict["maxlon"] + offset_lon,
                step_lon,
            )
            for y in np.arange(
                grid_dict["minlat"] + offset_lat,
                grid_dict["maxlat"] + offset_lat,
                step_lat,
            )
        ],
        crs=geometry.crs,
    )

    # Extract coordinates
    coords1 = np.array([[geom.x, geom.y] for geom in geometry])
    coords2 = np.array([[geom.x, geom.y] for geom in new_geometry])

    # Build KDTree on coarse coordinates
    tree = cKDTree(coords2)

    # Query nearest neighbor
    _, assignment = tree.query(coords1, k=1)

    return new_geometry, assignment


def _change_grid_resolution_xarray(geometry, scale_factor):

    all_coords = np.array([geom.coords[0] for geom in geometry]).T

    # Create grid as xarray Dataset
    ds = xr.Dataset(coords={"lat": sorted(all_coords[1]), "lon": sorted(all_coords[0])})

    # Coarsen the coordinate grid (no interpolation)
    coarse_ds = ds.coarsen(lat=scale_factor, lon=scale_factor, boundary="trim").reduce(
        lambda x: x[::1]
    )

    # Access the new coarsened grid
    new_lat = coarse_ds.lat.values
    new_lon = coarse_ds.lon.values
    return new_lat, new_lon
