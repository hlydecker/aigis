# -*- coding: utf-8 -*-
"""Tests for aigis.annotate.utils.

Bug #1 note: create_grid() uses Polygon (Shapely) on line 98 without
importing it.  The NameError is swallowed by a bare except-and-return-None
pattern, so the function returns None instead of raising.  The test below
pins this behaviour until the import is fixed.
"""
import json

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from aigis.annotate.utils import create_grid, geojson_csv_filter


# ---------------------------------------------------------------------------
# geojson_csv_filter
# ---------------------------------------------------------------------------


def _write_geojson(path, n_features):
    """Write a minimal GeoJSON with n_features, each with an 'id' field."""
    features = [
        {
            "type": "Feature",
            "properties": {"id": i},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[i, i], [i + 1, i], [i + 1, i + 1], [i, i + 1], [i, i]]
                ],
            },
        }
        for i in range(n_features)
    ]
    fc = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(fc, f)


def _write_id_csv(path, ids):
    pd.DataFrame({"id": ids}).to_csv(path, index=False)


def test_geojson_csv_filter_returns_matching_rows(tmp_path):
    geojson_path = str(tmp_path / "features.geojson")
    csv_path = str(tmp_path / "ids.csv")
    _write_geojson(geojson_path, 5)  # ids 0..4
    _write_id_csv(csv_path, [0, 2, 4])

    result = geojson_csv_filter(geojson_path, csv_path)
    assert len(result) == 3


def test_geojson_csv_filter_empty_csv_returns_empty(tmp_path):
    geojson_path = str(tmp_path / "features.geojson")
    csv_path = str(tmp_path / "empty.csv")
    _write_geojson(geojson_path, 5)
    _write_id_csv(csv_path, [])

    result = geojson_csv_filter(geojson_path, csv_path)
    assert len(result) == 0


def test_geojson_csv_filter_all_ids_match(tmp_path):
    geojson_path = str(tmp_path / "features.geojson")
    csv_path = str(tmp_path / "ids.csv")
    _write_geojson(geojson_path, 3)
    _write_id_csv(csv_path, [0, 1, 2])

    result = geojson_csv_filter(geojson_path, csv_path)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# create_grid  — pins Bug #1
# ---------------------------------------------------------------------------


def _make_boundary_gdf():
    """Return a simple GeoDataFrame suitable as boundary_data."""
    polygon = box(0, 0, 500, 500)
    return gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:3857")


def test_create_grid_returns_none_due_to_missing_import():
    """Bug #1: Polygon is not imported in annotate/utils.py.

    The NameError is caught and the function returns None.
    This test will need to be updated once the import bug is fixed.
    """
    boundary = _make_boundary_gdf()
    result = create_grid(boundary, grid_size=100)
    # Bug present: returns None instead of a GeoJSON dict
    assert result is None, (
        "create_grid should return None due to missing Polygon import (Bug #1). "
        "If this test fails, the bug has been fixed — update accordingly."
    )
