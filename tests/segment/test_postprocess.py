# -*- coding: utf-8 -*-
"""Tests for aigis.segment.postprocess."""
import sys
from unittest.mock import patch

import numpy as np
import pytest

supervision = pytest.importorskip(
    "supervision",
    reason="supervision not installed; skipping postprocess tests",
)

from aigis.segment.postprocess import convert_polygons_to_geospatial, detectron2_to_polygons


# ---------------------------------------------------------------------------
# convert_polygons_to_geospatial
# ---------------------------------------------------------------------------


def test_convert_polygons_row_count(small_geotiff):
    # Two flat polygon coordinate lists [col, row, col, row, ...]
    polygons = [
        [5, 5, 15, 5, 15, 15, 5, 15],
        [20, 20, 30, 20, 30, 30, 20, 30],
    ]
    gdf = convert_polygons_to_geospatial(polygons, small_geotiff)
    assert len(gdf) == 2


def test_convert_polygons_has_crs(small_geotiff):
    polygons = [[5, 5, 15, 5, 15, 15, 5, 15]]
    gdf = convert_polygons_to_geospatial(polygons, small_geotiff)
    assert gdf.crs is not None


def test_convert_polygons_crs_matches_raster(small_geotiff):
    import rasterio

    polygons = [[5, 5, 15, 5, 15, 15, 5, 15]]
    gdf = convert_polygons_to_geospatial(polygons, small_geotiff)
    with rasterio.open(small_geotiff) as src:
        expected_crs = src.crs
    assert gdf.crs == expected_crs


def test_convert_polygons_single_polygon(small_geotiff):
    polygons = [[0, 0, 10, 0, 10, 10, 0, 10]]
    gdf = convert_polygons_to_geospatial(polygons, small_geotiff)
    assert len(gdf) == 1
    assert not gdf.geometry.iloc[0].is_empty


# ---------------------------------------------------------------------------
# detectron2_to_polygons
# ---------------------------------------------------------------------------


def test_detectron2_to_polygons_returns_list(mock_detectron2_output):
    fake_poly = np.array([[5, 5], [9, 5], [9, 9], [5, 9]])
    with patch("aigis.segment.postprocess.sv.mask_to_polygons", return_value=[fake_poly]):
        result = detectron2_to_polygons(mock_detectron2_output)
    assert isinstance(result, list)


def test_detectron2_to_polygons_flat_coords(mock_detectron2_output):
    fake_poly = np.array([[5, 5], [9, 5], [9, 9], [5, 9]])
    with patch("aigis.segment.postprocess.sv.mask_to_polygons", return_value=[fake_poly]):
        result = detectron2_to_polygons(mock_detectron2_output)
    assert len(result) > 0
    for poly in result:
        assert isinstance(poly, list)
        for coord in poly:
            assert isinstance(coord, (int, float))


def test_detectron2_to_polygons_empty_mask_warns(mock_detectron2_output):
    """Empty sv.mask_to_polygons result → warning, polygon skipped."""
    with patch("aigis.segment.postprocess.sv.mask_to_polygons", return_value=[]):
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = detectron2_to_polygons(mock_detectron2_output)
        assert result == []
        assert any("empty" in str(warning.message).lower() for warning in w)
