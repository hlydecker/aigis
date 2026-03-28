# -*- coding: utf-8 -*-
"""Tests for aigis.convert.coordinates."""
import pytest
from shapely.geometry import Polygon

from aigis.convert.coordinates import (
    pixel_segmentation_to_spatial_rio,
    pixel_to_spatial_rio,
    spatial_polygon_to_pixel_rio,
    spatial_to_pixel_rio,
    wkt_parser,
)


# ---------------------------------------------------------------------------
# wkt_parser
# ---------------------------------------------------------------------------


def test_wkt_parser_local_cs_extracts_name():
    # String where LOCAL_CS[ appears as a quoted token after splitting on "
    wkt = '"LOCAL_CS["My Local CRS"'
    result = wkt_parser(wkt)
    assert result == "My Local CRS"


def test_wkt_parser_plain_string_returned_unchanged():
    plain = "EPSG:32755"
    assert wkt_parser(plain) == plain


def test_wkt_parser_no_keyword_returns_original():
    wkt = "some random WKT string without the keyword"
    assert wkt_parser(wkt) == wkt


# ---------------------------------------------------------------------------
# pixel_to_spatial_rio / spatial_to_pixel_rio  (round-trip)
# ---------------------------------------------------------------------------


def test_pixel_to_spatial_top_left(open_geotiff):
    x, y = pixel_to_spatial_rio(open_geotiff, 0, 0)
    # from_bounds(0, 0, 100, 100, 100, 100) → cell centre at (0.5, 99.5)
    assert abs(x - 0.5) < 1.0
    assert abs(y - 99.5) < 1.0


def test_pixel_to_spatial_centre(open_geotiff):
    x, y = pixel_to_spatial_rio(open_geotiff, 50, 50)
    # Centre pixel of 100×100 raster spanning [0,100]×[0,100]
    assert 45.0 < x < 55.0
    assert 45.0 < y < 55.0


def test_spatial_to_pixel_roundtrip(open_geotiff):
    x, y = pixel_to_spatial_rio(open_geotiff, 10, 20)
    row, col = spatial_to_pixel_rio(open_geotiff, x, y)
    assert abs(row - 10) <= 1
    assert abs(col - 20) <= 1


def test_spatial_to_pixel_within_bounds(open_geotiff):
    # Spatial centre of the raster should map near (50, 50)
    row, col = spatial_to_pixel_rio(open_geotiff, 50.0, 50.0)
    assert 0 <= row < 100
    assert 0 <= col < 100


# ---------------------------------------------------------------------------
# pixel_segmentation_to_spatial_rio
# ---------------------------------------------------------------------------


def test_pixel_segmentation_to_spatial_rio_is_polygon(open_geotiff):
    # Flat COCO-style segmentation: [col0, row0, col1, row1, ...]
    seg = [10, 10, 20, 10, 20, 20, 10, 20]
    result = pixel_segmentation_to_spatial_rio(open_geotiff, seg)
    assert isinstance(result, Polygon)


def test_pixel_segmentation_to_spatial_rio_in_range(open_geotiff):
    seg = [10, 10, 20, 10, 20, 20, 10, 20]
    polygon = pixel_segmentation_to_spatial_rio(open_geotiff, seg)
    bounds = open_geotiff.bounds
    centroid = polygon.centroid
    assert bounds.left <= centroid.x <= bounds.right
    assert bounds.bottom <= centroid.y <= bounds.top


# ---------------------------------------------------------------------------
# spatial_polygon_to_pixel_rio
# ---------------------------------------------------------------------------


def test_spatial_polygon_to_pixel_rio_within_bounds(open_geotiff):
    bounds = open_geotiff.bounds
    poly = Polygon(
        [
            (bounds.left + 10, bounds.bottom + 10),
            (bounds.left + 30, bounds.bottom + 10),
            (bounds.left + 30, bounds.bottom + 30),
            (bounds.left + 10, bounds.bottom + 30),
        ]
    )
    result = spatial_polygon_to_pixel_rio(open_geotiff, poly)
    for col, row in result:
        assert 0 <= row < 100
        assert 0 <= col < 100


def test_spatial_polygon_to_pixel_rio_returns_list(open_geotiff):
    bounds = open_geotiff.bounds
    poly = Polygon(
        [
            (bounds.left + 5, bounds.bottom + 5),
            (bounds.left + 15, bounds.bottom + 5),
            (bounds.left + 15, bounds.bottom + 15),
        ]
    )
    result = spatial_polygon_to_pixel_rio(open_geotiff, poly)
    assert isinstance(result, list)
    assert len(result) > 0
