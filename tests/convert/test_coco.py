# -*- coding: utf-8 -*-
"""Tests for aigis.convert.coco."""
import os
import warnings

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from aigis.convert.coco import (
    coco_bbox,
    coco_polygon_annotation,
    make_category,
    make_category_object,
    polygon_prep,
    raster_to_coco,
)


# ---------------------------------------------------------------------------
# make_category
# ---------------------------------------------------------------------------


def test_make_category_keys():
    cat = make_category("building", 1)
    assert set(cat.keys()) == {"supercategory", "id", "name"}


def test_make_category_values():
    cat = make_category("building", 3, supercategory="structure")
    assert cat["id"] == 3
    assert cat["name"] == "building"
    assert cat["supercategory"] == "structure"


def test_make_category_trim():
    cat = make_category("__tree", 1, trim=2)
    assert cat["name"] == "tree"


def test_make_category_default_supercategory():
    cat = make_category("tree", 1)
    assert cat["supercategory"] == "landuse"


# ---------------------------------------------------------------------------
# coco_bbox
# ---------------------------------------------------------------------------


def test_coco_bbox_square(simple_square_polygon):
    bbox = coco_bbox(simple_square_polygon)
    assert bbox == [0.0, 0.0, 10.0, 10.0]


def test_coco_bbox_format():
    poly = Polygon([(2, 3), (7, 3), (7, 8), (2, 8)])
    x, y, w, h = coco_bbox(poly)
    assert x == 2.0
    assert y == 3.0
    assert w == 5.0
    assert h == 5.0


def test_coco_bbox_non_axis_aligned():
    poly = Polygon([(0, 5), (5, 0), (10, 5), (5, 10)])
    bbox = coco_bbox(poly)
    assert len(bbox) == 4
    assert bbox[2] == 10.0  # width = 10 - 0
    assert bbox[3] == 10.0  # height = 10 - 0


# ---------------------------------------------------------------------------
# coco_polygon_annotation
# ---------------------------------------------------------------------------


def test_coco_polygon_annotation_keys():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    ann = coco_polygon_annotation(poly, image_id=1, annot_id=5, class_id=2)
    expected_keys = {"segmentation", "area", "iscrowd", "image_id", "bbox", "category_id", "id"}
    assert set(ann.keys()) == expected_keys


def test_coco_polygon_annotation_iscrowd_zero():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    ann = coco_polygon_annotation(poly, image_id=1, annot_id=1, class_id=1)
    assert ann["iscrowd"] == 0


def test_coco_polygon_annotation_segmentation_flat():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    ann = coco_polygon_annotation(poly, image_id=1, annot_id=1, class_id=1)
    assert isinstance(ann["segmentation"], list)
    # Flat list of numbers, not nested tuples
    for item in ann["segmentation"]:
        assert isinstance(item, (int, float))


def test_coco_polygon_annotation_ids():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    ann = coco_polygon_annotation(poly, image_id=7, annot_id=42, class_id=3)
    assert ann["image_id"] == 7
    assert ann["id"] == 42
    assert ann["category_id"] == 3


# ---------------------------------------------------------------------------
# polygon_prep
# ---------------------------------------------------------------------------


def test_polygon_prep_minimum_rotated_rectangle():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])
    result = polygon_prep(poly, minimum_rotated_rectangle=True)
    # minimum_rotated_rectangle of a square returns 4 or 5 vertices (closed)
    assert result.shape[0] in (4, 5)


def test_polygon_prep_simplify_reduces_vertices():
    # Dense polygon with many vertices
    angles = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    poly = np.column_stack([np.cos(angles) * 10, np.sin(angles) * 10])
    result = polygon_prep(poly, simplify_tolerance=2.0)
    assert result.shape[0] < 50


def test_polygon_prep_sub_three_points_warns():
    # polygon_prep warns for < 3 points, then Shapely raises ValueError
    # (Shapely 2.x requires >= 4 coords for a LinearRing).
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises((ValueError, Exception)):
            polygon_prep([(0, 0), (1, 1)], simplify_tolerance=0.0)
    assert len(w) >= 1
    assert any("less than 3 points" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# raster_to_coco
# ---------------------------------------------------------------------------


def test_raster_to_coco_returns_correct_dimensions(small_geotiff):
    img = raster_to_coco(small_geotiff, index=0)
    assert img.width == 100
    assert img.height == 100


def test_raster_to_coco_file_name(small_geotiff):
    img = raster_to_coco(small_geotiff, index=0)
    # file_name should be just the basename with .png extension
    assert img.file_name.endswith(".png")
    assert os.sep not in img.file_name


def test_raster_to_coco_writes_png(small_geotiff):
    raster_to_coco(small_geotiff, index=0)
    expected = os.path.splitext(small_geotiff)[0] + ".png"
    assert os.path.exists(expected)


def test_raster_to_coco_id(small_geotiff):
    img = raster_to_coco(small_geotiff, index=7)
    assert img.id == 7


# ---------------------------------------------------------------------------
# make_category_object
# ---------------------------------------------------------------------------


def test_make_category_object_count():
    gdf = gpd.GeoDataFrame(
        {"class": ["tree", "building", "tree", "water"], "geometry": [Point(0, 0)] * 4}
    )
    result = make_category_object(gdf, "class", trim=0)
    assert len(result) == 3  # 3 unique classes


def test_make_category_object_sequential_ids():
    gdf = gpd.GeoDataFrame(
        {"class": ["a", "b"], "geometry": [Point(0, 0)] * 2}
    )
    result = make_category_object(gdf, "class", trim=0)
    ids = sorted(cat["id"] for cat in result)
    assert ids == [0, 1]
