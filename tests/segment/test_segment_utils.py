# -*- coding: utf-8 -*-
"""Tests for aigis.segment.utils.

detectron2 and other heavy dependencies are mocked at the module level so
this file runs without GPU or model weights.
"""
import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import Point

# ---------------------------------------------------------------------------
# Mock heavy imports before aigis.segment.utils is loaded
# ---------------------------------------------------------------------------
for _mod in [
    "detectron2",
    "detectron2.utils",
    "detectron2.utils.visualizer",
    "folium",
]:
    sys.modules.setdefault(_mod, MagicMock())

from aigis.segment.utils import (  # noqa: E402 — must come after mocks
    assemble_coco_json,
    extract_output_annotations,
    generate_synthetic_coco_dataset,
    polygon_prep,
)


# ---------------------------------------------------------------------------
# generate_synthetic_coco_dataset
# ---------------------------------------------------------------------------


def test_generate_synthetic_coco_dataset_keys():
    dataset = generate_synthetic_coco_dataset(num_images=2, num_objects=3, num_classes=2)
    assert set(dataset.keys()) == {"info", "licenses", "images", "annotations", "categories"}


def test_generate_synthetic_coco_dataset_image_count():
    dataset = generate_synthetic_coco_dataset(num_images=4, num_objects=1, num_classes=1)
    assert len(dataset["images"]) == 4


def test_generate_synthetic_coco_dataset_annotation_count():
    dataset = generate_synthetic_coco_dataset(num_images=2, num_objects=3, num_classes=2)
    assert len(dataset["annotations"]) == 6  # 2 images × 3 objects


def test_generate_synthetic_coco_dataset_category_ids_valid():
    dataset = generate_synthetic_coco_dataset(num_images=5, num_objects=2, num_classes=3)
    valid_ids = {cat["id"] for cat in dataset["categories"]}
    for ann in dataset["annotations"]:
        assert ann["category_id"] in valid_ids


def test_generate_synthetic_coco_dataset_category_count():
    dataset = generate_synthetic_coco_dataset(num_classes=5)
    assert len(dataset["categories"]) == 5


# ---------------------------------------------------------------------------
# polygon_prep (segment version)
# ---------------------------------------------------------------------------


def test_polygon_prep_minimum_rotated_rectangle():
    poly = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    result = polygon_prep(poly, minimum_rotated_rectangle=True)
    assert result.shape[0] in (4, 5)


def test_polygon_prep_simplify_reduces_vertices():
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
            polygon_prep([(0, 0), (1, 1)])
    assert len(w) >= 1
    assert any("less than 3 points" in str(warning.message) for warning in w)


# ---------------------------------------------------------------------------
# extract_output_annotations
# ---------------------------------------------------------------------------


def test_extract_output_annotations_returns_four_lists(mock_detectron2_output):
    fake_poly = np.array([[5, 5], [9, 5], [9, 9], [5, 9]])
    sv_mock = sys.modules.get("supervision") or MagicMock()
    with patch("aigis.segment.utils.sv.mask_to_polygons", return_value=[fake_poly]):
        mask_arrays, polygons, bbox_list, labels_list = extract_output_annotations(
            mock_detectron2_output
        )
    assert isinstance(mask_arrays, list)
    assert isinstance(polygons, list)
    assert isinstance(bbox_list, list)
    assert isinstance(labels_list, list)


def test_extract_output_annotations_flatten_true(mock_detectron2_output):
    fake_poly = np.array([[5, 5], [9, 5], [9, 9], [5, 9]])
    with patch("aigis.segment.utils.sv.mask_to_polygons", return_value=[fake_poly]):
        _, polygons, _, _ = extract_output_annotations(
            mock_detectron2_output, flatten=True
        )
    assert len(polygons) > 0
    # Flattened polygon is a 1-D list of numbers
    for poly in polygons:
        assert isinstance(poly, list)
        for coord in poly:
            assert isinstance(coord, (int, float))


def test_extract_output_annotations_flatten_false(mock_detectron2_output):
    fake_poly = np.array([[5, 5], [9, 5], [9, 9], [5, 9]])
    with patch("aigis.segment.utils.sv.mask_to_polygons", return_value=[fake_poly]):
        _, polygons, _, _ = extract_output_annotations(
            mock_detectron2_output, flatten=False
        )
    assert len(polygons) > 0
    # Not flattened: each element is a list of [x, y] pairs
    for poly in polygons:
        assert isinstance(poly, list)
        for point in poly:
            assert len(point) == 2


def test_extract_output_annotations_empty_mask_warns(mock_detectron2_output):
    with patch("aigis.segment.utils.sv.mask_to_polygons", return_value=[]):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mask_arrays, polygons, bbox_list, labels_list = extract_output_annotations(
                mock_detectron2_output
            )
        assert polygons == []
        assert any("empty" in str(warning.message).lower() for warning in w)


# ---------------------------------------------------------------------------
# assemble_coco_json
# ---------------------------------------------------------------------------


def _make_png(path, size=(20, 20)):
    """Write a tiny PNG to disk for use as an image path."""
    img = Image.fromarray(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    img.save(path)


def test_assemble_coco_json_has_images_annotations_categories(tmp_path):
    # Create two tiny PNG images
    img_paths = []
    for i in range(2):
        p = str(tmp_path / f"img_{i}.png")
        _make_png(p)
        img_paths.append(p)

    annotations = pd.DataFrame(
        {
            "pixel_polygon": [
                [(0, 0), (5, 0), (5, 5), (0, 5)],
                [(2, 2), (8, 2), (8, 8), (2, 8)],
            ],
            "image_id": [0, 1],
            "class_id": [0, 1],
            "annot_id": [0, 1],
        }
    )

    result = assemble_coco_json(annotations, img_paths)

    assert hasattr(result, "images")
    assert hasattr(result, "annotations")
    assert hasattr(result, "categories")


def test_assemble_coco_json_image_count(tmp_path):
    img_paths = []
    for i in range(3):
        p = str(tmp_path / f"img_{i}.png")
        _make_png(p)
        img_paths.append(p)

    annotations = pd.DataFrame(
        {
            "pixel_polygon": [[(0, 0), (5, 0), (5, 5)]] * 3,
            "image_id": [0, 1, 2],
            "class_id": [0, 0, 0],
            "annot_id": [0, 1, 2],
        }
    )

    result = assemble_coco_json(annotations, img_paths)
    assert len(result.images) == 3
