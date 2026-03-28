# -*- coding: utf-8 -*-
"""Shared fixtures for the AIGIS test suite."""
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from shapely.geometry import Polygon
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Tier 1 — Pure Python / Shapely
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_square_polygon():
    """10x10 square polygon with area=100."""
    return Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


@pytest.fixture
def triangle_polygon():
    """3-vertex polygon for edge cases."""
    return Polygon([(0, 0), (5, 10), (10, 0)])


@pytest.fixture
def minimal_coco_dict():
    """Smallest valid COCO dict (area == bbox[2]*bbox[3])."""
    return {
        "type": "instances",
        "images": [
            {"id": 1, "file_name": "test.jpg", "height": 100, "width": 100}
        ],
        "categories": [
            {"id": 1, "name": "tree", "supercategory": "landuse"}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
                "area": 100.0,
                "bbox": [0, 0, 10, 10],
                "iscrowd": 0,
            }
        ],
    }


@pytest.fixture
def coco_image_entry():
    return {"id": 1, "file_name": "image.jpg", "height": 200, "width": 200}


@pytest.fixture
def coco_category_entry():
    return {"id": 1, "name": "building", "supercategory": "landuse"}


@pytest.fixture
def coco_annotation_entry():
    return {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]],
        "area": 100.0,
        "bbox": [0, 0, 10, 10],
        "iscrowd": 0,
    }


@pytest.fixture
def sample_csv_path(tmp_path):
    """Roboflow-style multiclass CSV written to disk; returns path."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "filename": ["img1.jpg", "img2.jpg", "img3.jpg"],
            "class_a": [1, 0, 0],
            "class_b": [0, 1, 0],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


# ---------------------------------------------------------------------------
# Tier 2 — Synthetic Raster (tmp_path + rasterio)
# ---------------------------------------------------------------------------


@pytest.fixture
def small_geotiff(tmp_path):
    """100×100 3-band uint8 GeoTIFF, EPSG:32755, 1 m pixels. Returns path."""
    path = str(tmp_path / "test.tif")
    transform = from_bounds(0, 0, 100, 100, 100, 100)
    rng = np.random.default_rng(42)
    data = rng.integers(0, 255, (3, 100, 100), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=3,
        dtype="uint8",
        crs="EPSG:32755",
        transform=transform,
    ) as dst:
        dst.write(data)
    return path


@pytest.fixture
def open_geotiff(small_geotiff):
    """Opens small_geotiff and yields a rasterio DatasetReader."""
    src = rasterio.open(small_geotiff)
    yield src
    src.close()


@pytest.fixture
def tile_filenames():
    """Four tile filenames matching tile_X-Y.tif convention (no disk I/O)."""
    return [
        "tile_0-0.tif",
        "tile_50-0.tif",
        "tile_0-50.tif",
        "tile_50-50.tif",
    ]


# ---------------------------------------------------------------------------
# Tier 3 — Mock ML Outputs
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_detectron2_instances():
    """MagicMock mimicking Detectron2 Instances with N=2, H=20, W=20."""
    H, W, N = 20, 20, 2
    mock = MagicMock()

    masks = np.zeros((N, H, W), dtype=bool)
    masks[0, 5:10, 5:10] = True
    masks[1, 12:18, 12:18] = True
    mock.pred_masks.to.return_value.numpy.return_value = masks

    mock.pred_classes.to.return_value.numpy.return_value = np.array([0, 1])

    boxes_mock = MagicMock()
    boxes_mock.numpy.return_value = np.array(
        [[5, 5, 10, 10], [12, 12, 18, 18]], dtype=float
    )
    mock.pred_boxes.to.return_value.tensor = boxes_mock

    return mock


@pytest.fixture
def mock_detectron2_output(mock_detectron2_instances):
    """Wraps mock_detectron2_instances as a Detectron2-style output dict."""
    return {"instances": mock_detectron2_instances}
