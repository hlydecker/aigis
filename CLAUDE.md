# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIGIS is a Python toolkit for aerial/satellite imagery processing using AI — annotation, conversion, and segmentation of geospatial imagery. It bridges geospatial formats (GeoJSON, GeoTIFF, Shapefile) with computer vision formats (COCO JSON) and provides end-to-end workflows from raw imagery to georeferenced predictions.

## Installation

```bash
conda create -n aigis python=3.9
conda activate aigis
pip install -e aigis
```

Key dependencies require separate installation (not bundled):
- Detectron2: install from source per [official instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
- GroundingDINO: `pip install groundingdino-py`

## Commands

**Run tests:**
```bash
pytest tests/
pytest tests/test_extract_output_annotations.py  # single test file
```

**Lint and format (via pre-commit):**
```bash
pre-commit run --all-files
black aigis/ scripts/
isort aigis/ scripts/
flake8 aigis/ scripts/
```

**Build docs:**
```bash
cd sphinx-docs && make html
```

## Architecture

The codebase has two layers: a **library** (`aigis/`) and **CLI scripts** (`scripts/`).

### Library (`aigis/`)

- **`aigis.convert`** — Core data transformation logic:
  - `coco.py`: COCO JSON dataclasses and construction helpers (`coco_json`, `coco_image`, `coco_poly_ann`)
  - `coordinates.py`: Pixel ↔ spatial coordinate conversion using rasterio transforms
  - `tiles.py`: GeoTIFF tiling and spatial grid generation
  - `utils.py`: CSV condensing, file organization, COCO filename recoding
  - `COCO_validator.py`: Assertion-based COCO format validation
  - `orthogonalise/`: Polygon orthogonalization for building footprints

- **`aigis.segment`** — Inference and post-processing:
  - `postprocess.py`: `detectron2_to_polygons()` extracts masks/polygons from Detectron2 outputs; `convert_polygons_to_geospatial()` georeferences pixel-space polygons
  - `eval.py`: `SegmentationModelEvaluator` class for IoU and confusion matrices
  - `utils.py`: Polygon simplification/rotation, GIF generation, Detectron2 visualization

- **`aigis.annotate`** — Mask visualization and GeoJSON filtering utilities

### CLI Scripts (`scripts/`)

Scripts are standalone entry points that import from the library. Key workflows:

| Script | Purpose |
|--------|---------|
| `geojson2coco.py` | Vector annotations + raster → tiled COCO dataset |
| `coco2geojson.py` | COCO predictions → georeferenced GeoJSON |
| `make_mask.py` | Text-prompted annotation via SAM + GroundingDINO |
| `fine_tuning_detectron2.py` | Mask R-CNN fine-tuning with wandb logging |
| `prediction_raster_detectron2.py` | Batch tile inference on GeoTIFF |
| `coco_split.py` | Train/val/test dataset splitting |
| `coco_balance.py` | Class balancing and sampling |
| `osm_cleaner.py` | OpenStreetMap vector data cleaning |

### Data Flow

```
Raw imagery + vector labels
    → geojson2coco.py (tiles raster, converts annotations to COCO)
    → coco_split.py / coco_balance.py (dataset prep)
    → fine_tuning_detectron2.py (train model)
    → prediction_raster_detectron2.py (inference)
    → postprocess.py (pixel polygons → geospatial polygons)
    → coco2geojson.py (COCO predictions → GeoJSON)
```

### Code Style

- Formatter: **black** (line length not enforced — E501 ignored in flake8)
- Import sorting: **isort** with black profile
- Flake8 ignores: E501, W503, E203, E265
