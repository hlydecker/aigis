# -*- coding: utf-8 -*-
"""Tests for aigis.convert.tiles."""
import os

import pytest
import rasterio

from aigis.convert.tiles import (
    get_tiles,
    get_tiles_list_from_dir,
    save_tiles,
    tile_neighbourhood_list,
)


# ---------------------------------------------------------------------------
# get_tiles
# ---------------------------------------------------------------------------


def test_get_tiles_count_no_offset(open_geotiff):
    # 100×100 raster, tile_size=50 → 2×2 = 4 windows
    windows = list(get_tiles(open_geotiff, tile_width=50, tile_height=50, map_units=False))
    assert len(windows) == 4


def test_get_tiles_with_offset_windows_overlap(open_geotiff):
    windows_no_offset = list(
        get_tiles(open_geotiff, tile_width=50, tile_height=50, map_units=False, offset=0.0)
    )
    windows_with_offset = list(
        get_tiles(open_geotiff, tile_width=50, tile_height=50, map_units=False, offset=10.0)
    )
    # With offset, windows are larger; their widths may exceed the tile_size
    assert len(windows_with_offset) == len(windows_no_offset)  # same tile count
    # At least one tile should be larger when offset > 0
    areas_no_off = [w.width * w.height for w, _ in windows_no_offset]
    areas_with_off = [w.width * w.height for w, _ in windows_with_offset]
    assert max(areas_with_off) >= max(areas_no_off)


def test_get_tiles_yields_window_and_transform(open_geotiff):
    for window, transform in get_tiles(open_geotiff, tile_width=50, tile_height=50):
        assert hasattr(window, "col_off")
        assert transform is not None
        break  # only need to check one


# ---------------------------------------------------------------------------
# save_tiles
# ---------------------------------------------------------------------------


def test_save_tiles_writes_four_files(tmp_path, small_geotiff):
    out_dir = str(tmp_path / "tiles")
    with rasterio.open(small_geotiff) as src:
        save_tiles(src, out_dir, tile_size=50, map_units=False)
    tif_files = [f for f in os.listdir(out_dir) if f.endswith(".tif")]
    assert len(tif_files) == 4


def test_save_tiles_valid_rasterio_files(tmp_path, small_geotiff):
    out_dir = str(tmp_path / "tiles")
    with rasterio.open(small_geotiff) as src:
        save_tiles(src, out_dir, tile_size=50, map_units=False)
    for fname in os.listdir(out_dir):
        if fname.endswith(".tif"):
            with rasterio.open(os.path.join(out_dir, fname)) as ds:
                assert ds.width > 0
                assert ds.height > 0


def test_save_tiles_naming_convention(tmp_path, small_geotiff):
    out_dir = str(tmp_path / "tiles")
    with rasterio.open(small_geotiff) as src:
        save_tiles(src, out_dir, tile_size=50, map_units=False)
    for fname in os.listdir(out_dir):
        if fname.endswith(".tif"):
            # Expected pattern: tile_{col}-{row}.tif
            assert fname.startswith("tile_")
            stem = fname[len("tile_"):-len(".tif")]
            parts = stem.split("-")
            assert len(parts) == 2
            assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# tile_neighbourhood_list
# ---------------------------------------------------------------------------


def test_tile_neighbourhood_2x2_all_tiles_are_neighbours(tile_filenames):
    # In a 2×2 grid every tile touches all others (all are diagonal neighbours)
    result = tile_neighbourhood_list(tile_filenames)
    for tile, info in result.items():
        assert len(info["neighbour_tiles"]) == 3


def test_tile_neighbourhood_single_tile():
    result = tile_neighbourhood_list(["tile_0-0.tif"])
    assert result["tile_0-0.tif"]["neighbour_tiles"] == []


def test_tile_neighbourhood_returns_dict(tile_filenames):
    result = tile_neighbourhood_list(tile_filenames)
    assert isinstance(result, dict)
    assert len(result) == len(tile_filenames)


# ---------------------------------------------------------------------------
# get_tiles_list_from_dir
# ---------------------------------------------------------------------------


def test_get_tiles_list_from_dir_only_tif(tmp_path):
    for name in ["a.tif", "b.tif", "c.png", "d.json"]:
        (tmp_path / name).touch()
    result = get_tiles_list_from_dir(str(tmp_path), extension="tif")
    basenames = [os.path.basename(p) for p in result]
    assert sorted(basenames) == ["a.tif", "b.tif"]


def test_get_tiles_list_from_dir_empty(tmp_path):
    result = get_tiles_list_from_dir(str(tmp_path), extension="tif")
    assert result == []
