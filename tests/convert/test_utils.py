# -*- coding: utf-8 -*-
"""Tests for aigis.convert.utils."""
import json
import os

import pandas as pd
import pytest

from aigis.convert.utils import condense_csv, recode_file_names


# ---------------------------------------------------------------------------
# condense_csv
# ---------------------------------------------------------------------------


def test_condense_csv_returns_correct_rows(sample_csv_path):
    result = condense_csv(sample_csv_path)
    # img1 → class_a, img2 → class_b, img3 → all zero (no output)
    assert len(result) == 2


def test_condense_csv_columns(sample_csv_path):
    result = condense_csv(sample_csv_path)
    assert list(result.columns) == ["filename", "class"]


def test_condense_csv_correct_mappings(sample_csv_path):
    result = condense_csv(sample_csv_path)
    row_a = result[result["class"] == "class_a"]
    assert len(row_a) == 1
    assert row_a.iloc[0]["filename"] == "img1.jpg"

    row_b = result[result["class"] == "class_b"]
    assert len(row_b) == 1
    assert row_b.iloc[0]["filename"] == "img2.jpg"


def test_condense_csv_all_zero_row_no_output(tmp_path):
    df = pd.DataFrame({"filename": ["x.jpg"], "class_a": [0], "class_b": [0]})
    path = str(tmp_path / "zero.csv")
    df.to_csv(path, index=False)
    result = condense_csv(path)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# recode_file_names
# ---------------------------------------------------------------------------


def _write_coco(path, file_names):
    data = {
        "images": [{"id": i, "file_name": fn} for i, fn in enumerate(file_names)],
        "annotations": [],
        "categories": [],
    }
    with open(path, "w") as f:
        json.dump(data, f)


def test_recode_file_names_transforms_slash(tmp_path):
    src = str(tmp_path / "input.json")
    out = str(tmp_path / "output.json")
    _write_coco(src, ["abc/def.jpg"])

    recode_file_names(src, str(tmp_path), out)

    with open(out) as f:
        data = json.load(f)
    assert data["images"][0]["file_name"] == "abc.jpg"


def test_recode_file_names_original_unchanged(tmp_path):
    src = str(tmp_path / "input.json")
    out = str(tmp_path / "output.json")
    _write_coco(src, ["abc/def.jpg"])

    recode_file_names(src, str(tmp_path), out)

    with open(src) as f:
        original = json.load(f)
    assert original["images"][0]["file_name"] == "abc/def.jpg"


def test_recode_file_names_multiple_images(tmp_path):
    src = str(tmp_path / "input.json")
    out = str(tmp_path / "output.json")
    _write_coco(src, ["folder1/img1.jpg", "folder2/img2.jpg"])

    recode_file_names(src, str(tmp_path), out)

    with open(out) as f:
        data = json.load(f)
    names = [img["file_name"] for img in data["images"]]
    assert names == ["folder1.jpg", "folder2.jpg"]
