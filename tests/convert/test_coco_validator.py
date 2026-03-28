# -*- coding: utf-8 -*-
"""Tests for aigis.convert.COCO_validator.

Bug #2 note: annotation_assertions() references module-global `coco_data`
(only defined in the __main__ block), not as a parameter.  Tests that
exercise annotation_assertions via main() must set the module-level
variable first as a workaround.
"""
import pytest

import aigis.convert.COCO_validator as validator


# ---------------------------------------------------------------------------
# assertions() — pure helper, no module-global dependency
# ---------------------------------------------------------------------------


def test_assertions_valid_returns_id_map():
    values = [{"id": 1, "name": "tree", "supercategory": "landuse"}]
    result = validator.assertions(
        "categories", values, ["id", "name", "supercategory"], "name"
    )
    assert result == {1: "tree"}


def test_assertions_missing_required_key_raises():
    # unique_key lookup happens before required-key assertions, so omit
    # unique_key here to test the AssertionError path cleanly.
    values = [{"id": 1}]  # missing "name"
    with pytest.raises(AssertionError, match="name"):
        validator.assertions("categories", values, ["id", "name"], unique_key=None)


def test_assertions_empty_list_returns_empty_dict():
    result = validator.assertions("categories", [], ["id", "name"], "name")
    assert result == {}


def test_assertions_no_unique_key_returns_empty_dict():
    values = [{"id": 1, "name": "tree"}]
    result = validator.assertions("categories", values, ["id", "name"], unique_key=None)
    assert result == {}


# ---------------------------------------------------------------------------
# main() — end-to-end validation (workaround for Bug #2)
# ---------------------------------------------------------------------------


def _set_coco_data(d):
    """Set module-level coco_data to satisfy annotation_assertions (Bug #2)."""
    validator.coco_data = d


def test_main_valid(minimal_coco_dict):
    _set_coco_data(minimal_coco_dict)
    validator.main(minimal_coco_dict)  # should not raise


def test_main_missing_top_level_key_raises(minimal_coco_dict):
    del minimal_coco_dict["type"]
    with pytest.raises(AssertionError):
        validator.main(minimal_coco_dict)


def test_main_empty_required_section_raises(minimal_coco_dict):
    minimal_coco_dict["images"] = []
    with pytest.raises(AssertionError):
        validator.main(minimal_coco_dict)


def test_main_bad_bbox_length_raises(minimal_coco_dict):
    minimal_coco_dict["annotations"][0]["bbox"] = [0, 0, 10]  # only 3 items
    _set_coco_data(minimal_coco_dict)
    with pytest.raises(AssertionError, match="bbox"):
        validator.main(minimal_coco_dict)


def test_main_invalid_iscrowd_raises(minimal_coco_dict):
    minimal_coco_dict["annotations"][0]["iscrowd"] = 2
    _set_coco_data(minimal_coco_dict)
    with pytest.raises(AssertionError, match="iscrowd"):
        validator.main(minimal_coco_dict)


def test_main_category_id_not_in_map_raises(minimal_coco_dict):
    minimal_coco_dict["annotations"][0]["category_id"] = 999
    _set_coco_data(minimal_coco_dict)
    with pytest.raises(AssertionError, match="category_id"):
        validator.main(minimal_coco_dict)


def test_main_image_id_not_in_map_raises(minimal_coco_dict):
    minimal_coco_dict["annotations"][0]["image_id"] = 999
    _set_coco_data(minimal_coco_dict)
    with pytest.raises(AssertionError, match="image_id"):
        validator.main(minimal_coco_dict)
