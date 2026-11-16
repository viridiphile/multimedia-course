from pathlib import Path

import pytest

from src.dataset import build_manifest, parse_filename


def test_parse_coil_filename():
    assert parse_filename("obj7__180.png") == (7, 180)
    with pytest.raises(ValueError):
        parse_filename("obj7_180.png")
    with pytest.raises(ValueError):
        parse_filename("obj101__180.png")


def test_query_gallery_split_is_deterministic_and_disjoint(tmp_path: Path):
    names = [
        "obj1__0.png",
        "obj1__5.png",
        "obj1__180.png",
        "obj1__185.png",
        "obj2__0.png",
        "obj2__5.png",
    ]
    for index, name in enumerate(names):
        (tmp_path / name).write_bytes(("unique-content-{}".format(index)).encode())

    records = build_manifest(
        tmp_path, strict_dataset=False, validate_content=True
    )
    assert [record.image_id for record in records] == [
        "obj1__0",
        "obj1__5",
        "obj1__180",
        "obj1__185",
        "obj2__0",
        "obj2__5",
    ]
    queries = {record.image_id for record in records if record.split == "query"}
    gallery = {record.image_id for record in records if record.split == "gallery"}
    assert queries == {"obj1__0", "obj1__180", "obj2__0"}
    assert gallery == {"obj1__5", "obj1__185", "obj2__5"}
    assert not queries.intersection(gallery)


def test_duplicate_views_are_rejected(tmp_path: Path):
    (tmp_path / "obj1__0.png").write_bytes(b"first")
    (tmp_path / "obj1__0.jpg").write_bytes(b"second")
    with pytest.raises(ValueError, match="Duplicate object/view"):
        build_manifest(tmp_path, strict_dataset=False, validate_content=False)


def test_duplicate_image_content_is_rejected(tmp_path: Path):
    (tmp_path / "obj1__0.png").write_bytes(b"same bytes")
    (tmp_path / "obj1__5.png").write_bytes(b"same bytes")
    with pytest.raises(ValueError, match="Duplicate image content"):
        build_manifest(tmp_path, strict_dataset=False, validate_content=True)


def test_strict_protocol_rejects_incomplete_dataset(tmp_path: Path):
    (tmp_path / "obj1__0.png").write_bytes(b"one")
    with pytest.raises(ValueError, match="view inventory mismatch"):
        build_manifest(tmp_path, strict_dataset=True, validate_content=False)
