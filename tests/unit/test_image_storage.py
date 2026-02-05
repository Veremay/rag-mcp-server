from pathlib import Path

from src.ingestion.storage.image_storage import ImageStorage


def test_save_and_get_path_roundtrip(tmp_path: Path) -> None:
    base_dir = tmp_path / "images"
    storage = ImageStorage(base_dir=base_dir)

    image_id = "img_001"
    saved_path = storage.save(
        collection="c1",
        image_id=image_id,
        data=b"\x89PNG\r\n\x1a\nfake",
        ext=".png",
    )
    assert saved_path.exists()
    assert saved_path.name == f"{image_id}.png"

    got = storage.get_path(collection="c1", image_id=image_id)
    assert got == saved_path.resolve()

    storage2 = ImageStorage(base_dir=base_dir)
    got2 = storage2.get_path(collection="c1", image_id=image_id)
    assert got2 == saved_path.resolve()


def test_get_path_missing_returns_none(tmp_path: Path) -> None:
    storage = ImageStorage(base_dir=tmp_path / "images")
    assert storage.get_path(collection="c1", image_id="missing") is None


def test_save_overwrite_updates_index(tmp_path: Path) -> None:
    base_dir = tmp_path / "images"
    storage = ImageStorage(base_dir=base_dir)

    image_id = "img_001"
    p1 = storage.save(collection="c1", image_id=image_id, data=b"a", ext=".png")
    p2 = storage.save(collection="c1", image_id=image_id, data=b"b", ext=".jpg")

    assert p1.exists()
    assert p2.exists()
    assert storage.get_path(collection="c1", image_id=image_id) == p2.resolve()
