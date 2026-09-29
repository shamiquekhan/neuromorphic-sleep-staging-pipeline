"""Cache directory resolution — the Streamlit Cloud demo fallback.

A clean checkout (Streamlit Cloud) contains ``data/cache_full/cache_index.csv``
but no epoch files. The resolver must not treat that committed index as a
populated cache, otherwise the deployed app resolves an empty directory,
finds zero subjects, and never reaches the committed ``data/cache_demo``
subset.
"""

from pathlib import Path

import sleep_staging.config as config


def _mk(tmp_path: Path, rel: str, files: list[str]) -> Path:
    d = tmp_path / rel
    d.mkdir(parents=True, exist_ok=True)
    for name in files:
        (d / name).write_bytes(b"")
    return d


def test_bare_index_does_not_shadow_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    _mk(tmp_path, "data/cache_full", ["cache_index.csv"])
    demo = _mk(tmp_path, "data/cache_demo", ["AGE_001_N1_meta.npz"])
    assert config._resolve_cache_dir() == demo


def test_populated_full_cache_beats_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    full = _mk(
        tmp_path, "data/cache_full",
        ["cache_index.csv", "SC4001N1_meta.npz", "SC4001N1_epochs.npy"],
    )
    _mk(tmp_path, "data/cache_demo", ["AGE_001_N1_meta.npz"])
    assert config._resolve_cache_dir() == full


def test_legacy_night_cache_still_selected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    legacy = _mk(tmp_path, "data/cache", ["SC4001N1_nightE0.npz"])
    assert config._resolve_cache_dir() == legacy


def test_empty_checkout_defaults_to_cache_full(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    assert config._resolve_cache_dir() == tmp_path / "data" / "cache_full"


def test_deployed_layout_resolves_demo(tmp_path, monkeypatch):
    """Simulate the Streamlit Cloud checkout exactly."""
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    _mk(tmp_path, "data/cache_full", ["cache_index.csv", "preprocessing_summary.json"])
    _mk(tmp_path, "data/cache", ["cache_index.csv"])
    demo = _mk(
        tmp_path, "data/cache_demo",
        ["AGE_001_N1_meta.npz", "AGE_001_N1_epochs.npy", "cache_index.csv"],
    )
    assert config._resolve_cache_dir() == demo
