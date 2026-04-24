from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "geo" / "augmentation" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "geo" / "augmentation" / "cdg.json"
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "geo_augmentation.review_bundle.json"

EXPECTED_FQDNS = {
    "sciona.atoms.geo.augmentation.gsd_aware_random_crop",
    "sciona.atoms.geo.augmentation.gsd_aware_shift_scale_rotate",
}


def test_augmentation_atoms_import() -> None:
    from sciona.atoms.geo.augmentation.atoms import gsd_aware_random_crop, gsd_aware_shift_scale_rotate

    assert callable(gsd_aware_random_crop)
    assert callable(gsd_aware_shift_scale_rotate)


def test_gsd_aware_random_crop_returns_requested_shape() -> None:
    from sciona.atoms.geo.augmentation.atoms import gsd_aware_random_crop

    image = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    np.random.seed(7)
    cropped = gsd_aware_random_crop(image, gsd=1.0, target_gsd=1.0, crop_size=32)
    assert cropped.shape == (32, 32, 3)


def test_gsd_aware_random_crop_rescales_before_cropping() -> None:
    from sciona.atoms.geo.augmentation.atoms import gsd_aware_random_crop

    image = np.tile(np.arange(8, dtype=np.uint8), (8, 1))
    manual_rng = np.random.RandomState(0)
    expected_top = int(manual_rng.randint(0, 5))
    expected_left = int(manual_rng.randint(0, 5))
    resized = cv2.resize(image, (16, 16), interpolation=cv2.INTER_CUBIC)
    expected = resized[expected_top : expected_top + 12, expected_left : expected_left + 12]

    np.random.seed(0)
    cropped = gsd_aware_random_crop(image, gsd=2.0, target_gsd=1.0, crop_size=12)

    assert cropped.shape == (12, 12)
    np.testing.assert_array_equal(cropped, expected)


def test_gsd_aware_random_crop_pads_when_scaled_image_is_too_small() -> None:
    from sciona.atoms.geo.augmentation.atoms import gsd_aware_random_crop

    image = np.ones((4, 4), dtype=np.uint8)
    np.random.seed(0)
    cropped = gsd_aware_random_crop(image, gsd=0.5, target_gsd=1.0, crop_size=6)

    assert cropped.shape == (6, 6)
    assert cropped.dtype == image.dtype


def test_gsd_aware_shift_scale_rotate_identity_when_limits_are_zero() -> None:
    from sciona.atoms.geo.augmentation.atoms import gsd_aware_shift_scale_rotate

    image = np.arange(25, dtype=np.uint8).reshape(5, 5)
    transformed, updated_gsd = gsd_aware_shift_scale_rotate(
        image,
        gsd=1.5,
        shift_limit=0.0,
        scale_limit=0.0,
        rotate_limit=0.0,
        rng=np.random.default_rng(123),
    )

    np.testing.assert_array_equal(transformed, image)
    assert updated_gsd == 1.5


def test_gsd_aware_shift_scale_rotate_updates_gsd_from_sampled_scale() -> None:
    from sciona.atoms.geo.augmentation.atoms import gsd_aware_shift_scale_rotate

    image = np.zeros((16, 16), dtype=np.uint8)
    control_rng = np.random.default_rng(42)
    _ = control_rng.uniform(-0.1, 0.1)
    _ = control_rng.uniform(-0.1, 0.1)
    expected_scale = 1.0 + float(control_rng.uniform(-0.2, 0.2))

    transformed, updated_gsd = gsd_aware_shift_scale_rotate(
        image,
        gsd=2.0,
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=10.0,
        rng=np.random.default_rng(42),
    )

    assert transformed.shape == image.shape
    np.testing.assert_allclose(updated_gsd, 2.0 * max(expected_scale, 1e-6), rtol=1e-6)


def test_gsd_aware_shift_scale_rotate_preserves_shape_for_rgb_images() -> None:
    from sciona.atoms.geo.augmentation.atoms import gsd_aware_shift_scale_rotate

    image = np.zeros((20, 24, 3), dtype=np.uint8)
    transformed, updated_gsd = gsd_aware_shift_scale_rotate(
        image,
        gsd=1.0,
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=15.0,
        rng=np.random.default_rng(1),
    )

    assert transformed.shape == image.shape
    assert updated_gsd > 0.0


def test_augmentation_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    assert {key.partition("@")[0] for key in payload["atoms"]} == EXPECTED_FQDNS


def test_augmentation_reference_ids_exist_in_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])

    for entry in payload["atoms"].values():
        for reference in entry["references"]:
            assert reference["ref_id"] in registry_ids
            assert reference["match_metadata"]["notes"]


def test_augmentation_cdg_contains_expected_nodes() -> None:
    payload = json.loads(CDG_PATH.read_text(encoding="utf-8"))
    node_ids = {node["node_id"] for node in payload["nodes"]}
    assert {"gsd_aware_random_crop", "gsd_aware_shift_scale_rotate"} <= node_ids


def test_augmentation_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.geo.augmentation.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        leaf = fqdn.removeprefix("sciona.atoms.geo.augmentation.")
        assert leaf in registered


def test_augmentation_review_bundle_exists_and_lists_expected_atoms() -> None:
    payload = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    assert payload["provider_repo"] == "sciona-atoms-geo"
    assert payload["review_status"] == "pending"
    assert {row["atom_key"] for row in payload["rows"]} == EXPECTED_FQDNS
