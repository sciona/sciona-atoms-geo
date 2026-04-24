from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
REFERENCES_PATH = ROOT / "src" / "sciona" / "atoms" / "geo" / "losses" / "references.json"
REGISTRY_PATH = ROOT / "data" / "references" / "registry.json"
CDG_PATH = ROOT / "src" / "sciona" / "atoms" / "geo" / "losses" / "cdg.json"
BUNDLE_PATH = ROOT / "data" / "review_bundles" / "geo_losses.review_bundle.json"

EXPECTED_FQDNS = {
    "sciona.atoms.geo.losses.r2_regression_loss",
    "sciona.atoms.geo.losses.circular_direction_loss",
}


def test_loss_atoms_import() -> None:
    from sciona.atoms.geo.losses.atoms import circular_direction_loss, r2_regression_loss

    assert callable(r2_regression_loss)
    assert callable(circular_direction_loss)


def test_r2_regression_loss_zero_for_perfect_predictions() -> None:
    from sciona.atoms.geo.losses.atoms import r2_regression_loss

    targets = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    result = r2_regression_loss(targets.copy(), targets)
    assert result == pytest.approx(0.0, abs=1e-7)


def test_r2_regression_loss_matches_manual_formula() -> None:
    from sciona.atoms.geo.losses.atoms import r2_regression_loss

    predictions = np.array([0.0, 3.0, 5.0], dtype=np.float64)
    targets = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    target_mean = 2.0

    expected = ((1.0**2 + 1.0**2 + 1.0**2) + 1e-8) / (((-1.0) ** 2 + 0.0**2 + 2.0**2) + 1e-8)
    assert r2_regression_loss(predictions, targets, target_mean=target_mean) == pytest.approx(expected)


def test_r2_regression_loss_default_target_mean_matches_explicit_mean() -> None:
    from sciona.atoms.geo.losses.atoms import r2_regression_loss

    predictions = np.array([3.0, 2.0, 6.0, 7.0], dtype=np.float64)
    targets = np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float64)
    expected = r2_regression_loss(predictions, targets, target_mean=float(np.mean(targets)))
    observed = r2_regression_loss(predictions, targets)
    assert observed == pytest.approx(expected)


def test_r2_regression_loss_worse_predictions_increase_loss() -> None:
    from sciona.atoms.geo.losses.atoms import r2_regression_loss

    targets = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    better = np.array([1.2, 2.8, 5.1, 6.9], dtype=np.float64)
    worse = np.array([7.0, 5.0, 3.0, 1.0], dtype=np.float64)

    assert r2_regression_loss(better, targets) < r2_regression_loss(worse, targets)


def test_circular_direction_loss_zero_for_matching_angles() -> None:
    from sciona.atoms.geo.losses.atoms import circular_direction_loss

    target_angle = np.array([0.0, np.pi / 4, np.pi], dtype=np.float64)
    pred_sin = np.sin(target_angle)
    pred_cos = np.cos(target_angle)

    assert circular_direction_loss(pred_sin, pred_cos, target_angle) == pytest.approx(0.0, abs=5e-4)


def test_circular_direction_loss_wraps_cleanly_across_two_pi() -> None:
    from sciona.atoms.geo.losses.atoms import circular_direction_loss

    target_angle = np.array([0.0], dtype=np.float64)
    predicted_angle = np.array([2.0 * np.pi - 1e-3], dtype=np.float64)

    loss = circular_direction_loss(np.sin(predicted_angle), np.cos(predicted_angle), target_angle)
    assert loss < 0.01


def test_circular_direction_loss_is_large_for_opposite_directions() -> None:
    from sciona.atoms.geo.losses.atoms import circular_direction_loss

    target_angle = np.array([0.0, 0.0], dtype=np.float64)
    pred_sin = np.array([0.0, 0.0], dtype=np.float64)
    pred_cos = np.array([-1.0, -1.0], dtype=np.float64)

    loss = circular_direction_loss(pred_sin, pred_cos, target_angle)
    assert loss == pytest.approx(np.pi, rel=1e-4)


def test_r2_regression_loss_torch_matches_numpy() -> None:
    torch = pytest.importorskip("torch")
    from sciona.atoms.geo.losses.atoms import r2_regression_loss
    from sciona.atoms.geo.losses.atoms_torch import r2_regression_loss_torch

    predictions = np.array([0.5, 2.5, 3.5], dtype=np.float64)
    targets = np.array([1.0, 2.0, 4.0], dtype=np.float64)

    numpy_loss = r2_regression_loss(predictions, targets)
    torch_loss = r2_regression_loss_torch(torch.tensor(predictions), torch.tensor(targets))
    assert torch_loss.item() == pytest.approx(numpy_loss, rel=1e-6)


def test_r2_regression_loss_torch_backpropagates() -> None:
    torch = pytest.importorskip("torch")
    from sciona.atoms.geo.losses.atoms_torch import r2_regression_loss_torch

    predictions = torch.tensor([0.5, 2.5, 3.5], requires_grad=True)
    targets = torch.tensor([1.0, 2.0, 4.0])

    loss = r2_regression_loss_torch(predictions, targets)
    loss.backward()

    assert predictions.grad is not None
    assert predictions.grad.shape == predictions.shape


def test_losses_references_json_has_expected_fqdns() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    atom_keys = set(payload["atoms"])
    assert {key.partition("@")[0] for key in atom_keys} == EXPECTED_FQDNS


def test_losses_reference_ids_exist_in_registry() -> None:
    payload = json.loads(REFERENCES_PATH.read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    registry_ids = set(registry["references"])

    for entry in payload["atoms"].values():
        for reference in entry["references"]:
            assert reference["ref_id"] in registry_ids
            assert reference["match_metadata"]["notes"]


def test_losses_cdg_contains_callable_injection_edge() -> None:
    payload = json.loads(CDG_PATH.read_text(encoding="utf-8"))
    assert any(
        edge["source"] == "r2_regression_loss"
        and edge["target"] == "geopose_citywise_r2_training_objective"
        and edge["kind"] == "callable_injection"
        for edge in payload["edges"]
    )


def test_loss_atom_leaf_names_are_registered() -> None:
    import_module("sciona.atoms.geo.losses.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        leaf = fqdn.removeprefix("sciona.atoms.geo.losses.")
        assert leaf in registered


def test_loss_review_bundle_exists_and_lists_expected_atoms() -> None:
    payload = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    assert payload["provider_repo"] == "sciona-atoms-geo"
    assert payload["review_status"] == "pending"
    assert {row["atom_key"] for row in payload["rows"]} == EXPECTED_FQDNS
