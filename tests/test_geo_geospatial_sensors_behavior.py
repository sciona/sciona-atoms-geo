from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import numpy as np

from sciona.ghost.registry import REGISTRY


ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "src/sciona/atoms/geo/geospatial_sensors"
BUNDLE_PATH = ROOT / "data/review_bundles/geo_geospatial_sensors.review_bundle.json"
REGISTRY_PATH = ROOT / "data/references/registry.json"

EXPECTED_FQDNS = {
    "sciona.atoms.geo.geospatial_sensors.lla_to_ecef",
    "sciona.atoms.geo.geospatial_sensors.ecef_to_lla",
    "sciona.atoms.geo.geospatial_sensors.ecef_to_enu",
    "sciona.atoms.geo.geospatial_sensors.correct_clock_bias",
    "sciona.atoms.geo.geospatial_sensors.filter_by_cn0",
    "sciona.atoms.geo.geospatial_sensors.filter_multipath",
    "sciona.atoms.geo.geospatial_sensors.detect_steps",
    "sciona.atoms.geo.geospatial_sensors.estimate_step_length_weinberg",
    "sciona.atoms.geo.geospatial_sensors.integrate_heading",
    "sciona.atoms.geo.geospatial_sensors.pdr_position_update",
    "sciona.atoms.geo.geospatial_sensors.snap_to_nearest",
    "sciona.atoms.geo.geospatial_sensors.resample_to_gsd",
    "sciona.atoms.geo.geospatial_sensors.rts_smooth",
    "sciona.atoms.geo.geospatial_sensors.apply_asymmetric_bias_correction",
}


def test_coordinate_conversions_round_trip_equator_origin() -> None:
    from sciona.atoms.geo.geospatial_sensors import ecef_to_lla, lla_to_ecef

    lat = np.array([0.0], dtype=np.float64)
    lon = np.array([0.0], dtype=np.float64)
    alt = np.array([0.0], dtype=np.float64)
    x, y, z = lla_to_ecef(lat, lon, alt)

    np.testing.assert_allclose(x, np.array([6378137.0]), atol=1e-6)
    np.testing.assert_allclose(y, np.array([0.0]), atol=1e-6)
    np.testing.assert_allclose(z, np.array([0.0]), atol=1e-6)

    out_lat, out_lon, out_alt = ecef_to_lla(x, y, z)
    np.testing.assert_allclose(out_lat, lat, atol=1e-7)
    np.testing.assert_allclose(out_lon, lon, atol=1e-7)
    np.testing.assert_allclose(out_alt, alt, atol=1e-6)


def test_ecef_to_enu_maps_reference_point_to_zero() -> None:
    from sciona.atoms.geo.geospatial_sensors import ecef_to_enu, lla_to_ecef

    x, y, z = lla_to_ecef(
        np.array([37.0], dtype=np.float64),
        np.array([-122.0], dtype=np.float64),
        np.array([10.0], dtype=np.float64),
    )
    east, north, up = ecef_to_enu(x, y, z, ref_lat=37.0, ref_lon=-122.0, ref_alt=10.0)
    np.testing.assert_allclose(east, np.array([0.0]), atol=1e-8)
    np.testing.assert_allclose(north, np.array([0.0]), atol=1e-8)
    np.testing.assert_allclose(up, np.array([0.0]), atol=1e-8)


def test_gnss_clock_and_quality_filters() -> None:
    from sciona.atoms.geo.geospatial_sensors import correct_clock_bias, filter_by_cn0, filter_multipath

    pseudorange = np.array([20_000_000.0, 21_000_000.0], dtype=np.float64)
    sat_bias = np.array([1e-9, 0.0], dtype=np.float64)
    rx_bias = np.array([2e-9, 1e-9], dtype=np.float64)
    corrected = correct_clock_bias(pseudorange, sat_bias, rx_bias)
    expected = pseudorange - 299792458.0 * (rx_bias - sat_bias)
    np.testing.assert_allclose(corrected, expected)

    measurements = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    np.testing.assert_allclose(
        filter_by_cn0(measurements, np.array([25.0, 30.0, 20.0]), threshold=28.0),
        np.array([np.nan, 2.0, np.nan]),
        equal_nan=True,
    )
    np.testing.assert_allclose(
        filter_multipath(measurements, np.array([0.1, 0.5, 1.2]), threshold=0.5),
        np.array([1.0, 2.0, np.nan]),
        equal_nan=True,
    )


def test_pdr_step_heading_and_position_atoms() -> None:
    from sciona.atoms.geo.geospatial_sensors import (
        detect_steps,
        estimate_step_length_weinberg,
        integrate_heading,
        pdr_position_update,
    )

    accel = np.array([0.0, 1.2, 0.0, 1.3, 0.0, 1.4, 0.0], dtype=np.float64)
    steps = detect_steps(accel, threshold=1.0, prominence=0.5, min_distance=2)
    np.testing.assert_array_equal(steps, np.array([1, 3, 5]))

    lengths = estimate_step_length_weinberg(accel, steps, k_constant=0.5)
    assert lengths.shape == (2,)
    assert np.all(lengths > 0.0)

    headings = integrate_heading(np.array([0.0, 1.0, 1.0], dtype=np.float64), dt=1.0)
    np.testing.assert_allclose(headings, np.array([0.0, 0.5, 1.5]))

    path = pdr_position_update(
        np.array([1.0, 1.0], dtype=np.float64),
        np.array([0.0, np.pi / 2.0], dtype=np.float64),
        (0.0, 0.0),
    )
    np.testing.assert_allclose(path, np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]]), atol=1e-12)


def test_snap_resample_smoother_and_bias_atoms() -> None:
    from sciona.atoms.geo.geospatial_sensors import (
        apply_asymmetric_bias_correction,
        resample_to_gsd,
        rts_smooth,
        snap_to_nearest,
    )

    trajectory = np.array([[0.1, 0.2], [2.2, 1.9]], dtype=np.float64)
    grid = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]], dtype=np.float64)
    np.testing.assert_allclose(snap_to_nearest(trajectory, grid), np.array([[0.0, 0.0], [2.0, 2.0]]))

    image = np.arange(4, dtype=np.float64).reshape(2, 2)
    enlarged = resample_to_gsd(image, current_gsd=2.0, target_gsd=1.0)
    assert enlarged.shape == (4, 4)

    filtered_states = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
    predicted_states = filtered_states.copy()
    covs = np.repeat(np.eye(1, dtype=np.float64)[None, :, :], 3, axis=0)
    transitions = covs.copy()
    smoothed_states, smoothed_covs = rts_smooth(filtered_states, covs, predicted_states, covs, transitions)
    np.testing.assert_allclose(smoothed_states, filtered_states)
    np.testing.assert_allclose(smoothed_covs, covs)

    predictions = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    mask = np.array([True, False, True])
    np.testing.assert_allclose(apply_asymmetric_bias_correction(predictions, 1.1, mask), np.array([1.1, 2.0, 3.3]))


def test_geospatial_sensors_metadata_files_are_consistent() -> None:
    refs = json.loads((FAMILY / "references.json").read_text(encoding="utf-8"))
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

    assert {key.partition("@")[0] for key in refs["atoms"]} == EXPECTED_FQDNS
    registry_ids = set(registry["references"])
    for record in refs["atoms"].values():
        assert record["references"]
        for ref in record["references"]:
            assert ref["ref_id"] in registry_ids
            assert ref["match_metadata"]["notes"]

    assert bundle["provider_repo"] == "sciona-atoms-geo"
    assert {row["atom_key"] for row in bundle["rows"]} == EXPECTED_FQDNS


def test_geospatial_sensors_atoms_are_registered() -> None:
    import_module("sciona.atoms.geo.geospatial_sensors.atoms")
    registered = {name for name in REGISTRY if not name.startswith("witness_")}
    for fqdn in EXPECTED_FQDNS:
        assert fqdn.rsplit(".", 1)[-1] in registered
