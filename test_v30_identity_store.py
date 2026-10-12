import pytest

from digital_twin.identity_store import IdentityStateStore


def test_traits_are_immutable_and_registration_is_idempotent(tmp_path):
    with IdentityStateStore(tmp_path / "identity.sqlite") as store:
        first = store.register("rider", "r:1", traits={"income_band": "middle"})
        second = store.register("rider", "r:1", traits={"income_band": "middle"})
        assert first == second
        with pytest.raises(ValueError, match="immutable"):
            store.register("rider", "r:1", traits={"income_band": "high"})


def test_direct_pii_is_rejected(tmp_path):
    with IdentityStateStore(tmp_path / "identity.sqlite") as store:
        with pytest.raises(ValueError, match="PII"):
            store.register("rider", "r:1", traits={"email": "person@example.com"})


def test_optimistic_concurrency_rejects_stale_write(tmp_path):
    with IdentityStateStore(tmp_path / "identity.sqlite") as store:
        original = store.register("rider", "r:1", traits={})
        updated = store.update_state("rider", "r:1", {"x": 1}, expected_version=original.version)
        assert updated.version == 1
        with pytest.raises(RuntimeError, match="stale"):
            store.update_state("rider", "r:1", {"x": 2}, expected_version=original.version)


def test_rider_memory_is_persistent_and_identity_specific(tmp_path):
    path = tmp_path / "identity.sqlite"
    with IdentityStateStore(path) as store:
        store.register("rider", "r:1", traits={"threshold": 2.0})
        store.register("rider", "r:2", traits={"threshold": 2.0})
        state = store.update_rider_experience(
            "r:1", platform_experience=1.0, outside_experience=-0.5, alpha=0.2
        )
        assert state.state["platform_experience_ewma"] == pytest.approx(0.2)
        assert store.get("rider", "r:2").state == {}
    with IdentityStateStore(path) as store:
        assert store.get("rider", "r:1").state["outside_experience_ewma"] == pytest.approx(-0.1)


def test_driver_memory_tracks_supply_response(tmp_path):
    with IdentityStateStore(tmp_path / "identity.sqlite") as store:
        store.register("driver", "d:1", traits={"home_region": "north"})
        state = store.update_driver_experience(
            "d:1", availability=1.0, acceptance=0.5, earnings_per_hour=30.0, alpha=0.25
        )
        assert state.state["availability_ewma"] == pytest.approx(0.25)
        assert state.state["acceptance_ewma"] == pytest.approx(0.125)
        assert state.state["earnings_per_hour_ewma"] == pytest.approx(7.5)


def test_unknown_identity_and_invalid_alpha_fail(tmp_path):
    with IdentityStateStore(tmp_path / "identity.sqlite") as store:
        with pytest.raises(KeyError):
            store.update_rider_experience(
                "missing", platform_experience=0, outside_experience=0, alpha=0.1
            )
        store.register("rider", "r:1", traits={})
        with pytest.raises(ValueError, match="alpha"):
            store.update_rider_experience(
                "r:1", platform_experience=0, outside_experience=0, alpha=1.1
            )


def test_counts_and_schema_survive_restart(tmp_path):
    path = tmp_path / "identity.sqlite"
    with IdentityStateStore(path) as store:
        store.register("rider", "r:1", traits={})
        store.register("driver", "d:1", traits={})
    with IdentityStateStore(path) as store:
        assert store.counts() == {"driver": 1, "rider": 1}
