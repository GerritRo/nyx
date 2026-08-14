"""Instrument file formats: the 2.0 lattice format and the 1.x fallback."""

import astropy.units as u
import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from nyx import ASSETS_PATH
from nyx.core.geometry import Geometry
from nyx.instrument import EffectiveApertureInstrument, save_instrument
from nyx.instrument.io import FORMAT_VERSION

INSTRUMENT = ASSETS_PATH + "HESS_CT1.h5"


@pytest.fixture(scope="module")
def geo():
    return Geometry(wvls=jnp.linspace(300, 700, 8) * u.nm, nside=4, ngrid=2, fov=3.5 * u.deg)


@pytest.fixture(scope="module")
def instrument(geo):
    return EffectiveApertureInstrument.load(INSTRUMENT, geo)


def _write_legacy(path, instrument):
    """Write *instrument* in the pre-2.0 format, with explicit coordinates."""
    with h5py.File(INSTRUMENT, "r") as src, h5py.File(path, "w") as f:
        src.copy("bandpass", f)
        f.create_dataset("grid", data=np.asarray(instrument.grid, dtype=np.float32))
        f.create_dataset("values", data=np.asarray(instrument.pixel_values))
        f.attrs["nyx_instrument_version"] = "1.0"
        f.attrs["instrument_type"] = "EffectiveApertureInstrument"


def test_bundled_instrument_is_current_format():
    with h5py.File(INSTRUMENT, "r") as f:
        assert str(f.attrs["nyx_instrument_version"]) == FORMAT_VERSION
        assert "lattice" in f
        assert "grid" not in f  # superseded by the lattice


def test_legacy_format_is_refused(geo, instrument, tmp_path):
    """Pre-2.0 files must be migrated, not read."""
    legacy = tmp_path / "legacy.h5"
    _write_legacy(legacy, instrument)
    with pytest.raises(ValueError, match="migrate_instrument"):
        EffectiveApertureInstrument.load(legacy, geo)


def test_migrated_legacy_file_recovers_the_same_lattice(geo, instrument, tmp_path):
    """Migration must recover the lattice the old coordinates encoded."""
    from scripts.migrate_instrument import migrate

    legacy = tmp_path / "legacy.h5"
    migrated = tmp_path / "migrated.h5"
    _write_legacy(legacy, instrument)
    migrate(legacy, migrated)
    loaded = EffectiveApertureInstrument.load(migrated, geo)

    assert np.allclose(loaded.lattice.origin, instrument.lattice.origin, rtol=1e-6)
    assert np.allclose(loaded.lattice.step, instrument.lattice.step, rtol=1e-6)
    assert np.array_equal(np.asarray(loaded.lattice.offset), np.asarray(instrument.lattice.offset))
    assert loaded.lattice.shape == instrument.lattice.shape
    assert np.allclose(loaded.weight, instrument.weight, rtol=1e-5)


def test_save_load_round_trip(geo, instrument, tmp_path):
    path = tmp_path / "round_trip.h5"
    save_instrument(instrument, path)
    with h5py.File(path, "r") as f:
        assert str(f.attrs["nyx_instrument_version"]) == FORMAT_VERSION

    loaded = EffectiveApertureInstrument.load(path, geo)
    assert np.array_equal(np.asarray(loaded.lattice.offset), np.asarray(instrument.lattice.offset))
    assert np.allclose(loaded.lattice.origin, instrument.lattice.origin)
    assert np.allclose(loaded.lattice.step, instrument.lattice.step)
    assert np.allclose(loaded.pixel_values, instrument.pixel_values)
    assert np.allclose(loaded.grid, instrument.grid)
    assert np.allclose(loaded.weight, instrument.weight)


def test_migration_preserves_response_and_bandpass(instrument, tmp_path):
    """The 1.x -> 2.0 migration must touch nothing but the geometry."""
    from scripts.migrate_instrument import migrate

    legacy = tmp_path / "legacy.h5"
    migrated = tmp_path / "migrated.h5"
    _write_legacy(legacy, instrument)
    report = migrate(legacy, migrated)

    assert report["residual_steps"] < 1e-3
    with h5py.File(legacy, "r") as a, h5py.File(migrated, "r") as b:
        for key in ("values", "bandpass/wavelength", "bandpass/transmission"):
            assert np.array_equal(np.array(a[key]), np.array(b[key]))
        assert "grid" not in b
        assert str(b.attrs["nyx_instrument_version"]) == FORMAT_VERSION


def test_migration_refuses_current_format(tmp_path):
    from scripts.migrate_instrument import migrate

    with pytest.raises(ValueError, match="already format"):
        migrate(INSTRUMENT, tmp_path / "out.h5")


def test_load_rejects_foreign_file(tmp_path):
    path = tmp_path / "not_nyx.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("values", data=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="not a nyx instrument file"):
        EffectiveApertureInstrument.load(path, None)
