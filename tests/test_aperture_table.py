"""Reading an iactrace effective-aperture table.

The scan that produces one is hours of ray tracing in another package.  nyx's
job is to read the file it left behind, and to do that with numpy alone -- a
user who was handed a ``.npz`` should not have to install a ray tracer to
build an instrument from it.

The fixtures here write the file directly rather than scanning for it, which
is what keeps this a test of nyx.
"""

import json

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest

from nyx.core.geometry import Geometry
from nyx.instrument import EffectiveApertureInstrument, load_aperture_table
from nyx.instrument._iactrace import _TABLE_FORMAT, _TABLE_MAJOR

N_PIXELS = 4
WINDOW = 5
STEP = 1e-3


def write_table(path, **overrides):
    """An aperture archive in iactrace's format, without iactrace.

    A ring of four pixels around the axis, each a single bright node, which is
    enough to have centres, a bandpass and a lattice without being a physics
    fixture.
    """
    values = np.zeros((N_PIXELS, WINDOW, WINDOW), np.float32)
    values[:, WINDOW // 2, WINDOW // 2] = 1.0
    offset = np.array([[0, 0], [0, 4], [4, 0], [4, 4]], np.int32)
    wavelengths = np.linspace(300.0, 700.0, 9)

    payload = {
        "origin": np.array([-4 * STEP, -4 * STEP]),
        "step": np.array([STEP, STEP]),
        "offset": offset,
        "values": values,
        "on_axis_area": np.asarray(2.0),
        "wavelengths": wavelengths,
        "spectral_area": np.full(wavelengths.size, 2.0),
        "meta": np.asarray(json.dumps({"telescope": "rig", "n_samples": 8})),
        "format": np.asarray(_TABLE_FORMAT),
        "format_version": np.asarray(f"{_TABLE_MAJOR}.0"),
    }
    payload.update({k: np.asarray(v) for k, v in overrides.items()})
    np.savez_compressed(path, **payload)
    return path


@pytest.fixture
def table_file(tmp_path):
    return write_table(tmp_path / "rig.npz")


@pytest.fixture
def geo():
    return Geometry(wvls=jnp.linspace(300, 700, 16) * u.nm, nside=16, ngrid=2, fov=4.0 * u.deg)


class TestReading:
    def test_fields_come_back_as_written(self, table_file):
        table = load_aperture_table(table_file)
        assert table.values.shape == (N_PIXELS, WINDOW, WINDOW)
        assert table.offset.shape == (N_PIXELS, 2)
        assert table.on_axis_area == 2.0
        assert np.allclose(table.step, STEP)

    def test_meta_is_a_plain_dict(self, table_file):
        assert load_aperture_table(table_file).meta == {"telescope": "rig", "n_samples": 8}

    def test_reading_does_not_need_iactrace(self, table_file, monkeypatch):
        """The whole point of the format: numpy is the only reader required."""
        import builtins

        real_import = builtins.__import__

        def no_iactrace(name, *args, **kwargs):
            if name.split(".")[0] == "iactrace":
                raise ImportError("iactrace is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_iactrace)
        assert load_aperture_table(table_file).values.shape[0] == N_PIXELS

    def test_repr_says_what_it_holds(self, table_file):
        text = repr(load_aperture_table(table_file))
        assert f"{N_PIXELS} pixels" in text
        assert f"{WINDOW}x{WINDOW}" in text
        assert "300-700 nm" in text


class TestRejection:
    def test_a_foreign_npz_is_named_as_such(self, tmp_path):
        path = tmp_path / "junk.npz"
        np.savez(path, values=np.zeros(3))
        with pytest.raises(ValueError, match=_TABLE_FORMAT):
            load_aperture_table(path)

    def test_a_future_major_version_is_refused(self, tmp_path):
        path = write_table(tmp_path / "rig.npz", format_version="99.0")
        with pytest.raises(ValueError, match="99.0"):
            load_aperture_table(path)


class TestBuildingAnInstrument:
    def test_a_path_is_accepted_directly(self, geo, table_file):
        inst = EffectiveApertureInstrument.from_iactrace_table(geo, table_file)
        assert inst.pixel_values.shape == (N_PIXELS, WINDOW, WINDOW)

    def test_a_path_and_a_table_agree(self, geo, table_file):
        from_path = EffectiveApertureInstrument.from_iactrace_table(geo, table_file)
        from_object = EffectiveApertureInstrument.from_iactrace_table(
            geo, load_aperture_table(table_file)
        )
        assert np.allclose(np.asarray(from_path.pixel_values), np.asarray(from_object.pixel_values))
        assert np.allclose(np.asarray(from_path.bandpass), np.asarray(from_object.bandpass))

    def test_response_is_normalised_by_the_on_axis_area(self, geo, table_file):
        """``values`` are areas in m^2; an instrument's response is dimensionless."""
        inst = EffectiveApertureInstrument.from_iactrace_table(geo, table_file)
        assert np.isclose(float(np.asarray(inst.pixel_values).max()), 1.0 / 2.0)

    def test_a_dark_table_is_refused(self, geo, tmp_path):
        path = write_table(tmp_path / "dark.npz", on_axis_area=0.0)
        with pytest.raises(ValueError, match="on-axis"):
            EffectiveApertureInstrument.from_iactrace_table(geo, path)

    def test_pixel_centres_land_on_the_lattice(self, geo, table_file):
        """Each fixture pixel is one lit node, so its centroid is that node."""
        inst = EffectiveApertureInstrument.from_iactrace_table(geo, table_file)
        table = load_aperture_table(table_file)
        expected = table.origin + (table.offset + WINDOW // 2) * table.step
        assert np.allclose(np.asarray(inst.centers), expected, atol=1e-9)
