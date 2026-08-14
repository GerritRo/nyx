from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d

from nyx.instrument._interpolation import PixelLattice

try:
    import astropy.units as u
except ImportError:
    u = None


#: The only instrument file format nyx reads or writes.  It stores the
#: focal-plane geometry as the lattice the pixel response grids are windows
#: onto -- an origin, a step, and one integer node offset per pixel -- rather
#: than as an explicit ``(n_pixels, 2, grid_dim)`` table of coordinates.
#: Convert files written before 2.0 with ``scripts/migrate_instrument.py``.
FORMAT_VERSION = "2.0"


# Shared helpers


def _load_bandpass(f):
    """Read bandpass from an open HDF5 file handle, return callable."""
    wvl_tab = f["bandpass/wavelength"][:]
    transmission_tab = f["bandpass/transmission"][:]
    interp = interp1d(wvl_tab, transmission_tab, kind="linear", bounds_error=False, fill_value=0.0)

    def bandpass_func(wavelength):
        wvl_val = wavelength.value if hasattr(wavelength, "value") else wavelength
        return interp(wvl_val)

    return bandpass_func


def _save_common(f, inst, wavelength_range, wavelength_samples, metadata):
    """Write bandpass, lattice, and metadata shared by all instrument types."""
    wvl_tab = np.linspace(wavelength_range[0], wavelength_range[1], wavelength_samples)
    if u is not None:
        bandpass_tab = inst._bandpass_func(wvl_tab * u.nm)
    else:
        bandpass_tab = inst._bandpass_func(wvl_tab)

    bp_grp = f.create_group("bandpass")
    bp_grp.create_dataset("wavelength", data=wvl_tab)
    bp_grp.create_dataset("transmission", data=bandpass_tab)

    lat_grp = f.create_group("lattice")
    lat_grp.create_dataset("origin", data=np.asarray(inst.lattice.origin, dtype=np.float64))
    lat_grp.create_dataset("step", data=np.asarray(inst.lattice.step, dtype=np.float64))
    lat_grp.create_dataset("offset", data=np.asarray(inst.lattice.offset, dtype=np.int32))

    if metadata is not None:
        meta_grp = f.create_group("metadata")
        for key, value in metadata.items():
            if isinstance(value, str):
                meta_grp.attrs[key] = value
            else:
                meta_grp.create_dataset(key, data=value)


# Public API


def save_instrument(
    inst, filepath, wavelength_range=(200, 1000), wavelength_samples=1000, metadata=None
):
    """Save any instrument to HDF5 file.

    Dispatches on the instrument type to write the correct format.

    Parameters
    ----------
    inst : InstrumentModel
        Instrument to save.
    filepath : str or Path
        Output HDF5 path.
    wavelength_range : tuple
        (min, max) wavelength in nm for bandpass tabulation.
    wavelength_samples : int
        Number of wavelength samples to tabulate.
    metadata : dict or None
        Optional metadata to store.
    """
    from nyx.instrument.effective_aperture import (
        EffectiveApertureMisalignmentInstrument,
    )

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(filepath, "w") as f:
        _save_common(f, inst, wavelength_range, wavelength_samples, metadata)

        if isinstance(inst, EffectiveApertureMisalignmentInstrument):
            f.create_dataset("values", data=np.asarray(inst.all_pixel_values))
            f.create_dataset("sigma_x_coords", data=np.asarray(inst.sigma_x_coords))
            f.create_dataset("sigma_y_coords", data=np.asarray(inst.sigma_y_coords))
            f.attrs["instrument_type"] = "EffectiveApertureMisalignmentInstrument"
        else:
            f.create_dataset("values", data=np.asarray(inst.pixel_values))
            f.attrs["instrument_type"] = "EffectiveApertureInstrument"
        f.attrs["nyx_instrument_version"] = FORMAT_VERSION


def load_instrument(filepath, geo):
    """Load any instrument from HDF5 file.

    Dispatches on the ``instrument_type`` attribute stored in the file.

    Parameters
    ----------
    filepath : str or Path
        Path to an HDF5 file in format :data:`FORMAT_VERSION`.  Convert
        older files with ``scripts/migrate_instrument.py``.
    geo : Geometry
        Resolution configuration.

    Returns
    -------
    InstrumentModel
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        version = str(f.attrs.get("nyx_instrument_version", ""))
        if not version:
            raise ValueError(f"{filepath} is not a nyx instrument file")
        if not version.startswith("2."):
            raise ValueError(
                f"{filepath} is format {version}; convert it to {FORMAT_VERSION} with "
                f"scripts/migrate_instrument.py"
            )
        itype = f.attrs.get("instrument_type", "EffectiveApertureInstrument")
        bandpass_func = _load_bandpass(f)
        values = f["values"][:]
        geometry = PixelLattice(
            origin=f["lattice/origin"][:],
            step=f["lattice/step"][:],
            offset=f["lattice/offset"][:],
            grid_shape=values.shape[-2:],
        )

        if itype == "EffectiveApertureMisalignmentInstrument":
            sigma_x_coords = f["sigma_x_coords"][:]
            sigma_y_coords = f["sigma_y_coords"][:]

    if itype == "EffectiveApertureMisalignmentInstrument":
        from nyx.instrument.effective_aperture import (
            EffectiveApertureMisalignmentInstrument,
        )

        return EffectiveApertureMisalignmentInstrument(
            geo,
            bandpass_func,
            geometry,
            values,
            sigma_x_coords,
            sigma_y_coords,
        )

    from nyx.instrument.effective_aperture import EffectiveApertureInstrument

    return EffectiveApertureInstrument(geo, bandpass_func, geometry, values)
