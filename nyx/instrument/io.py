from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import interp1d

try:
    import astropy.units as u
except ImportError:
    u = None


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
    """Write bandpass, grid, and metadata shared by all instrument types."""
    wvl_tab = np.linspace(wavelength_range[0], wavelength_range[1], wavelength_samples)
    if u is not None:
        bandpass_tab = inst._bandpass_func(wvl_tab * u.nm)
    else:
        bandpass_tab = inst._bandpass_func(wvl_tab)

    bp_grp = f.create_group("bandpass")
    bp_grp.create_dataset("wavelength", data=wvl_tab)
    bp_grp.create_dataset("transmission", data=bandpass_tab)
    f.create_dataset("grid", data=np.asarray(inst.grid))

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
            f.attrs["nyx_instrument_version"] = "1.1"
            f.attrs["instrument_type"] = "EffectiveApertureMisalignmentInstrument"
        else:
            f.create_dataset("values", data=np.asarray(inst.pixel_values))
            f.attrs["nyx_instrument_version"] = "1.0"
            f.attrs["instrument_type"] = "EffectiveApertureInstrument"


def load_instrument(filepath, geo, batch_size=None):
    """Load any instrument from HDF5 file.

    Dispatches on the ``instrument_type`` attribute stored in the file.

    Parameters
    ----------
    filepath : str or Path
        Path to HDF5 file.
    geo : Geometry
        Resolution configuration.
    batch_size : int or None
        Pixel-chunk size used by ``project_catalog``.  See
        :func:`nyx.instrument._interpolation.compute_pixel_weights`
        for semantics.  ``None`` (default) is fully parallel over
        pixels.

    Returns
    -------
    InstrumentModel
    """
    filepath = Path(filepath)
    with h5py.File(filepath, "r") as f:
        if "nyx_instrument_version" not in f.attrs:
            raise ValueError("Not a valid nyx instrument file")
        itype = f.attrs.get("instrument_type", "EffectiveApertureInstrument")
        bandpass_func = _load_bandpass(f)
        grid = f["grid"][:]
        values = f["values"][:]

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
            grid,
            values,
            sigma_x_coords,
            sigma_y_coords,
            batch_size=batch_size,
        )

    from nyx.instrument.effective_aperture import EffectiveApertureInstrument

    return EffectiveApertureInstrument(geo, bandpass_func, grid, values, batch_size=batch_size)
