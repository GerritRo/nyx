"""Instrument models: what the telescope and camera do to the sky.

An instrument carries a bandpass (effective aperture times transmission,
per wavelength) and a per-pixel angular response, and both normally come
out of an :mod:`iactrace` ray trace rather than being written by hand.

Three ways in, in the order you would use them:

- :func:`load_aperture_table` reads a table iactrace saved to ``.npz``,
  which needs numpy and nothing else --
  :meth:`~nyx.instrument.effective_aperture.EffectiveApertureInstrument.from_iactrace_table`
  takes the path directly.
- :meth:`~nyx.instrument.effective_aperture.EffectiveApertureInstrument.from_iactrace`
  runs the scan itself, which needs the ``nyx[iactrace]`` extra.
- :func:`load_instrument` reads an instrument nyx itself saved, in nyx's
  own HDF5 format.
"""

from ._iactrace import ApertureTable, load_aperture_table
from ._interpolation import PixelLattice
from .effective_aperture import EffectiveApertureInstrument, EffectiveApertureMisalignmentInstrument
from .io import load_instrument, save_instrument, tabulated_bandpass

__all__ = [
    "ApertureTable",
    "EffectiveApertureInstrument",
    "EffectiveApertureMisalignmentInstrument",
    "PixelLattice",
    "load_aperture_table",
    "load_instrument",
    "save_instrument",
    "tabulated_bandpass",
]
