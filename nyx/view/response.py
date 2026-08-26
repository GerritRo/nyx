from __future__ import annotations

import dataclasses
from typing import Any

import astropy.units as u
import numpy as np
from astropy.constants import c as _C
from astropy.constants import h as _H

from nyx.core.spectral import bin_widths

__all__ = ["XYZ_TO_SRGB", "SpectralResponse", "cie_xyz", "to_linear_srgb"]

# photon energy [J] = _HC_J_NM / wavelength[nm]
_HC_J_NM = float((_H * _C).to(u.J * u.nm).value)

#: Linear sRGB (IEC 61966-2-1) primaries from CIE XYZ, D65 white point.
XYZ_TO_SRGB = np.array(
    [
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ]
)

#: Rec. 709 luminance weights, the Y row of the inverse of :data:`XYZ_TO_SRGB`.
LUMA = np.array([0.2126, 0.7152, 0.0722])


# Colour matching


def _as_nm(wvls: Any) -> np.ndarray:
    """Any wavelength grid to a plain float array in nm."""
    if isinstance(wvls, u.Quantity):
        return np.asarray(wvls.to(u.nm).value, dtype=float)
    return np.asarray(wvls, dtype=float)


def _lobe(lam: np.ndarray, mu: float, sigma_lo: float, sigma_hi: float) -> np.ndarray:
    """Gaussian with a different width either side of its peak."""
    sigma = np.where(lam < mu, sigma_lo, sigma_hi)
    return np.exp(-0.5 * ((lam - mu) / sigma) ** 2)


def cie_xyz(wvls: Any) -> np.ndarray:
    """CIE 1931 2-degree colour matching functions ``x``, ``y``, ``z``.

    Evaluated from the multi-lobe Gaussian fit of Wyman, Sloan & Shirley
    (2013), *Simple Analytic Approximations to the CIE XYZ Color Matching
    Functions*, JCGT 2(2).  The fit reproduces the Planckian locus to
    better than 0.001 in ``x`` and ``y`` between 3000 K and 10000 K,
    which is far inside the accuracy any sky model brings to the
    question, and it needs no tabulated data.

    Parameters
    ----------
    wvls : array-like or astropy Quantity
        Wavelength grid (nm if unitless).

    Returns
    -------
    np.ndarray, shape (n_wvl, 3)
        Response per unit *energy* radiance, in ``[X, Y, Z]`` order.
    """
    lam = _as_nm(wvls)
    x = (
        1.056 * _lobe(lam, 599.8, 37.9, 31.0)
        + 0.362 * _lobe(lam, 442.0, 16.0, 26.7)
        - 0.065 * _lobe(lam, 501.1, 20.4, 26.2)
    )
    y = 0.821 * _lobe(lam, 568.8, 46.9, 40.5) + 0.286 * _lobe(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _lobe(lam, 437.0, 11.8, 36.0) + 0.681 * _lobe(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


@dataclasses.dataclass(frozen=True)
class SpectralResponse:
    """The channels a view integrates the sky into.

    One object covers both presentations: an instrument's bandpass is a
    single channel, a colour camera is three.  The renderer only ever
    sees the matrix, so it does not know or care which it was given.

    Parameters
    ----------
    channels : np.ndarray, shape (n_wvl, k)
        Per-channel response to *photon* radiance, already multiplied by
        the wavelength bin widths -- the same convention as
        :attr:`~nyx.core.protocols.InstrumentModel.bandpass`, so
        contracting an emitter's ``(..., n_wvl)`` radiance against it
        integrates over the band.  A 1-D array is accepted and read as a
        single channel.
    to_srgb : np.ndarray or None
        ``(3, 3)`` matrix taking the channels to linear sRGB, for a
        three-channel response.  ``None`` when the channels are linear
        sRGB already, or when there is nothing to display in colour.
    names : tuple of str
        Channel labels, for reporting.  Defaults to ``band`` for one
        channel and ``ch0, ch1, ...`` otherwise.

    Examples
    --------
    ::

        SpectralResponse.cie(geo.wvls)              # colour
        SpectralResponse.from_bandpass(instrument)  # one telescope band
    """

    channels: np.ndarray
    to_srgb: np.ndarray | None = None
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = np.atleast_2d(np.asarray(self.channels, dtype=float))
        if values.shape[0] == 1 and np.ndim(self.channels) == 1:
            values = values.T  # a bare curve is one channel, not one wavelength
        if values.ndim != 2:
            raise ValueError(f"channels must be (n_wvl, k), got shape {values.shape}")
        object.__setattr__(self, "channels", values)
        if self.to_srgb is not None and values.shape[1] != 3:
            raise ValueError(f"to_srgb needs three channels to convert from, got {values.shape[1]}")
        if not self.names:
            default = (
                ("band",)
                if values.shape[1] == 1
                else tuple(f"ch{i}" for i in range(values.shape[1]))
            )
            object.__setattr__(self, "names", default)
        elif len(self.names) != values.shape[1]:
            raise ValueError(f"got {len(self.names)} names for {values.shape[1]} channels")

    @property
    def n_wvl(self) -> int:
        """Length of the wavelength grid the response is defined on."""
        return int(self.channels.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of channels, ``k``."""
        return int(self.channels.shape[1])

    @property
    def colour(self) -> bool:
        """Whether this response can be shown as colour."""
        return self.to_srgb is not None or self.n_channels == 3

    @classmethod
    def from_bandpass(cls, bandpass: Any, *, name: str = "band") -> SpectralResponse:
        """One channel, from an instrument or an already-integrated bandpass.

        This is what makes a map view and a photograph the same
        computation: hand the renderer a telescope's own passband and it
        produces exactly what
        :meth:`~nyx.core.scene.Scene.sky_view` used to.

        Parameters
        ----------
        bandpass : InstrumentModel or array-like
            An instrument (its ``bandpass`` attribute is read) or the
            ``(n_wvl,)`` curve itself.  Either way the values are taken
            to include the wavelength bin widths already, as
            :attr:`~nyx.core.protocols.InstrumentModel.bandpass` does --
            see :meth:`from_curves` for raw per-nm transmission.
        name : str
            Label for the single channel.

        Returns
        -------
        SpectralResponse
        """
        # Duck-typed on purpose: nyx.view must not import nyx.instrument.
        values = np.asarray(getattr(bandpass, "bandpass", bandpass), dtype=float)
        if values.ndim != 1:
            raise ValueError(f"a bandpass must be one-dimensional, got shape {values.shape}")
        return cls(values[:, None], None, (name,))

    @classmethod
    def from_curves(
        cls,
        wvls: Any,
        curves: Any,
        *,
        to_srgb: np.ndarray | None = None,
        names: tuple[str, ...] = (),
    ) -> SpectralResponse:
        """From raw per-nm sensitivity curves, e.g. a camera you measured.

        Parameters
        ----------
        wvls : array-like or astropy Quantity
            Wavelength grid the curves are sampled on (nm if unitless).
        curves : array-like, shape (n_wvl, k) or (n_wvl,)
            Sensitivity per photon.  The quadrature weight
            (:func:`~nyx.core.spectral.bin_widths`) is applied here, so
            pass the curves as measured.
        to_srgb : np.ndarray, optional
            ``(3, 3)`` colour matrix, for three curves that are not
            already sRGB-like.
        names : tuple of str, optional
            Channel labels.
        """
        lam = _as_nm(wvls)
        values = np.atleast_2d(np.asarray(curves, dtype=float))
        if values.shape[0] != lam.size:
            values = values.T
        if values.shape[0] != lam.size:
            raise ValueError(
                f"curves of shape {np.shape(curves)} do not match {lam.size} wavelengths"
            )
        return cls(values * np.asarray(bin_widths(lam))[:, None], to_srgb, names)

    @classmethod
    def cie(cls, wvls: Any, *, qe: Any = None) -> SpectralResponse:
        """Three channels, colorimetric: the CIE 1931 matching functions.

        nyx works in photon radiance, ``photon / s / m^2 / nm / sr``,
        while the colour matching functions are defined against energy.
        The two differ by the photon energy ``hc / lambda``, which is
        folded in here -- so contracting a nyx radiance against
        :attr:`channels` gives CIE tristimulus values directly.  Without
        that factor red light comes out systematically too bright, since
        a red photon carries less energy than a blue one.

        Parameters
        ----------
        wvls : array-like or astropy Quantity
            Wavelength grid (nm if unitless), e.g. ``geo.wvls``.
        qe : array-like, optional
            Extra per-wavelength efficiency (lens transmission, sensor
            QE) multiplying every channel.  Affects brightness and
            colour balance, not the colorimetry of a flat response.

        Returns
        -------
        SpectralResponse
            Channels ``X``, ``Y``, ``Z`` plus the sRGB matrix.
            Normalised so the ``Y`` channel peaks at one, which keeps the
            numbers in a readable range; absolute scale is set by the
            exposure anyway.
        """
        lam = _as_nm(wvls)
        # photon -> energy, then energy -> tristimulus, then the quadrature weight
        weight = (_HC_J_NM / lam) * np.asarray(bin_widths(lam))
        if qe is not None:
            weight = weight * np.asarray(qe, dtype=float)
        channels = cie_xyz(lam) * weight[:, None]
        peak = float(np.max(channels[:, 1]))
        if peak <= 0:
            raise ValueError("the wavelength grid carries no visible response")
        return cls(channels / peak, XYZ_TO_SRGB, ("X", "Y", "Z"))


def to_linear_srgb(values: np.ndarray, response: SpectralResponse) -> np.ndarray:
    """Channel values ``(..., 3)`` to non-negative linear sRGB.

    Applied by :meth:`Camera.expose` on its way out; exposed separately
    for turning a spectrum, or one pixel of a map, into a colour.

    Colours outside the sRGB gamut -- the 557.7 nm airglow line is well
    outside it -- come back from the matrix with a negative component.
    They are desaturated towards white by the smallest amount that clears
    the gamut, which keeps the hue direction and the luminance rather
    than clipping a channel to zero and shifting both.
    """
    if response.to_srgb is None:
        rgb = np.asarray(values, dtype=float)
    else:
        rgb = np.einsum("ij,...j->...i", response.to_srgb, values)
    deficit = np.clip(-rgb.min(axis=-1, keepdims=True), 0.0, None)
    return np.clip(rgb + deficit, 0.0, None)


# sRGB transfer function


def srgb_encode(values: np.ndarray) -> np.ndarray:
    """Linear light to sRGB display values (IEC 61966-2-1)."""
    return np.where(values <= 0.0031308, 12.92 * values, 1.055 * values ** (1 / 2.4) - 0.055)


def srgb_decode(value: float) -> float:
    """An sRGB display value back to linear light."""
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
