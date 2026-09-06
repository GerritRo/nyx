from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import healpy as hp
import numpy as np

from nyx.core.units import to_angle_rad
from nyx.view.response import to_linear_srgb
from nyx.view.skyrender import PointField

if TYPE_CHECKING:
    from nyx.view.skyrender import SkyRender

__all__ = ["Camera"]


def _as_rgb(values: np.ndarray) -> np.ndarray:
    """Channel values ``(..., k)`` widened to the three the frame carries.

    A single-band render photographs as a grey image; everything
    downstream (:func:`~nyx.view.response.to_linear_srgb`, the tone
    curve) works in three channels.
    """
    values = np.asarray(values)
    return np.repeat(values, 3, axis=-1) if values.shape[-1] == 1 else values


@dataclasses.dataclass(frozen=True)
class Camera:
    """A virtual camera looking at the sky from the observer's position.

    The frame is the sky as a photograph would frame it: azimuth and
    altitude give the direction the lens points, ``fov`` the horizontal
    field of view.  A frame with the horizon in the lower third -- the
    usual landscape shot -- comes from an altitude of roughly a third of
    the vertical field.

    Parameters
    ----------
    az, alt : float or astropy Quantity
        Direction the camera points (radians if unitless).  Azimuth
        follows the AltAz convention: north is 0, east is 90 degrees.
    fov : float or astropy Quantity
        Horizontal field of view (radians if unitless).  A 24 mm lens on
        full frame is about 74 degrees, a 50 mm lens about 40.
    size : tuple of int
        ``(width, height)`` in pixels.
    roll : float or astropy Quantity
        Rotation of the camera about its optical axis.
    projection : {'rectilinear', 'fisheye'}
        ``'rectilinear'`` is an ordinary lens: straight lines stay
        straight, and the corners of a wide frame are stretched.
        ``'fisheye'`` is equidistant (``r = f * theta``), which is what
        an all-sky shot needs; there ``fov`` may reach 360 degrees.
    psf : float
        Point spread function width (FWHM) in pixels.  Sets how big a
        faint star is; 1.5 to 3 looks like a real lens.
    psf_beta : float
        Moffat index of the point spread function.  Low values give
        broad wings, which is what makes a bright star bloom into a disc
        rather than a hard dot; 2.5 is a good default.
    psf_floor : float
        Where a source's wings stop being drawn, as a fraction of the
        sky background.  The extent is worked out per source, so a star
        a million times the sky spreads far beyond one that is merely
        visible -- which is what makes an overexposed Moon look
        overexposed instead of pasted on.
    psf_max : float
        Hard cap on that extent, in pixels.

    Examples
    --------
    ::

        cam = Camera(az=200 * u.deg, alt=25 * u.deg, fov=70 * u.deg, size=(1600, 1000))
        image = cam.expose(sky)                      # linear sRGB
        rgba = tonemap(image, alpha=cam.horizon())   # ready to composite
    """

    az: Any
    alt: Any
    fov: Any
    size: tuple[int, int]
    roll: Any = 0.0
    projection: str = "rectilinear"
    psf: float = 2.0
    psf_beta: float = 2.5
    psf_floor: float = 0.02
    psf_max: float = 400.0

    def __post_init__(self) -> None:
        if self.projection not in ("rectilinear", "fisheye"):
            raise ValueError(
                f"projection must be 'rectilinear' or 'fisheye', not {self.projection!r}"
            )
        fov = float(to_angle_rad(self.fov))
        if not 0 < fov < (2 * np.pi if self.projection == "fisheye" else np.pi):
            raise ValueError(f"fov out of range for a {self.projection} camera: {fov} rad")
        if self.psf <= 0:
            raise ValueError("psf must be positive")

    # Geometry

    @property
    def width(self) -> int:
        """Frame width in pixels."""
        return int(self.size[0])

    @property
    def height(self) -> int:
        """Frame height in pixels."""
        return int(self.size[1])

    @property
    def focal_px(self) -> float:
        """Focal length in pixels, from the field of view."""
        half = float(to_angle_rad(self.fov)) / 2
        if self.projection == "fisheye":
            return (self.width / 2) / half
        return (self.width / 2) / np.tan(half)

    def _basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Optical axis, right and up vectors in a (north, east, up) frame."""
        az, alt = float(to_angle_rad(self.az)), float(to_angle_rad(self.alt))
        forward = np.array([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)])
        # Cross with the zenith so 'right' runs east when the camera looks north.
        right = np.cross([0.0, 0.0, 1.0], forward)
        norm = np.linalg.norm(right)
        # Straight up or straight down: any azimuthal reference will do.
        right = np.array([np.sin(az), -np.cos(az), 0.0]) if norm < 1e-9 else right / norm
        up = np.cross(forward, right)
        roll = float(to_angle_rad(self.roll))
        if roll:
            cr, sr = np.cos(roll), np.sin(roll)
            right, up = cr * right - sr * up, sr * right + cr * up
        return forward, right, up

    def directions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sky direction of every pixel.

        Returns
        -------
        az, alt : np.ndarray, shape (height, width)
            Azimuth and altitude in radians.
        inside : np.ndarray of bool
            False where the pixel does not image the sky at all -- the
            corners outside a fisheye's circle.  Pixels below the horizon
            are *not* excluded here; see :meth:`horizon`.
        """
        forward, right, up = self._basis()
        col, row = np.meshgrid(np.arange(self.width), np.arange(self.height))
        x = (col - (self.width - 1) / 2) / self.focal_px
        y = ((self.height - 1) / 2 - row) / self.focal_px

        if self.projection == "rectilinear":
            vec = forward[:, None, None] + x * right[:, None, None] + y * up[:, None, None]
            inside = np.ones(x.shape, dtype=bool)
        else:
            theta = np.hypot(x, y)
            inside = theta <= np.pi
            safe = np.where(theta > 0, theta, 1.0)
            radial = np.sin(np.where(inside, theta, 0.0)) / safe
            vec = np.cos(np.where(inside, theta, 0.0)) * forward[:, None, None] + radial * (
                x * right[:, None, None] + y * up[:, None, None]
            )
        vec = vec / np.linalg.norm(vec, axis=0)
        alt = np.arcsin(np.clip(vec[2], -1.0, 1.0))
        az = np.arctan2(vec[1], vec[0]) % (2 * np.pi)
        return az, alt, inside

    def horizon(self) -> np.ndarray:
        """Boolean mask of the pixels that see sky above the horizon.

        This is the alpha channel of a plate meant to sit behind a
        foreground: everything below the horizon (and outside a
        fisheye's image circle) is transparent.
        """
        _, alt, inside = self.directions()
        return inside & (alt > 0)

    def solid_angle(self) -> np.ndarray:
        """Solid angle subtended by each pixel, in steradian.

        Not a constant: a rectilinear frame stretches towards its
        corners, so the same radiance lands on more pixels there.
        """
        col, row = np.meshgrid(np.arange(self.width), np.arange(self.height))
        x = (col - (self.width - 1) / 2) / self.focal_px
        y = ((self.height - 1) / 2 - row) / self.focal_px
        if self.projection == "rectilinear":
            return (1.0 + x**2 + y**2) ** -1.5 / self.focal_px**2
        theta = np.hypot(x, y)
        radial = np.where(theta > 1e-9, np.sin(theta) / np.where(theta > 0, theta, 1.0), 1.0)
        return np.asarray(radial / self.focal_px**2)

    def project(self, az: np.ndarray, alt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sky directions to pixel coordinates.

        Parameters
        ----------
        az, alt : np.ndarray, shape (n,)
            Directions in radians.

        Returns
        -------
        col, row : np.ndarray, shape (n,)
            Fractional pixel coordinates.
        visible : np.ndarray of bool
            Whether the direction falls inside the frame.
        """
        forward, right, up = self._basis()
        vec = np.stack([np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)])
        fwd, xr, yu = forward @ vec, right @ vec, up @ vec

        if self.projection == "rectilinear":
            safe = np.where(fwd > 1e-9, fwd, 1.0)
            x, y = xr / safe, yu / safe
            ahead = fwd > 1e-9
        else:
            theta = np.arccos(np.clip(fwd, -1.0, 1.0))
            radial = np.hypot(xr, yu)
            scale = np.where(radial > 1e-12, theta / np.where(radial > 0, radial, 1.0), 0.0)
            x, y = xr * scale, yu * scale
            # Every direction maps somewhere on an equidistant plane, so
            # the frame bounds below are the only test that applies.  The
            # corners of a frame reach past half the horizontal field.
            ahead = np.ones(theta.shape, dtype=bool)

        col = x * self.focal_px + (self.width - 1) / 2
        row = (self.height - 1) / 2 - y * self.focal_px
        # A bright source just outside the frame still throws its wings
        # into it, so the margin is the widest halo the camera can draw.
        margin = self.psf_max
        visible = (
            ahead
            & (col > -margin)
            & (col < self.width + margin)
            & (row > -margin)
            & (row < self.height + margin)
        )
        return col, row, visible

    # Exposure

    def expose(
        self,
        sky: SkyRender,
        sources: list[str] | str | None = None,
        *,
        direct: bool = True,
        inscatter: bool = True,
        points: bool = True,
    ) -> np.ndarray:
        """Photograph *sky*, or any subset of its components.

        The diffuse maps are sampled bilinearly and weighted by each
        pixel's solid angle; point sources are placed at their exact
        position and spread by the camera's point spread function.  The
        result is linear -- no exposure, no tone curve -- so component
        images add exactly as the physics does.

        Parameters
        ----------
        sky : SkyRender
            Rendered sky, from :func:`~nyx.view.allsky.render_sky`.
        sources : list of str or str, optional
            Emitters to include (default: all).  This is the knob for
            building a picture up one component at a time.
        direct : bool
            Include the direct (extincted) light.  A point source *is*
            direct light -- the Moon shines down one line of sight and
            its scattered light lives in the in-scattered maps -- so
            turning this off drops the point sources with it, and
            ``direct=False`` leaves purely what the atmosphere
            redirected.
        inscatter : bool
            Include the light the atmosphere scatters into the line of
            sight.  Turning this off is the honest way to show what
            in-scattering does to a photograph.
        points : bool
            Include emitters rendered as individual point sources.  Set
            it False to keep the direct diffuse light but drop the stars
            -- useful for measuring the sky background itself.

        Returns
        -------
        np.ndarray, shape (height, width, 3)
            Linear sRGB, non-negative, in
            ``response units / s / m^2`` per pixel.  Multiply by the
            aperture area and the shutter time for a photon count.
        """
        if sky.response.n_channels not in (1, 3):
            raise ValueError(
                f"a photograph needs a one- or three-channel response, got "
                f"{sky.response.n_channels}; render the sky with "
                f"SpectralResponse.cie(geo.wvls) or a single bandpass"
            )
        az, alt, inside = self.directions()
        image = np.zeros((self.height, self.width, 3))

        if direct or inscatter:
            diffuse = _as_rgb(sky.diffuse(sources, direct=direct, inscatter=inscatter))
            # Clamp into the populated hemisphere so the bilinear stencil
            # never mixes in the empty half-sphere and darkens the horizon.
            theta = np.minimum(np.pi / 2 - alt, sky.horizon_theta).ravel()
            phi = az.ravel()
            sampled = np.stack(
                [hp.get_interp_val(diffuse[:, c], theta, phi) for c in range(3)], axis=-1
            )
            image += sampled.reshape(self.height, self.width, 3) * self.solid_angle()[..., None]

        if points and direct:
            background = image.max(axis=-1)
            level = background[background > 0]
            field = sky.point_field(sources).above_horizon()
            self._splat(
                image,
                PointField(field.az, field.alt, _as_rgb(field.flux)),
                float(np.median(level)) if level.size else 0.0,
            )

        image *= (inside & (alt > 0))[..., None]
        return to_linear_srgb(image, sky.response)

    def _splat(self, image: np.ndarray, field: PointField, background: float) -> None:
        """Add point sources to *image* through the Moffat point spread function.

        Each source gets a stamp only as wide as its own wings need: out
        to where the profile drops to :attr:`psf_floor` of the sky
        background, since past that it is not in the picture anyway.  A
        fixed stamp cannot do both jobs -- narrow enough for fifteen
        thousand stars, wide enough for the Moon -- and a Moon whose
        wings are cut off while still far above the sky comes out as a
        bright rectangle.

        The stamp is normalised to sum to one, so a source deposits its
        whole flux however far the profile is carried.
        """
        col, row, visible = self.project(field.az, field.alt)
        col, row, flux = col[visible], row[visible], field.flux[visible]
        if col.size == 0:
            return

        alpha = self.psf / (2 * np.sqrt(2 ** (1 / self.psf_beta) - 1))
        peak = (self.psf_beta - 1) / (np.pi * alpha**2)
        brightness = flux.max(axis=-1)
        floor = self.psf_floor * background
        if floor <= 0:  # nothing but point sources in the frame
            floor = self.psf_floor * float(np.min(brightness[brightness > 0], initial=1.0)) * peak

        with np.errstate(over="ignore"):
            ratio = np.clip(brightness * peak / floor, 1.0, None) ** (1 / self.psf_beta)
        radius = np.clip(alpha * np.sqrt(ratio - 1.0), self.psf, self.psf_max)
        # Grouped into octaves so that near-equal sources share one stamp.
        octave = np.exp2(np.ceil(np.log2(np.maximum(radius, 1.0)))).astype(int)

        base_col, base_row = np.round(col).astype(int), np.round(row).astype(int)
        for size in np.unique(octave):
            sel = np.flatnonzero(octave == size)
            offsets = np.arange(-size, size + 1)
            dy, dx = np.meshgrid(offsets, offsets, indexing="ij")
            # Chunked over sources: the stamp array is (n, (2r+1)^2, 3).
            stride = max(1, 2_000_000 // dx.size)
            for lo in range(0, sel.size, stride):
                take = sel[lo : lo + stride]
                cc, rr = base_col[take, None, None], base_row[take, None, None]
                rsq = (cc + dx - col[take, None, None]) ** 2 + (
                    rr + dy - row[take, None, None]
                ) ** 2
                profile = (1 + rsq / alpha**2) ** -self.psf_beta
                profile /= profile.sum(axis=(1, 2), keepdims=True)

                cols, rows = cc + dx, rr + dy
                keep = (cols >= 0) & (cols < self.width) & (rows >= 0) & (rows < self.height)
                values = profile[..., None] * flux[take, None, None, :]
                np.add.at(image, (rows[keep], cols[keep]), values[keep])
