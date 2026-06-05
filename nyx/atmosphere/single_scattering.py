from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from nyx.atmosphere.components import (
    AIRMASS_FUNCTIONS,
    HenyeyGreensteinComponent,
    RayleighComponent,
    gradation_function,
    tau_ozone,
)
from nyx.core.coordinates import cos_angular_separation_jax
from nyx.core.protocols import AtmosphereModel, AtmosphereResult


class SingleScattering(AtmosphereModel):
    """Single-scattering atmosphere.

    Composes named :class:`ScatteringComponent` modules, each providing
    its own optical depth, phase function, and trainable parameters.
    The scattering kernel generically blends all components weighted by
    their optical depths.
    """

    _trainable = ()  # trainable params live on individual components

    components: dict
    _wvls: jax.Array  # (n_wvl,) wavelengths in nm
    _pixel_area: jax.Array  # scalar, healpix pixel area
    _airmass_func: Callable = eqx.field(static=True)

    def __init__(self, geo, components, airmass_formula="kasten_young_1989"):
        """
        Parameters
        ----------
        geo : Geometry
            Resolution configuration (provides wavelengths, pixel_area).
        components : dict[str, ScatteringComponent]
            Named scattering and absorption components to compose. Each
            must have been constructed against ``geo.wvls``.
        airmass_formula : str
            Key into AIRMASS_FUNCTIONS.
        """
        for name, c in components.items():
            if not jnp.array_equal(c._rendering_wvls, geo.wvls):
                raise ValueError(
                    f"Component {name!r} was built against a different "
                    f"wavelength grid than geo.wvls"
                )
        self.components = dict(components)
        self._wvls = geo.wvls
        self._pixel_area = jnp.array(float(geo.pixel_area))
        self._airmass_func = AIRMASS_FUNCTIONS[airmass_formula]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.components[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute or component "
                f"named '{name}'. Components: {list(self.components.keys())}"
            ) from None

    def _airmass(self, altitudes):
        """Compute airmass from source altitudes."""
        return self._airmass_func(jnp.pi / 2 - altitudes)

    def _tau_components(self, height_km):
        """Compute per-component and total optical depth.

        Returns
        -------
        tau_total : jax.Array, shape (n_wvl,)
        taus : list of jax.Array, each shape (n_wvl,)
        """
        taus = [c.tau(height_km) for c in self.components.values()]
        tau_total = taus[0]
        for t in taus[1:]:
            tau_total = tau_total + t
        return tau_total, taus

    def extinct(self, altitudes, spectra, height_km):
        """Apply extinction to spectra at given sky positions.

        Parameters
        ----------
        altitudes : jax.Array, shape (n_sources, 1)
            Source altitudes in radians.
        spectra : jax.Array, shape (n_sources, n_wvl)
            Source spectra to extinct.
        height_km : jax.Array
            Observer height above sea level in km (from SkyGeometry).

        Returns
        -------
        jax.Array, shape (n_sources, n_wvl)
            Extincted spectra.
        """
        tau_total, _ = self._tau_components(height_km)
        sec_z = self._airmass(altitudes)
        extinction = jnp.exp(-tau_total[None, :] * sec_z)
        return spectra * extinction

    def _scattering_kernel_from_tau(self, cos_scat_angle, sec_z_fov, sec_z_source, tau_total, taus):
        """Compute scattering kernel from pre-computed optical depths.

        Parameters
        ----------
        cos_scat_angle : jax.Array, shape (ngrid, ngrid, n_sources)
        sec_z_fov : jax.Array, shape (ngrid, ngrid)
        sec_z_source : jax.Array, shape (n_sources,)
        tau_total : jax.Array, shape (n_wvl,)
        taus : list of jax.Array, each shape (n_wvl,)

        Returns
        -------
        jax.Array, shape (ngrid, ngrid, n_sources, n_wvl)
        """
        indicatrix = jnp.zeros(cos_scat_angle.shape + (len(self._wvls),))
        for comp, tau_i in zip(self.components.values(), taus, strict=True):
            p_i = comp.phase(cos_scat_angle)
            indicatrix = indicatrix + tau_i * p_i[..., None]
        indicatrix = indicatrix / jnp.maximum(tau_total, 1e-10)

        grad = gradation_function(
            tau_total[None, None, None, :],
            sec_z_fov[:, :, None, None],
            sec_z_source[None, None, :, None],
        )

        return indicatrix * grad

    def _scattering_kernel(self, cos_scat_angle, sec_z_fov, sec_z_source, height_km):
        """Compute scattering kernel: indicatrix * gradation.

        Parameters
        ----------
        cos_scat_angle : jax.Array, shape (ngrid, ngrid, n_sources)
            Cosine of the angle between FOV grid cells and source
            positions.
        sec_z_fov : jax.Array, shape (ngrid, ngrid)
            Airmass at each FOV grid cell.
        sec_z_source : jax.Array, shape (n_sources,)
            Airmass at each source position.
        height_km : jax.Array
            Observer height above sea level.

        Returns
        -------
        jax.Array, shape (ngrid, ngrid, n_sources, n_wvl)
        """
        tau_total, taus = self._tau_components(height_km)
        return self._scattering_kernel_from_tau(
            cos_scat_angle,
            sec_z_fov,
            sec_z_source,
            tau_total,
            taus,
        )

    def evaluate(self, sky):
        """Compute extinction and scattering kernel.

        Parameters
        ----------
        sky : SkyGeometry
            Hemisphere altaz, FOV grid altaz, scattering angles.

        Returns
        -------
        AtmosphereResult
        """
        alt_hp = sky.altaz_coord[..., 1]
        sec_z_hp = self._airmass(alt_hp)

        tau_total, taus = self._tau_components(sky.height_km)

        kernel = self._scattering_kernel_from_tau(
            jnp.cos(sky.scattering_angle),
            self._airmass(sky.fov_altaz_grid[..., 1]),
            sec_z_hp,
            tau_total,
            taus,
        )
        scattering_map = kernel * self._pixel_area

        extinction_hp = jnp.exp(-tau_total[None, :] * sec_z_hp[:, None])

        return AtmosphereResult(
            extinction_hp=extinction_hp,
            scattering_map=scattering_map,
            npix=sky.hemisphere_mask.shape[0],
        )

    def scatter_sources(self, sky, source_coords, source_spectra, bp):
        """Compute scattered contribution of discrete point sources onto FOV grid.

        Parameters
        ----------
        sky : SkyGeometry
        source_coords : jax.Array, shape (n_src, 2)
            Source positions in AltAz (az, alt) radians.
        source_spectra : jax.Array, shape (n_src, n_wvl)
            Source spectra in flux units.
        bp : jax.Array, shape (n_wvl,)
            Bandpass weights.

        Returns
        -------
        jax.Array, shape (ngrid, ngrid)
            Scattered radiance on the FOV evaluation grid.
        """
        fov_az = sky.fov_altaz_grid[..., 0]  # (ngrid, ngrid)
        fov_alt = sky.fov_altaz_grid[..., 1]  # (ngrid, ngrid)
        sec_z_fov = self._airmass(fov_alt)  # (ngrid, ngrid)

        src_az = source_coords[:, 0]  # (n_src,)
        src_alt = source_coords[:, 1]  # (n_src,)
        sec_z_src = self._airmass(src_alt)  # (n_src,)

        cos_scat_angle = cos_angular_separation_jax(
            fov_az[:, :, None],
            fov_alt[:, :, None],
            src_az[None, None, :],
            src_alt[None, None, :],
        )  # (ngrid, ngrid, n_src)

        kernel = self._scattering_kernel(
            cos_scat_angle,
            sec_z_fov,
            sec_z_src,
            sky.height_km,
        )  # (ngrid, ngrid, n_src, n_wvl)

        # No pixel_area: point sources are delta functions
        return jnp.sum(
            kernel * bp * source_spectra[None, None, :, :],
            axis=(-2, -1),
        )  # (ngrid, ngrid)


def HGNoAbsorption(
    geo, aod_500=0.1, angstrom_exp=1.5, hg_asymmetry=0.75, airmass_formula="kasten_young_1989"
):
    """Standard single-scattering atmosphere (Rayleigh + Mie).

    Standard Rayleigh scattering plus Angstrom-law aerosol (Mie) extinction
    with no molecular absorption. Scattering is HG for aerosols.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration (provides wavelengths, pixel_area).
    aod_500 : float
        Aerosol optical depth at 500 nm (default 0.1).
    angstrom_exp : float
        Angstrom exponent (default 1.5).
    hg_asymmetry : float
        Henyey-Greenstein asymmetry parameter (default 0.75).
    airmass_formula : str
        Key into ``AIRMASS_FUNCTIONS`` (default ``'kasten_young_1989'``).

    Returns
    -------
    SingleScattering
    """
    w = geo.wvls
    components = {
        "Rayleigh": RayleighComponent(w),
        "Mie": HenyeyGreensteinComponent(
            w,
            aod_500=aod_500,
            angstrom_exp=angstrom_exp,
            hg_asymmetry=hg_asymmetry,
        ),
    }
    return SingleScattering(geo, components, airmass_formula=airmass_formula)


def HGOzoneAbsorption(
    geo, aod_500=0.1, angstrom_exp=1.5, hg_asymmetry=0.75, airmass_formula="kasten_young_1989"
):
    """Standard single-scattering atmosphere (Rayleigh + Mie).

    Standard Rayleigh scattering plus Angstrom-law aerosol (Mie)
    extinction with ozone absorption. Scattering is HG for aerosols.

    Parameters
    ----------
    geo : Geometry
        Resolution configuration (provides wavelengths, pixel_area).
    aod_500 : float
        Aerosol optical depth at 500 nm (default 0.1).
    angstrom_exp : float
        Angstrom exponent (default 1.5).
    hg_asymmetry : float
        Henyey-Greenstein asymmetry parameter (default 0.75).
    airmass_formula : str
        Key into ``AIRMASS_FUNCTIONS`` (default ``'kasten_young_1989'``).

    Returns
    -------
    SingleScattering
    """
    w = geo.wvls
    components = {
        "Rayleigh": RayleighComponent(w),
        "Mie": HenyeyGreensteinComponent(
            w,
            aod_500=aod_500,
            angstrom_exp=angstrom_exp,
            hg_asymmetry=hg_asymmetry,
        ),
        "O3": tau_ozone(w),
    }

    return SingleScattering(geo, components, airmass_formula=airmass_formula)
