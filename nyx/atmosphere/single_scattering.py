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
from nyx.core.protocols import AtmosphereModel
from nyx.core.records import AtmosphereResult

__all__ = [
    "SingleScattering",
]


class SingleScattering(AtmosphereModel):
    """Single-scattering atmosphere, composed of named components.

    Each :class:`~nyx.atmosphere.components.ScatteringComponent` supplies its
    own optical depth, phase function, single-scattering albedo, airmass and
    trainable parameters.
    """

    components: dict
    _wvls: jax.Array  # (n_wvl,) wavelengths in nm
    _pixel_area: jax.Array  # scalar, healpix pixel area
    _airmass_func: Callable = eqx.field(static=True)
    _geo_signature: tuple = eqx.field(static=True)

    def __init__(self, geo, components, airmass_formula="kasten_young_1989"):
        """Build the atmosphere.

        Parameters
        ----------
        geo : Geometry
        components : dict of str to ScatteringComponent
            Components to compose, each built against ``geo.wvls``.
        airmass_formula : str
            Key into :data:`~nyx.atmosphere.components.AIRMASS_FUNCTIONS`.
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
        self._geo_signature = geo.signature

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
        """Airmass at the given altitudes, in radians."""
        return self._airmass_func(jnp.pi / 2 - altitudes)

    def _tau_components(self, height_km):
        """Total and per-component optical depth, each ``(n_wvl,)``."""
        taus = [c.tau(height_km) for c in self.components.values()]
        tau_total = taus[0]
        for t in taus[1:]:
            tau_total = tau_total + t
        return tau_total, taus

    def _optical_path(self, altitudes, height_km):
        """Slant optical depth ``sum_i tau_i * X_i`` along each line of sight.

        For altitudes of shape ``S`` the result is ``S + (n_wvl,)``.  Each
        component supplies its own airmass.
        """
        zenith = jnp.pi / 2 - jnp.asarray(altitudes)
        path = jnp.zeros(zenith.shape + (len(self._wvls),))
        for c in self.components.values():
            airmass = jnp.asarray(c.airmass(zenith, self._airmass_func))
            path = path + c.tau(height_km) * airmass[..., None]
        return path

    def extinct(self, altitudes, spectra, height_km):
        """Extinct point-source spectra, shape ``(n_sources, n_wvl)``."""
        path = self._optical_path(jnp.asarray(altitudes)[..., 0], height_km)
        return spectra * jnp.exp(-path)

    def _scattering_kernel_from_tau(self, cos_scat_angle, sec_z_fov, sec_z_source, tau_total, taus):
        """Scattering kernel, given the optical depths.

        Parameters
        ----------
        cos_scat_angle : jax.Array, shape (n_lon, n_lat, n_sources)
        sec_z_fov : jax.Array, shape (n_lon, n_lat)
        sec_z_source : jax.Array, shape (n_sources,)
        tau_total : jax.Array, shape (n_wvl,)
        taus : dict of str to jax.Array
            Per-component optical depth, each of shape ``(n_wvl,)``.

        Returns
        -------
        jax.Array, shape (n_lon, n_lat, n_sources, n_wvl)
        """
        indicatrix = jnp.zeros(cos_scat_angle.shape + (len(self._wvls),))
        for comp, tau_i in zip(self.components.values(), taus, strict=True):
            p_i = comp.phase(cos_scat_angle)
            indicatrix = indicatrix + comp.ssa * tau_i * p_i[..., None]
        denom = jnp.where(jnp.abs(tau_total) > 1e-10, tau_total, 1e-10)
        indicatrix = indicatrix / denom

        grad = gradation_function(
            tau_total[None, None, None, :],
            sec_z_fov[:, :, None, None],
            sec_z_source[None, None, :, None],
        )

        return indicatrix * grad

    @staticmethod
    def _cos_scattering_angle(sky):
        """Cosine of the angle from every FOV cell to every sky pixel.

        Returns
        -------
        jax.Array, shape (n_lon, n_lat, nsky)
        """
        return cos_angular_separation_jax(
            sky.fov_altaz_grid[..., 0][..., None],
            sky.fov_altaz_grid[..., 1][..., None],
            sky.altaz_coord[..., 0],
            sky.altaz_coord[..., 1],
        )

    def _scattering_kernel(self, cos_scat_angle, sec_z_fov, sec_z_source, height_km):
        """Scattering kernel, computing the optical depths at ``height_km`` first."""
        tau_total, taus = self._tau_components(height_km)
        return self._scattering_kernel_from_tau(
            cos_scat_angle,
            sec_z_fov,
            sec_z_source,
            tau_total,
            taus,
        )

    def evaluate(self, sky):
        """Extinction and scattering kernel over the whole hemisphere.

        Parameters
        ----------
        sky : SkyGeometry

        Returns
        -------
        AtmosphereResult
        """
        alt_hp = sky.altaz_coord[..., 1]
        sec_z_hp = self._airmass(alt_hp)

        tau_total, taus = self._tau_components(sky.height_km)

        kernel = self._scattering_kernel_from_tau(
            self._cos_scattering_angle(sky),
            self._airmass(sky.fov_altaz_grid[..., 1]),
            sec_z_hp,
            tau_total,
            taus,
        )
        scattering_map = kernel * self._pixel_area

        extinction_hp = jnp.exp(-self._optical_path(alt_hp, sky.height_km))

        return AtmosphereResult(
            extinction_hp=extinction_hp,
            scattering_map=scattering_map,
            npix=sky.hemisphere_mask.shape[0],
        )

    def scatter_sources(self, sky, source_coords, source_spectra, bp):
        """Scatter discrete point sources onto the FOV grid.

        Parameters
        ----------
        sky : SkyGeometry
        source_coords : jax.Array, shape (n_src, 2)
            AltAz in radians.
        source_spectra : jax.Array, shape (n_src, n_wvl)
        bp : jax.Array, shape (n_wvl,)
            Bandpass weights.

        Returns
        -------
        jax.Array, shape (n_lon, n_lat)
        """
        fov_az = sky.fov_altaz_grid[..., 0]  # (n_lon, n_lat)
        fov_alt = sky.fov_altaz_grid[..., 1]  # (n_lon, n_lat)
        sec_z_fov = self._airmass(fov_alt)  # (n_lon, n_lat)

        src_az = source_coords[:, 0]  # (n_src,)
        src_alt = source_coords[:, 1]  # (n_src,)
        sec_z_src = self._airmass(src_alt)  # (n_src,)

        cos_scat_angle = cos_angular_separation_jax(
            fov_az[:, :, None],
            fov_alt[:, :, None],
            src_az[None, None, :],
            src_alt[None, None, :],
        )  # (n_lon, n_lat, n_src)

        kernel = self._scattering_kernel(
            cos_scat_angle,
            sec_z_fov,
            sec_z_src,
            sky.height_km,
        )  # (n_lon, n_lat, n_src, n_wvl)

        return jnp.sum(
            kernel * bp * source_spectra[None, None, :, :],
            axis=(-2, -1),
        )  # (n_lon, n_lat)

    @classmethod
    def from_hg(
        cls,
        geo,
        aod_500=0.1,
        angstrom_exp=1.5,
        hg_asymmetry=0.65,
        hg_ssa=0.9,
        airmass_formula="kasten_young_1989",
        pressure_hpa=None,
    ):
        """Rayleigh scattering plus Angstrom-law aerosol, no molecular absorption.

        Parameters
        ----------
        geo : Geometry
        aod_500 : float
            Aerosol optical depth at 500 nm.
        angstrom_exp : float
            Angstrom exponent.
        hg_asymmetry, hg_ssa : float
            Henyey-Greenstein asymmetry and aerosol single-scattering albedo.
        airmass_formula : str
            Key into :data:`~nyx.atmosphere.components.AIRMASS_FUNCTIONS`.
        pressure_hpa : float or None
            Station pressure for the Rayleigh column; ``None`` uses the
            barometric estimate from the observer height.

        Returns
        -------
        SingleScattering
        """
        components = _rayleigh_mie(geo, aod_500, angstrom_exp, hg_asymmetry, hg_ssa, pressure_hpa)
        return cls(geo, components, airmass_formula=airmass_formula)

    @classmethod
    def from_hg_ozone(
        cls,
        geo,
        aod_500=0.1,
        angstrom_exp=1.5,
        hg_asymmetry=0.65,
        hg_ssa=0.9,
        airmass_formula="kasten_young_1989",
        pressure_hpa=None,
        ozone_height_km=25.0,
    ):
        """:meth:`from_hg` plus an ozone absorption layer.

        Parameters
        ----------
        geo : Geometry
        aod_500, angstrom_exp, hg_asymmetry, hg_ssa : float
        airmass_formula : str
        pressure_hpa : float or None
            Station pressure for the Rayleigh column; ``None`` uses the
            barometric estimate from the observer height.
        ozone_height_km : float or None
            Height of the ozone layer, used for its own airmass; ``None`` puts it
            back on the shared airmass formula.

        Returns
        -------
        SingleScattering
        """
        components = _rayleigh_mie(geo, aod_500, angstrom_exp, hg_asymmetry, hg_ssa, pressure_hpa)
        components["O3"] = tau_ozone(geo.wvls, layer_height_km=ozone_height_km)
        return cls(geo, components, airmass_formula=airmass_formula)


def _rayleigh_mie(geo, aod_500, angstrom_exp, hg_asymmetry, hg_ssa, pressure_hpa):
    """Classic rayleigh and mie scattering pair.

    Returns
    -------
    dict of str to ScatteringComponent
    """
    w = geo.wvls
    return {
        "Rayleigh": RayleighComponent(w, pressure_hpa=pressure_hpa),
        "Mie": HenyeyGreensteinComponent(
            w,
            aod_500=aod_500,
            angstrom_exp=angstrom_exp,
            hg_asymmetry=hg_asymmetry,
            hg_ssa=hg_ssa,
        ),
    }
