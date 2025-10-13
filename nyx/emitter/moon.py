from nyx import ASSETS_PATH
from nyx.utils.spectra import SolarSpectrumRieke2008

import jax.numpy as jnp
import numpy as np
import healpy as hp
import astropy
import astropy.units as u
from astropy.constants import c, h
from scipy.interpolate import UnivariateSpline

from nyx.core import SpectralHandler
from nyx.core.scene import ComponentType
from nyx.core import get_wavelengths, get_healpix_nside
from nyx.core import CatalogQuery, ParameterSpec
from nyx.core.model import EmitterProtocol
from nyx.atmosphere import get_airmass_formula

from nyx.units import nixify

class Jones2013(EmitterProtocol):
    def __init__(self):
        """
        Parameters
        ----------
        """
        self.rolo = np.genfromtxt(ASSETS_PATH + "noll2013_lunar_rolo.dat", delimiter=",")

    def lnA(self, p, g, s_sel):
        p_1, p_2, p_3, p_4 = 4.06054, 12.8802, np.deg2rad(-30.5858), np.deg2rad(16.7498)

        sum_a = p[0] + p[1] * g + p[2] * g**2 + p[3] * g**3
        sum_b = p[4] * s_sel + p[5] * s_sel**3 + p[6] * s_sel**5
        sum_c = (
            p[7] * np.exp(-g / p_1)
            + p[8] * np.exp(-g / p_2)
            + p[9] * np.cos((g - p_3) / p_4)
        )

        return sum_a + sum_b + sum_c

    def calc_norm(self, lam, moon_dist, g, s_sel):
        omega_moon = 6.4177 * 1e-5

        res = []
        for j in range(22):
            res.append(np.exp(self.lnA(self.rolo[j][1:], g, s_sel)))

        s = UnivariateSpline(self.rolo[:22, 0], np.asarray(res), k=2)

        return omega_moon**2 / np.pi * s(lam) * (384400 / moon_dist) ** 2

    def get_generator(self, observation):
        # Load relevant global parameters:
        wavelengths = get_wavelengths()
        nside = get_healpix_nside()

        # Pre-compute hemisphere grid:
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        mask = theta < np.pi/2  # Upper hemisphere

        # Get Sun-Moon angle:
        sun = astropy.coordinates.get_sun(observation.AltAz.obstime)
        moon = astropy.coordinates.get_body("moon", observation.AltAz.obstime)
        sun_angle = moon.separation(sun)
        alpha = astropy.coordinates.Angle("180°") - sun_angle

        # Get Moon coordinate:
        coord = moon.transform_to(observation.AltAz)

        # Get norm:
        norm = self.calc_norm(wavelengths, moon.distance.to(u.km).value, alpha.rad, sun_angle.rad)
        
        # Get solar spectrum and normalize to moon:
        wvl, spectrum = SolarSpectrumRieke2008()
        flx = nixify(spectrum, 'flux', wavelength=wvl)
        spec_samp = norm*SpectralHandler.resample(wvl.to(u.nm).value, flx, wavelengths)
        # Calculate airmass:
        airmass = get_airmass_formula()
        sec_Z = jnp.array(airmass(jnp.array(np.pi/2 - coord.alt.rad))).reshape(1,1)

        # Determine image coordinate:
        i_coords = coord.transform_to(observation.frame)
        i_coords = jnp.array([i_coords.lon.rad, i_coords.lat.rad]).reshape((1,2))

        # Create flux map:
        flux_map = np.zeros((npix,len(wavelengths)))
        hp_ind = hp.ang2pix(nside, coord.spherical.lon.deg, coord.spherical.lat.deg, lonlat=True)
        flux_map[hp_ind] = spec_samp / hp.nside2pixarea(nside)**2
        
        def generator(params):
            return CatalogQuery(sec_Z=sec_Z, image_coords=i_coords, flux_values=spec_samp, flux_map=flux_map[mask])

        param_specs = {}

        return generator, param_specs, ComponentType.CATALOG