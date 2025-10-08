from nsb2 import ASSETS_PATH
from nsb2.core.utils import SolarSpectrumRieke2008

import jax.numpy as jnp
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.constants import c, h

from scipy.interpolate import RegularGridInterpolator

from nsb3.core import SunRelativeEclipticFrame
from nsb3.core import get_wavelengths, get_healpix_nside
from nsb3.core import DiffuseQuery, ParameterSpec
from nsb3.core.model import EmitterProtocol

class Leinert1998(EmitterProtocol):
    def __init__(self):
        """
        Parameters
        ----------
        """
        zod = np.genfromtxt(ASSETS_PATH+"leinert1998_zodiacal_light.dat", delimiter=",")
        self.A = RegularGridInterpolator(points=[np.deg2rad(zod[1:, 0]), np.deg2rad(zod[0, 1:])], values=zod[1:, 1:])

    def color_correction(self, lam, elon):
        elon_f = -0.3 * (np.clip(elon, 30, 90) - 30) / 60
        return 1 + (1.2 + elon_f[:, np.newaxis]) * np.log(lam / 500)

    def get_generator(self, observation):
        # Load relevant global parameters:
        wavelengths = get_wavelengths()
        nside = get_healpix_nside()
        
        # Pre-compute hemisphere grid:
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        mask = theta < np.pi/2  # Upper hemisphere
        hp_coords = SkyCoord(phi[mask], np.pi/2-theta[mask], unit='rad', frame=observation.AltAz)

        # Transform_coordinates to SunRelativeEclipticFrame
        hp_coords = hp_coords.transform_to(SunRelativeEclipticFrame())
        
        # Evaluate on spline:
        lon, lat = hp_coords.alpha.rad, hp_coords.beta.rad
        weights = self.A(np.abs(np.asarray([(lon + np.pi) % (2 * np.pi) - np.pi, lat]).T)) * 1e-8 * u.W/u.m**2/u.sr/u.micron
        
        # Get solar spectrum:
        wvl, spectrum = SolarSpectrumRieke2008()
        value_500nm = np.interp(0.5*u.micron, wvl, spectrum)
        spec_samp = np.interp(wavelengths*u.nm, wvl, spectrum)/value_500nm/(h*c/wavelengths*u.nm) 
        spectra = spec_samp*self.color_correction(wavelengths, np.clip(np.rad2deg(lat), 30, 90))

        # Combined and transform:
        flux_map = jnp.array((weights[:,None]*spectra).value)
        
        def generator(params):
            return DiffuseQuery(flux_map=flux_map)

        param_specs = {}

        return generator, param_specs