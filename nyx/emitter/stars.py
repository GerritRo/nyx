import jax.numpy as jnp
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.utils.data import download_file
from sklearn.neighbors import BallTree
from jax.scipy.interpolate import RegularGridInterpolator

from nyx.core.scene import ComponentType
from nyx.core import get_wavelengths, get_healpix_nside
from nyx.core import CatalogQuery, ParameterSpec
from nyx.core.model import EmitterProtocol
from nyx.atmosphere import get_airmass_formula

from nyx.units import nixify
from nyx.core.spectral import Bandpass, SpectralHandler
from nyx.utils.spectra import create_color_grid, PicklesTRDSAtlas1998

class GaiaDR3(EmitterProtocol):   
    def __init__(self, lim_mag=15.0):
        # Download catalog
        catalog_file = download_file('https://zenodo.org/records/15396676/files/gaiadr3.npy', cache=True)
        catalog = np.load(catalog_file)

        self.catalog = catalog[catalog['phot_g_mean_mag']<lim_mag]
        self.coords = SkyCoord(self.catalog['ra']*u.deg, self.catalog['dec']*u.deg, frame='icrs')

        # Get Bandpass
        G  = Bandpass.from_SVO('GAIA/GAIA3.G')
        BP = Bandpass.from_SVO('GAIA/GAIA3.Gbp')
        RP = Bandpass.from_SVO('GAIA/GAIA3.Grp')

        # Create color grid
        rp_bp = self.catalog['phot_rp_mean_mag'] - self.catalog['phot_bp_mean_mag']
        spec_grid = create_color_grid(G, (RP, BP), [np.nanmin(rp_bp), 0.5], PicklesTRDSAtlas1998())

        # Get median as spectrum for now:
        self.points = spec_grid.points
        self.wvl = nixify(spec_grid.wvl, 'wavelength')
        self.flx = nixify(np.nanmedian(spec_grid.flx, axis=-1)*u.ph, 'flux', wavelength=spec_grid.wvl)
        
        # Build spatial index for efficient queries
        self.build_balltree()

    def build_balltree(self):
        self.balltree = BallTree(self.skycoord2localcoord(self.coords), metric='haversine')
        
    def skycoord2localcoord(self, skycoord):
        skycoord = skycoord.transform_to('icrs')
        return np.vstack([skycoord.spherical.lat.rad, skycoord.spherical.lon.rad]).T

    def query_fov(self, center_coord, radius):
        center_rad = self.skycoord2localcoord(center_coord)
        
        indices = self.balltree.query_radius(center_rad, radius)[0]
        return indices

    def get_generator(self, observation):       
        # Load global parameters
        wavelengths = get_wavelengths()
        nside = get_healpix_nside()

        # Query catalog
        star_indices = self.query_fov(observation.target, observation.fov.rad)
            
        # Get star data for FOV
        fov_stars = self.catalog[star_indices]
        fov_coords = self.coords[star_indices]
        
        # Transform to observation frame
        star_coords_altaz = fov_coords.transform_to(observation.AltAz)
        star_coords_image = star_coords_altaz.transform_to(observation.frame)
        
        # Calculate airmass
        airmass_func = get_airmass_formula()
        sec_Z = airmass_func(np.pi/2 - star_coords_altaz.alt.rad)
        
        # Get image coordinates
        image_coords = np.column_stack([
            star_coords_image.lon.rad,
            star_coords_image.lat.rad
        ])

        # Calculate flux of stars in view:
        new_samp = SpectralHandler.resample(self.wvl, self.flx, wavelengths)
        interpol = RegularGridInterpolator((self.points), new_samp, method='linear', fill_value=0)
        rp_bp = fov_stars['phot_rp_mean_mag'] - fov_stars['phot_bp_mean_mag']
        rp_bp[~np.isfinite(rp_bp)] = np.nanmedian(rp_bp)
        rp_bp = jnp.clip(jnp.array(rp_bp), a_max=0.49) # Clip to avoid illegal values
        flux_values = 10**(-0.4*fov_stars['phot_g_mean_mag'])[:,None]*interpol(rp_bp)

        # Create (empty) map for now:
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        mask = theta < np.pi/2  # Upper hemisphere
        flux_map = np.zeros((np.sum(mask), len(wavelengths)))

        # Convert to jax arrays
        sec_Z = jnp.array(sec_Z)
        image_coords = jnp.array(image_coords)
        flux_values = jnp.array(flux_values)
        flux_map = jnp.array(flux_map)
        
        def generator(params):
            return CatalogQuery(
                sec_Z=sec_Z,
                image_coords=image_coords,
                flux_values=flux_values,
                flux_map=flux_map
            )
        
        param_specs = {}

        return generator, param_specs, ComponentType.CATALOG