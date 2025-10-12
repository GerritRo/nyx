import jax.numpy as jnp
import numpy as np
import healpy as hp
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.constants import c, h, k_B
from astropy.utils.data import download_file
from sklearn.neighbors import BallTree

from nyx.core.scene import ComponentType
from nyx.core import get_wavelengths, get_healpix_nside
from nyx.core import CatalogQuery, ParameterSpec
from nyx.core.model import EmitterProtocol
from nyx.atmosphere import get_airmass_formula


class SpectralModel:
    """Abstract base class for spectral models"""
    def get_spectrum(self, wavelengths, magnitude_data):
        """
        Returns spectrum for given magnitude data
        
        Parameters
        ----------
        wavelengths : array in nm
        magnitude_data : dict with 'G', 'BP', 'RP' magnitudes
        
        Returns
        -------
        spectrum : array of flux values
        """
        raise NotImplementedError


class BlackbodySpectralModel(SpectralModel):
    """Simple blackbody model based on color temperature"""
    
    def __init__(self):
        # Approximate color temperature from BP-RP color
        # This is a rough approximation - real stars are more complex
        self.color_to_temp = lambda bp_rp: 7500 / (bp_rp + 0.8)  # Very rough!
        
    def get_spectrum(self, wavelengths, magnitude_data):
        """Generate blackbody spectrum based on color"""
        bp_rp = magnitude_data['BP'] - magnitude_data['RP']
        temp = self.color_to_temp(bp_rp)
        
        # Blackbody spectrum
        wl_m = wavelengths * 1e-9  # nm to m
        # Planck function
        B_lambda = (2 * h * c**2 / wl_m**5 / 
                    (jnp.exp(h * c / (wl_m * k_B * temp * u.K)) - 1))
        
        # Scale by magnitude (simplified - should properly handle Vega magnitudes)
        flux_scale = 10**(-0.4 * magnitude_data['G'])
        
        # Convert to photon flux (photons/s/m^2/nm)
        photon_energy = h * c / wl_m
        photon_flux = B_lambda / photon_energy * flux_scale * 1e-9  # per nm
        
        return photon_flux.value


class GaiaDR3Stars(EmitterProtocol):   
    def __init__(self, catalog_file=None, spectral_model=None, mag_limit=15.0):
        self.mag_limit = mag_limit
        self.spectral_model = spectral_model or BlackbodySpectralModel()
        
        # Load catalog
        if catalog_file is None:
            # Download from Zenodo (using the URL from nsb2)
            catalog_file = download_file(
                'https://zenodo.org/records/15396676/files/gaiadr3.npy', 
                cache=True
            )
        
        self.catalog = np.load(catalog_file)
        
        # Filter by magnitude
        mask = self.catalog['phot_g_mean_mag'] < mag_limit
        self.catalog = self.catalog[mask]
        
        # Store coordinates
        self.coords = SkyCoord(
            self.catalog['ra'] * u.deg,
            self.catalog['dec'] * u.deg,
            frame='icrs'
        )
        
        # Build spatial index for efficient queries
        self.build_balltree()
        
    def build_balltree(self):
        """Build BallTree for efficient spatial queries"""
        # Convert to radians for haversine metric
        coords_rad = np.vstack([
            self.coords.spherical.lat.rad,
            self.coords.spherical.lon.rad
        ]).T
        self.balltree = BallTree(coords_rad, metric='haversine')
        
    def query_fov(self, center_coord, radius):
        center_rad = np.array([
            center_coord.spherical.lat.rad,
            center_coord.spherical.lon.rad
        ])
        
        indices = self.balltree.query_radius([center_rad], radius)[0]
        return indices
        
    def get_generator(self, observation):       
        # Load global parameters
        wavelengths = get_wavelengths()
        nside = get_healpix_nside()
        
        # Query stars in FOV (with some margin)
        fov_center = observation.target
        fov_radius = observation.fov.value if observation.fov else 0.1  # radians
        
        star_indices = self.query_fov(fov_center, fov_radius * 1.5)
        
        if len(star_indices) == 0:
            # No stars in FOV - return empty catalog
            def generator(params):
                return CatalogQuery(
                    sec_Z=jnp.array([]),
                    image_coords=jnp.array([]).reshape(0, 2),
                    flux_values=jnp.array([]).reshape(0, len(wavelengths)),
                    flux_map=jnp.zeros((hp.nside2npix(nside)//2, len(wavelengths)))
                )
            return generator, {}, ComponentType.CATALOG
            
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
        
        # Generate spectra for each star
        flux_values = []
        for i in range(len(fov_stars)):
            magnitude_data = {
                'G': fov_stars['phot_g_mean_mag'][i],
                'BP': fov_stars['phot_bp_mean_mag'][i],
                'RP': fov_stars['phot_rp_mean_mag'][i]
            }
            spectrum = self.spectral_model.get_spectrum(wavelengths, magnitude_data)
            flux_values.append(spectrum)
        
        flux_values = np.array(flux_values)
        
        # Create HEALPix map representation
        # This is a simplified version - could be improved
        npix = hp.nside2npix(nside)
        flux_map = np.zeros((npix, len(wavelengths)))
        
        # Add stars to map (simplified - just nearest pixel)
        for i, coord in enumerate(star_coords_altaz):
            if coord.alt.deg > 0:  # Above horizon
                hp_idx = hp.ang2pix(
                    nside,
                    coord.az.deg,
                    coord.alt.deg,
                    lonlat=True
                )
                flux_map[hp_idx] += flux_values[i]
        
        # Only keep upper hemisphere
        theta, phi = hp.pix2ang(nside, np.arange(npix))
        mask = theta < np.pi/2
        flux_map_hemisphere = flux_map[mask]
        
        # Convert to JAX arrays
        sec_Z_jax = jnp.array(sec_Z)
        image_coords_jax = jnp.array(image_coords)
        flux_values_jax = jnp.array(flux_values)
        flux_map_jax = jnp.array(flux_map_hemisphere)
        
        def generator(params):
            # Apply any runtime parameters (e.g., extinction correction)
            extinction_corr = params.get('extinction_correction', 1.0)
            
            return CatalogQuery(
                sec_Z=sec_Z_jax,
                image_coords=image_coords_jax,
                flux_values=flux_values_jax * extinction_corr,
                flux_map=flux_map_jax * extinction_corr
            )
        
        param_specs = {
            'extinction_correction': ParameterSpec(
                (1,), 1.0, 
                description="Extinction correction factor",
                bounds=(0.1, 2.0)
            )
        }
        
        return generator, param_specs, ComponentType.CATALOG


class GaiaStarsWithMap(EmitterProtocol):    
    def __init__(self, 
                 bright_catalog_file=None,
                 faint_map_file=None,
                 bright_limit=11.0,
                 spectral_model=None):
        
        self.bright_limit = bright_limit
        self.spectral_model = spectral_model or BlackbodySpectralModel()
        
        # Load bright star catalog
        self.bright_stars = GaiaDR3Stars(
            catalog_file=bright_catalog_file,
            spectral_model=spectral_model,
            mag_limit=bright_limit
        )
        
        # Load faint star map
        if faint_map_file is None:
            faint_map_file = download_file(
                'https://zenodo.org/records/15396676/files/gaia_mag15plus.npy',
                cache=True
            )
        
        self.faint_map = np.load(faint_map_file)
        # faint_map structure: [G_mag, BP_mag, RP_mag] maps
        
    def get_generator(self, observation):
        """Combine bright catalog and faint map"""
        
        # Get generators for both components
        bright_gen, bright_specs, _ = self.bright_stars.get_generator(observation)
        
        # For faint stars, create diffuse-like contribution
        # (This is simplified - full implementation would be more complex)
        
        # Combine into single generator
        def generator(params):
            bright_query = bright_gen(params)
            # Add faint star contribution to flux_map
            # ... (implementation details)
            return bright_query
            
        return generator, bright_specs, ComponentType.CATALOG