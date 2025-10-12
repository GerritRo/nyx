import astropy.units as u
from astropy.utils.data import download_file
from astropy.io import fits


def SolarSpectrumRieke2008():
    f_down = download_file('https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/solar_spec.fits', cache=True)
    hdul = fits.open(f_down)
    return hdul[1].data['WAVELENGTH']*u.angstrom, hdul[1].data['FLUX']*u.erg/u.s/u.cm**2/u.angstrom
