Data Attribution
================

nyx bundles and/or downloads scientific datasets produced by third parties.
This document records, for each dataset, its provenance, the reference to
cite, and its license or usage terms.

If you publish results obtained with nyx, you are responsible for citing the
datasets you used in addition to citing nyx itself (see ``CITATION.cff`` in
the repository root).

The nyx source code is licensed under BSD-3-Clause (see ``LICENSE.rst``).
The datasets described below are **not** covered by that license; they
remain under the terms of their respective originators. In particular, the
Gaia-derived dataset downloaded at runtime is distributed by ESA under a
non-commercial (CC BY-NC 3.0 IGO) license; commercial use of nyx in
configurations that rely on this dataset requires separate arrangement with
ESA.


Bundled data (shipped in ``nyx/data/``)
---------------------------------------

Pickles (1998) stellar spectral atlas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Files: ``pickles1998_trds_atlas.dat``
:Used by: ``nyx.utils.spectra.PicklesTRDSAtlas1998``
:Description: Library of stellar spectral templates spanning a wide range of
   spectral types, used here as the empirical spectral model for stars.
:Reference: Pickles, A. J. 1998, "A Stellar Spectral Flux Library:
   1150-25000 A", PASP, 110, 863. DOI: 10.1086/316197
:Usage terms: Published in the peer-reviewed literature and freely available
   for research use; please cite the reference above.

Leinert et al. (1998) zodiacal light
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Files: ``leinert1998_zodiacal_light.dat``
:Used by: ``nyx.emitter.zodiacal.Leinert1998``
:Description: Tabulated zodiacal light brightness as a function of ecliptic
   coordinates.
:Reference: Leinert, C., Bowyer, S., Haikala, L. K., et al. 1998, "The 1997
   reference of diffuse night sky brightness", A&AS, 127, p. 1-99
   DOI: 10.1051/aas:1998105
:Usage terms: Published in the peer-reviewed literature and freely available
   for research use; please cite the reference above.

ROLO lunar irradiance model coefficients
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Files: ``jones2013_lunar_rolo.dat``
:Used by: ``nyx.emitter.moon.Jones2013``
:Description: Fit coefficients of the ROLO (RObotic Lunar Observatory) lunar
   irradiance model, in 22 wavelength bands.
:Note on naming: The file name and the ``Jones2013`` API name refer to the
   sky-brightness model of Jones et al. (2013), which adopts the ROLO model.
   The coefficient values themselves originate with Kieffer & Stone (2005).
   Both works should be cited.
:References:
   * Kieffer, H. H. & Stone, T. C. 2005, "The Spectral Irradiance of the
     Moon", AJ, 129, 2887. DOI: 10.1086/430185
   * Jones, A., Noll, S., Kausch, W., et al. 2013, "An advanced scattered
     moonlight model for Cerro Paranal", A&A, 560, A91.
     DOI: 10.1051/0004-6361/201322433
:Usage terms: Published in the peer-reviewed literature and freely available
   for research use; please cite the references above.

ESO SkyCalc airglow and ozone spectra
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Files: ``eso_skycalc_airglow_130sfu.dat``,
   ``eso_skycalc_ozone_absorption.dat``
:Used by: ``nyx.emitter.airglow.ESOSkyCalc`` and the ozone transmission code
:Description: Airglow emission spectrum (at 130 solar flux units) and ozone
   absorption spectrum, generated with the ESO SkyCalc Sky Model Calculator
   (https://www.eso.org/observing/etc/skycalc/).
:References:
   * Noll, S., Kausch, W., Barden, M., et al. 2012, "An atmospheric radiation
     model for Cerro Paranal", A&A, 543, A92. DOI: 10.1051/0004-6361/201219040
   * Jones, A., Noll, S., Kausch, W., et al. 2013, A&A, 560, A91.
     DOI: 10.1051/0004-6361/201322433
:Usage terms: Output of the publicly available ESO SkyCalc tool; please cite
   the references above and acknowledge the ESO SkyCalc tool.

Data downloaded at runtime
--------------------------

These datasets are not shipped with nyx; they are fetched on first use and
cached locally via :func:`astropy.utils.data.download_file`.

Gaia DR3 stellar catalogue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Source: Zenodo record 15396676
   (``gaiadr3.npy``, ``gaia_mag15plus.npy``);
   https://zenodo.org/records/15396676
:Used by: ``nyx.emitter.stars.GaiaDR3``
:Description: Simplified star catalogue derived from Gaia Data Release 3
   catalogue, repackaged for stellar-contribution simulations.
:References:
   * Gaia Collaboration, Prusti, T., de Bruijne, J. H. J., et al. 2016,
     "The Gaia mission", A&A, 595, A1. DOI: 10.1051/0004-6361/201629272
   * Gaia Collaboration, Vallenari, A., Brown, A. G. A., et al. 2023,
     "Gaia Data Release 3", A&A, 674, A1.
     DOI: 10.1051/0004-6361/202243940
:License: Distributed under the `CC BY-NC 3.0 IGO
   <https://creativecommons.org/licenses/by-nc/3.0/igo/deed.en>`_ license,
   matching the terms under which ESA distributes Gaia data. See the
   `Gaia Data License <https://www.cosmos.esa.int/web/gaia-users/license>`_
   and the `ESA Science Archives Terms and Conditions
   <https://www.cosmos.esa.int/web/esdc/terms-and-conditions>`_.
   **Commercial use requires separate arrangement with ESA.**
   Credit: ESA / Gaia / DPAC (Gaia DR3); ESA (Hipparcos).
:Required acknowledgement: Publications must include the verbatim Gaia
   acknowledgement:

      This work has made use of data from the European Space Agency (ESA)
      mission Gaia (https://www.cosmos.esa.int/gaia), processed by the Gaia
      Data Processing and Analysis Consortium (DPAC,
      https://www.cosmos.esa.int/web/gaia/dpac/consortium). Funding for the
      DPAC has been provided by national institutions, in particular the
      institutions participating in the Gaia Multilateral Agreement.

   See the `Gaia DR3 credit and citation instructions
   <https://gea.esac.esa.int/archive/documentation/GDR3/Miscellaneous/sec_credit_and_citation_instructions/>`_
   for full requirements.

STScI solsys solar spectrum (Rieke 2008)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:Source: STScI synphot reference atlases, ``grid/solsys/solar_spec.fits``
   (https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/)
:Used by: ``nyx.utils.spectra.SolarSpectrumRieke2008``,
   ``nyx.utils.spectra.load_solar_flux``
:Description: Composite solar spectrum (0.2–30 µm) distributed with Rieke
   et al. (2008). Built from Thuillier et al. (2003) below 2.4 µm, the
   Holweger & Müller (1974) LTE model at longer wavelengths, with the
   4–6.5 µm region adjusted to match Wallace & Livingston (2003) CO
   features. Normalized to 1 AU.
:Reference: Rieke, G. H., Blaylock, M., Decin, L., et al. 2008,
   "Absolute Physical Calibration in the Infrared", AJ, 135, 2245.
   DOI: 10.1088/0004-6256/135/6/2245
:Usage terms: Publicly distributed by STScI.

STScI CALSPEC Vega spectrum
~~~~~~~~~~~~~~~~~~~~~~~~~~~
:Source: STScI CALSPEC database, ``alpha_lyr_stis_011.fits``
   (https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec)
:Used by: the ``Bandpass`` zero-point code
:Description: Composite absolute-flux SED of Vega (α Lyr) on the HST/CALSPEC
   flux scale, combining STIS spectrophotometry with a tailored Kurucz
   9550 K model atmosphere. Used as the primary optical/IR flux standard.
:Reference: Bohlin, R. C., Hubeny, I. & Rauch, T. 2020, "New Grids of
   Pure-hydrogen White Dwarf NLTE Model Atmospheres and the HST/STIS Flux
   Calibration", AJ, 160, 21. DOI: 10.3847/1538-3881/ab94b4;
   Bohlin, R. C. 2014, "Hubble Space Telescope CALSPEC Flux Standards:
   Sirius (and Vega)", AJ, 147, 127. DOI: 10.1088/0004-6256/147/6/127
:Usage terms: Publicly distributed by STScI.

SVO Filter Profile Service
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:Source: Spanish Virtual Observatory Filter Profile Service
   (http://svo2.cab.inta-csic.es/theory/fps/)
:Used by: ``nyx.utils.spectra.Bandpass.from_SVO`` (e.g. the Gaia DR3 G/BP/RP
   passbands)
:Description: Filter transmission curves.
:References:
   * Rodrigo, C., Cruz, P., Aguilar, J.F., et al. 2024; https://ui.adsabs.harvard.edu/abs/2024A%26A...689A..93R/abstract
   * The SVO Filter Profile Service. Rodrigo, C., Solano, E., Bayo, A., 2012; https://ui.adsabs.harvard.edu/abs/2012ivoa.rept.1015R/abstract
   * The SVO Filter Profile Service. Rodrigo, C., Solano, E., 2020; https://ui.adsabs.harvard.edu/abs/2020sea..confE.182R/abstract
:Required acknowledgement: Publications should include this acknowledgement:

      This research has made use of the SVO Filter Profile Service "Carlos Rodrigo",
      funded by MCIN/AEI/10.13039/501100011033/ through grant PID2023-146210NB-I00.