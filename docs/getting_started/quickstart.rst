Quickstart Guide
================

This guide introduces the basic workflow for using nyx to simulate the night
sky brightness, then shows how to fit the model to data and save the result.

Basic Concepts
--------------

nyx models astronomical observations through a small set of composable
components, all sharing a common :class:`~nyx.core.Geometry`:

1. **Instrument** — telescope characteristics (effective aperture, bandpass,
   pixel response).
2. **Atmosphere** — atmospheric scattering and absorption.
3. **Emitters** — sky brightness sources (zodiacal light, airglow, moon,
   stars).
4. **Observation** — observer location, time(s), and pointing.

These are combined into a :class:`~nyx.core.Scene`, which renders to per-pixel
photon detection rates.

Working with Units
------------------

nyx is Astropy-aware: physical inputs can be supplied as
:class:`~astropy.units.Quantity` objects and are converted internally.

.. code-block:: python

   import astropy.units as u

   fov = 3.5 * u.deg          # field of view half-angle
   wvls = [300, 400, 500] * u.nm

The Geometry
------------

The :class:`~nyx.core.Geometry` defines the resolution and grid shared by every
component:

.. code-block:: python

   import jax.numpy as jnp
   import astropy.units as u

   from nyx.core.geometry import Geometry

   geo = Geometry(
       wvls=jnp.linspace(300, 700, 50) * u.nm,  # wavelength grid (nm)
       nside=16,                                # HEALPix resolution of the sky hemisphere
       ngrid=2,                                 # in-scattering grid points per FOV axis
       fov=3.5 * u.deg,                         # field of view half-angle
   )

Higher ``nside`` and ``ngrid`` increase accuracy at the cost of compute and
memory; start small and increase as needed.

A Complete Example
------------------

The following builds a full scene and renders it to pixel rates. The emitters
use convenience constructors that load their backing datasets (Leinert 1998,
ESO SkyCalc, Gaia DR3, Jones 2013):

.. code-block:: python

   import jax.numpy as jnp
   import astropy.units as u
   from astropy.coordinates import EarthLocation, SkyCoord
   from astropy.time import Time

   from nyx.core import Scene
   from nyx.core.geometry import Geometry
   from nyx.core.observation import Observation
   from nyx.instrument import EffectiveApertureInstrument
   from nyx.atmosphere.single_scattering import HGNoAbsorption
   from nyx.emitter import Moon, Airglow, ZodiacalLight, Stars

   # 0. Shared render geometry
   geo = Geometry(wvls=jnp.linspace(300, 700, 50) * u.nm,
                  nside=16, ngrid=2, fov=3.5 * u.deg)

   # 1. Instrument (loaded from an HDF5 definition)
   instrument = EffectiveApertureInstrument.load("instrument.h5", geo)

   # 2. Atmosphere
   atmosphere = HGNoAbsorption(geo)

   # 3. Emitters, keyed by name
   sources = {
       "airglow": Airglow.from_eso_skycalc(geo, sfu=100.0),
       "zodiacal": ZodiacalLight.from_leinert1998(geo),
       "GaiaDR3": Stars.from_gaia_dr3(geo, lim_mag=12),
       "moon": Moon.from_jones2013(geo),
   }

   # 4. Observation: location, time(s), target, geometry
   location = EarthLocation.of_site("Roque de los Muchachos")
   times = Time(["2026-06-05T23:00:00"])
   target = SkyCoord(ra=83.6 * u.deg, dec=22.0 * u.deg)
   obs = Observation(location, times, target, geo)

   # 5. Build the scene
   scene = Scene.build(instrument, atmosphere, sources, obs)

   # 6. Render to per-pixel photon rates
   rates = scene.render()

``Scene.render()`` returns a ``dict`` keyed by instrument name; each value is a
JAX array of photon detection rates per pixel ``[photon/s]``. A single
instrument is automatically named ``"instrument"``:

.. code-block:: python

   pixel_rates = rates["instrument"]  # shape (n_obs, n_pixels)

Visualizing the Result
----------------------

Because the model is built on JAX arrays, results drop straight into the usual
plotting tools:

.. code-block:: python

   import matplotlib.pyplot as plt

   plt.hist(pixel_rates[0])
   plt.xlabel("photon rate [photon/s]")
   plt.ylabel("pixels")
   plt.show()

Fitting the Model to Data
-------------------------

The whole pipeline is differentiable, so you can fit model parameters to
measured pixel rates with gradient-based optimisation. Parameters are
:class:`~nyx.core.Parameter` objects; use :func:`~nyx.core.freeze` /
:func:`~nyx.core.unfreeze` (with selector lambdas that address the parameter in
the scene pytree) to choose what is trained.

.. code-block:: python

   import optimistix as optx
   from nyx.core import Optimizer, freeze_all, unfreeze

   # Start from everything frozen, then unfreeze what you want to fit.
   # The selector returns the Parameter to train, e.g. the instrument
   # pixel efficiency:
   scene = freeze_all(scene)
   scene = unfreeze(scene, lambda s: s.instruments["instrument"].pixel_efficiency)

   # Residuals against measured rates (least-squares).
   def residuals(scene):
       return scene.render()["instrument"] - measured_rates

   opt = Optimizer(residuals, optx.LevenbergMarquardt(rtol=1e-6, atol=1e-6))
   fitted, sol = opt.run(scene)

   # 1-sigma parameter uncertainties at the solution
   errors = opt.errors(fitted)

For jointly fitting several targets/observations, see
:class:`~nyx.core.MultiTargetFit`.

Saving and Loading a Fit
------------------------

Fitted scenes can be serialised to HDF5 and reloaded later. The observation
dict is required because the scene keeps only precomputed JAX pytrees, not the
original Astropy objects:

.. code-block:: python

   from nyx.core import save_fit, load_fit

   save_fit("fit.h5", fitted, {"instrument": obs})
   result = load_fit("fit.h5")

Next Steps
----------

- Work through the example notebooks:

  - :doc:`/examples/Example`
  - :doc:`/examples/MultiTargetFlatfieldFit`
  - :doc:`/examples/GlobalPointingFit`
  - :doc:`/examples/PosteriorEstimation`

- Read about the design and capabilities in :doc:`/about/features`.
- Explore the full :doc:`/api/index` reference.