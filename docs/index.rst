nyx
===

**Python package for differentiable simulation of the night sky background in IACTs**

nyx models the night sky background for IACT observations,
enabling parameter estimation through HMC fitting or gradient-based optimization.

.. code-block:: python

   import jax.numpy as jnp
   import astropy.units as u

   from nyx.core import Scene
   from nyx.core.geometry import Geometry
   from nyx.core.observation import Observation
   from nyx.instrument import EffectiveApertureInstrument
   from nyx.atmosphere.single_scattering import HGNoAbsorption
   from nyx.emitter import Moon, Airglow, ZodiacalLight, Stars

   geo = Geometry(wvls=jnp.linspace(300, 700, 50) * u.nm,
                  nside=16, ngrid=2, fov=3.5 * u.deg)

   instrument = EffectiveApertureInstrument.load("instrument.h5", geo)
   atmosphere = HGNoAbsorption(geo)
   sources = {
      'airglow': Airglow.from_eso_skycalc(geo, sfu=100.0),
      'zodiacal': ZodiacalLight.from_leinert1998(geo),
      'GaiaDR3': Stars.from_gaia_dr3(geo, lim_mag=12),
      'moon': Moon.from_jones2013(geo),
   }

   obs = Observation(location, times, target, geo)
   scene = Scene.build(instrument, atmosphere, sources, obs)
   rates = scene.render()

----

.. toctree::
   :maxdepth: 2
   :caption: About This Project

   about/introduction
   about/motivation
   about/features
   about/data

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart

.. toctree::
   :maxdepth: 2
   :caption: Example Gallery

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`