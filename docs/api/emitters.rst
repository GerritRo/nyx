Emitters
========

The emitters are the physical light sources of the night sky. Each one is
two objects:

- a **builder** -- :class:`~nyx.emitter.base.Emitter` and its subclasses --
  which holds astropy and numpy state and turns an
  :class:`~nyx.core.observation.Observation` into arrays, in ``prepare(obs)``.
- a **runtime source** -- :class:`~nyx.emitter.sources.SpectralSource` --
  which the builder hands to the scene in ``model()``.  It is a JAX pytree,
  it is traced and differentiated, and it owns every
  :class:`~nyx.core.parameter.Parameter` the optimizer can reach.

============================ ========= ======= =================================
Emitter                      Path      Scales? Model
============================ ========= ======= =================================
:class:`~nyx.emitter.Airglow`        diffuse no  ESO SkyCalc, with an SFU curve
:class:`~nyx.emitter.ZodiacalLight`  diffuse yes Leinert et al. (1998)
:class:`~nyx.emitter.Moon`           point   yes ROLO, after Jones et al. (2013)
:class:`~nyx.emitter.Stars`          both    yes Gaia DR3 + Pickles (1998)
:class:`~nyx.emitter.BrightStars`    point   yes XHIP or Gaia, points only
:class:`~nyx.emitter.PointSource`    point   yes one source, one spectrum
============================ ========= ======= =================================

"Scales" is whether the emitter takes a ``brightness``: a fittable amplitude
on everything it emits, on both the diffuse and the point path. 
``Airglow`` is the exception: its spectral model already evaluates
``0.2 + 0.00614 * sfu(t)`` with a trainable ``sfu``.

Writing your own emitter
------------------------

Subclass :class:`~nyx.emitter.base.Emitter`, call ``super().__init__(geo,
spectral_model, ...)`` and implement ``_prepare(obs)``.  The base records the
geometry signature that lets :meth:`~nyx.core.scene.Scene.build` catch a
component built against a different ``Geometry``, validates ``brightness``,
and checks the frame count before every ``prepare``.

``_prepare`` returns a :class:`~nyx.core.records.SourceObsData`. Two flags on
it choose the render path:

``direct``
    Whether diffuse radiance is extincted along the line of sight.
``inscatter``
    Whether point sources are in-scattered individually, giving each its own
    scattered halo.

The builder layer
-----------------

.. automodapi:: nyx.emitter.base
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

.. automodapi:: nyx.emitter.catalog
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

The runtime layer
-----------------

.. automodapi:: nyx.emitter.sources
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

Zodiacal Light
--------------

.. automodapi:: nyx.emitter.zodiacal
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

Airglow
-------

.. automodapi:: nyx.emitter.airglow
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

Moon
----

.. automodapi:: nyx.emitter.moon
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

Stars
-----

.. automodapi:: nyx.emitter.stars
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

Point Sources
-------------

.. automodapi:: nyx.emitter.point_source
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

Catalogs
--------

.. automodapi:: nyx.emitter.catalogs.gaia
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

.. automodapi:: nyx.emitter.catalogs.xhip
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

.. automodapi:: nyx.emitter.catalogs.astrometry
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

.. automodapi:: nyx.emitter.catalogs.index
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx
