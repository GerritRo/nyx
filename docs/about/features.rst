Features
========

This page provides an overview of nyx's capabilities.

Fully differentiable forward model
----------------------------------

The entire rendering pipeline is built on `JAX <https://jax.dev>`_, so every
predicted pixel rate is differentiable with respect to the model parameters.

- **End-to-end gradients** through emitters, atmosphere, and instrument enable
  gradient-based optimisation and Bayesian inference (HMC/NUTS).
- **JIT compilation and vectorisation** via ``jax.jit`` and ``jax.vmap`` for
  fast evaluation over many observations.
- **Pytree-based models** built on `Equinox <https://docs.kidger.site/equinox/>`_
  modules, so models compose cleanly and play well with JAX transformations.

Modular, composable architecture
---------------------------------

A simulation is assembled from three interchangeable component types and a
shared observation/geometry description, then combined into a
:class:`~nyx.core.Scene` for rendering:

- **Instrument** — the detector response (:mod:`nyx.instrument`).
- **Atmosphere** — extinction and scattering (:mod:`nyx.atmosphere`).
- **Emitters** — the physical light sources (:mod:`nyx.emitter`).

Each component follows a small set of protocols (:class:`~nyx.core.InstrumentModel`,
:class:`~nyx.core.AtmosphereModel`, :class:`~nyx.core.SkySource`), so custom
components can be dropped in without changing the rest of the pipeline. The
:func:`~nyx.core.render` pipeline routes each source through the appropriate
rendering path (diffuse map scattering, line-of-sight extinction, and
individual in-scattering) based on per-observation flags.

Physical sky emitters
---------------------

nyx ships with the dominant contributors to the night sky background, each
backed by a published model or dataset:

- **Zodiacal light** — :class:`~nyx.emitter.ZodiacalLight`, based on the
  Leinert et al. (1998) brightness tables with solar-spectrum colour
  correction.
- **Airglow** — :class:`~nyx.emitter.Airglow`, with a solar-activity (SFU)
  scaling and a van Rhijn zenith-angle dependence; supports ESO SkyCalc
  templates.
- **Lunar brightness** — :class:`~nyx.emitter.Moon`, a scattered-moonlight
  model conditioned on phase angle, Earth–Moon distance, and Sun angle.
- **Stellar catalogs** — :class:`~nyx.emitter.Stars`, combining a diffuse
  all-sky map with resolved bright stars, with support for Gaia DR3. Resolved
  star flux is handled separately from the diffuse map to avoid
  double-counting.

Atmospheric extinction and scattering
-------------------------------------

The atmosphere module provides a differentiable single-scattering radiative
transfer model (:class:`~nyx.atmosphere.SingleScattering`) assembled from
reusable physical components:

- **Rayleigh scattering** with the Rayleigh phase function
  (:class:`~nyx.atmosphere.RayleighComponent`).
- **Mie / aerosol scattering** via a Henyey–Greenstein phase function
  (:class:`~nyx.atmosphere.HenyeyGreensteinComponent`).
- **Tabulated absorption** such as ozone (:class:`~nyx.atmosphere.TabulatedAbsorption`),
  with built-in optical-depth helpers (:func:`~nyx.atmosphere.tau_rayleigh`,
  :func:`~nyx.atmosphere.tau_mie`, :func:`~nyx.atmosphere.tau_ozone`).
- **Airmass models** including Kasten & Young (1989) and the plane-parallel
  approximation.

A lightweight :class:`~nyx.atmosphere.HGNoAbsorption` variant is available for
fast experiments without absorption.

Instrument response
-------------------

The instrument model captures the telescope's effective collecting area and
spectral response:

- **Effective-aperture instrument** (:class:`~nyx.instrument.EffectiveApertureInstrument`)
  with a wavelength-dependent bandpass and per-pixel efficiency.
- **Mirror misalignment** support
  (:class:`~nyx.instrument.EffectiveApertureMisalignmentInstrument`) for
  modelling and fitting mirror misalignment effects.
- **Save/load** of instrument definitions to HDF5
  (:func:`~nyx.instrument.save_instrument`, :func:`~nyx.instrument.load_instrument`).

Flexible spectral models
------------------------

Source spectra are described by composable spectral models
(:mod:`nyx.core.spectral`):

- **Stored spectra** (:class:`~nyx.core.StoredSpectrum`) from empirical or
  template libraries.
- **Parametric spectra** (:class:`~nyx.core.ParametricSpectrum`) driven by a
  user-supplied, differentiable function of physical conditions.
- **Pass-through spectra** (:class:`~nyx.core.PassThroughSpectrum`) for sources
  that already carry per-wavelength radiance.
- Flux-conserving resampling onto the model wavelength grid
  (:func:`~nyx.core.resample_flux`).

nyx bundles solar, stellar (Pickles 1998), and empirical spectral templates,
together with CALSPEC standards and SVO filter profiles (see :doc:`data`).

Parameter handling and fitting
------------------------------

Model parameters are differentiable objects with explicit
characteristic scales (:class:`~nyx.core.Parameter`), which keeps the optimiser
well-conditioned across physically disparate quantities:

- **Freeze / unfreeze** parameters individually or in bulk
  (:func:`~nyx.core.freeze`, :func:`~nyx.core.unfreeze`,
  :func:`~nyx.core.freeze_all`, :func:`~nyx.core.unfreeze_all`) to control what
  is fit.
- **Automatic scaling** of parameters (:func:`~nyx.core.autoscale`).
- **Optimisation** via the :class:`~nyx.core.Optimizer`, built on
  `Optimistix <https://docs.kidger.site/optimistix/>`_, with multi-target joint
  fitting through :class:`~nyx.core.MultiTargetFit`.
- **Uncertainty estimation** with :func:`~nyx.core.parameter_errors`.

Observation and geometry model
------------------------------

- **HEALPix-based sky discretisation** (:class:`~nyx.core.Geometry`) with
  configurable resolution (``nside``), in-scattering grid, and field of view.
- **Observation description** (:class:`~nyx.core.Observation`) tying together
  observer location, time, target, geometry and atmospheric conditions.
See :doc:`/api/index` for the complete API reference, and the
:doc:`/examples/index` for worked end-to-end examples.