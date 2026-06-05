Motivation
==========

Why do a Differentiable Implementation?
-------------------------------

nyx is a follow-up of the `nsb2 <https://github.com/GerritRo/nsb2>`_ package, 
differing mainly in its differentiability and faster computation, as well as optimization features.
This new implementation has a variety of advantages:

**Gradient-based optimization**
   Rather than exhaustive grid searches or stochastic sampling, we can use
   efficient gradient descent to find simulation parameters. This is
   particularly powerful for high-dimensional problems like determining flatfielding
   coefficients or fitting 100s of pointing offsets at the same time.

**Fast Observation Planning/Simulation**
   The GPU based implementation means computation speed is 100x faster than `nsb2 <https://github.com/GerritRo/nsb2>`_,
   meaning that complicated observation campaigns or simulations for trialing optical calibration methods can
   be computed in minutes, instead of hours.

**Optical Astronomy**
   The better inclusion of different emitter types and direct optimization support allows nyx to be used
   for tasks in optical astronomy with IACTs, such as fitting novae brightness or determining satellite tracks while
   accounting for nuisance parameters.

Why JAX?
--------

IACTrace is built on `JAX <https://github.com/jax-ml/jax>`_, a numerical
computing library that provides:

**Automatic differentiation**
   JAX can differentiate through arbitrary Python/NumPy code, including loops,
   conditionals, and custom data structures. This eliminates the need to
   manually derive and implement gradient formulas.

**Hardware acceleration**
   The same code runs on CPU, GPU, or TPU with minimal modification. This enables
   efficient parallel simulations of skyfields, reducing the time requirement for 
   simulating many sources at once.