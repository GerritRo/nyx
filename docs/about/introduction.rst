Introduction
============

nyx is a differentiable implementation of night sky simulation for IACTs (Imaging Atmospheric Cherenkov Telescopes).
It is built on JAX and Equinox to support automatic differentiation to efficiently recover observation parameters 
from real observations of the NSB (Night Sky Background) in IACTs.

What is nyx?
-----------------

nyx provides a computational framework for:

- **Night Sky Simulation**: Simulate the expected brightness in photons/s/m^2 at any location on earth.
- **Telescope Response**: Allows a flexible format for instrument characterization, independent of origin. 
Weights night sky simulation with actual telescope response to create realistic NSB images.
- **Fast Rendering**: Due to inbuilt GPU support and just-in-time compilation of JAX, NSB images are rendered in ms.
- **Optimization**: nyx is fully differentiable via JAX's automatic differentiation, enabling gradient based recovery of 
instrument, atmosphere and emitter properties.

Target Audience
---------------

This library is designed for:

- **Calibration specialists**, especially in the context of solving inverse problems on NSB images in IACTs
- **Python extremists** looking for a pip-installable python-native package
- **Speed freaks** dissatisfied with the speed of the alternatives

The library assumes familiarity with Python, NumPy-style array programming, and
basic optics concepts. Experience with JAX is helpful but not required.

License
-------

nyx is released under the BSD-3-Clause license. See the
`LICENSE <https://github.com/GerritRo/nyx/-/blob/main/licenses/LICENSE.rst>`_
file for details.

Citation
--------

If you use IACTrace in your research, please cite::

   @software{iactrace,
     author = {Roellinghoff, Gerrit},
     title = {nyx: Differentiable Simulation of Night Sky Background in Imaging Atmospheric Cherenkov Telescopes},
     url = {https://github.com/GerritRo/nyx},
     year = {2026}
   }