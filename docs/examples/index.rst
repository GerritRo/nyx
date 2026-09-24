Example Gallery
===============

Jupyter notebooks demonstrating nyx capabilities.

.. note::

   Run :doc:`Example` first. It ray-traces a telescope with
   `iactrace <https://github.com/GerritRo/iactrace>`_ and writes
   ``CT1_aperture.npz``; the rest load that table rather than repeating the
   scan. That first notebook therefore needs ``pip install "nyx[iactrace]"``
   and an iactrace telescope configuration; the others need neither.

.. toctree::
   :maxdepth: 1

   Example
   GlobalPointingFit
   MirrorMisalignment
   MultiTargetFlatfieldFit
   ProfileScan
   RecoveringLightcurves
   PosteriorEstimation