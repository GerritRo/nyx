Instrument
==========

Effective Aperture Instruments
------------------------------
.. automodapi:: nyx.instrument.effective_aperture
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx

Reading an iactrace scan
------------------------

An instrument's bandpass and per-pixel response normally come out of an
`iactrace <https://github.com/GerritRo/iactrace>`_ ray trace.  That scan is
hours of Monte-Carlo work, so do it once, save the table, and build from the
file afterwards::

    # once, wherever iactrace is installed
    table = iactrace.analysis.effective_aperture(telescope, camera, ...)
    table.save("CT3_aperture.npz")

    # ever after, with no ray tracer in the environment
    inst = EffectiveApertureInstrument.from_iactrace_table(geo, "CT3_aperture.npz")

The archive is plain numpy, so reading it needs nothing beyond nyx's own
dependencies; the ``nyx[iactrace]`` extra is required only to *run* a scan
through
:meth:`~nyx.instrument.effective_aperture.EffectiveApertureInstrument.from_iactrace`.

.. autofunction:: nyx.instrument.load_aperture_table

.. autoclass:: nyx.instrument.ApertureTable
   :members:

I/O
---
.. automodapi:: nyx.instrument.io
   :no-inheritance-diagram:
   :no-main-docstring:
   :allowed-package-names: nyx
