Installation
============

Requirements
------------

nyx requires **Python 3.12 or later**.

Dependencies
~~~~~~~~~~~~

Core dependencies are installed automatically:

- numpy, scipy, h5py
- astropy, healpy, jax-healpy
- jax, equinox, chex, optimistix
- dust-extinction

Quick Install
-------------

The fastest way to get nyx is to install the latest version directly from
GitHub:

.. code-block:: bash

   pip install git+https://github.com/GerritRo/nyx/

Installation from Source
------------------------

For local development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/GerritRo/nyx.git
   cd nyx
   python -m venv .venv && source .venv/bin/activate
   pip install -e .

Optional Dependencies
---------------------

For development (tests, linting, type checking):

.. code-block:: bash

   pip install -e ".[dev]"

For building the documentation:

.. code-block:: bash

   pip install -e ".[docs]"

JAX Backend (CPU / GPU)
-----------------------

nyx is built on `JAX <https://jax.dev>`_. By default ``pip`` installs the
**CPU** build, which is enough to get started. For GPU acceleration install a
matching CUDA wheel *after* installing nyx, for example:

.. code-block:: bash

   pip install -U "jax[cuda12]"

See the `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
for the wheel that matches your CUDA/driver versions.

.. note::

   nyx sets ``jax_default_matmul_precision="highest"`` on import and the
   instrument model runs in single precision (float32). This keeps results
   consistent across CPU and GPU backends.

Scientific Datasets
-------------------

Several emitters rely on third-party scientific datasets. Some are bundled
with the package, while others (such as the Gaia DR3 catalog used by
:class:`~nyx.emitter.Stars`) are **downloaded on first use** and cached by
Astropy (``~/.astropy/cache`` by default). This means:

- The first run of a model that needs a remote dataset requires network
  access; subsequent runs use the cache and work offline.
- These datasets carry their own citation and acknowledgement requirements.
  If you publish results, please cite the datasets you used — see
  :doc:`../about/data` for details.

Verifying Installation
----------------------

To verify the installation, open a Python shell and import the package:

.. code-block:: python

   import nyx
   print(nyx.__version__)

Troubleshooting
---------------

- **GPU not detected.** Verify ``import jax; jax.devices()`` lists a GPU. If
  not, the CUDA wheel does not match your driver — reinstall the correct
  ``jax[cuda...]`` build.
- **healpy / jax-healpy build errors.** These packages require a working C/C++
  toolchain. On most platforms prebuilt wheels are available for recent Python
  versions; make sure you are on Python 3.12+.
- **Remote dataset download fails.** Check network access and the Astropy
  cache directory. Once a dataset is cached, the model runs offline.

