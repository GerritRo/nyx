import os

import jax

__version__ = "0.1.0"

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "data/")


class NyxWarning(UserWarning):
    """Category for every warning nyx raises about a model or a fit.

    A :class:`UserWarning`, so it shows by default, but with its own
    category so it can be filtered or promoted apart from third-party
    noise::

        warnings.filterwarnings("error", category=nyx.NyxWarning)
    """


jax.config.update("jax_default_matmul_precision", "highest")
