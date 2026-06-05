import os

import jax

__version__ = "0.1.0"

ASSETS_PATH = os.path.join(os.path.dirname(__file__), "data/")

jax.config.update("jax_default_matmul_precision", "highest")
