import os

import jax

__version__ = "0.1.0"


def _assets_path() -> str:
    """Directory holding the bundled data files, with a trailing separator.

    Returns
    -------
    str

    Raises
    ------
    FileNotFoundError
        If the data directory is in neither location.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (
        os.path.join(os.path.dirname(here), "data"),  # repository root
        os.path.join(here, "data"),  # installed wheel
    ):
        if os.path.isdir(candidate):
            return candidate + os.sep
    raise FileNotFoundError(
        "nyx cannot find its bundled data directory; it is expected at "
        f"{os.path.join(os.path.dirname(here), 'data')} in a source checkout"
    )


#: Directory of the bundled scientific datasets, with a trailing separator.
ASSETS_PATH = _assets_path()


class NyxWarning(UserWarning):
    """Category for every warning nyx raises about a model or a fit."""


jax.config.update("jax_default_matmul_precision", "highest")
