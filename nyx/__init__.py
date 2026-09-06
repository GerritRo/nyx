import os

import jax

__version__ = "0.1.0"


def _assets_path() -> str:
    """Directory holding the bundled data files, with a trailing separator.

    The files live in ``data/`` at the repository root, which is where a
    source checkout finds them.  An installed wheel has no repository
    root, so setuptools maps that same directory into the package as
    ``nyx/data`` (see ``[tool.setuptools.package-dir]``); that is the
    second place looked at.
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


#: Directory of the bundled scientific datasets, with a trailing separator,
#: so ``ASSETS_PATH + 'leinert1998_zodiacal_light.dat'`` is a full path.
ASSETS_PATH = _assets_path()


class NyxWarning(UserWarning):
    """Category for every warning nyx raises about a model or a fit.

    A :class:`UserWarning`, so it shows by default, but with its own
    category so it can be filtered or promoted apart from third-party
    noise::

        warnings.filterwarnings("error", category=nyx.NyxWarning)
    """


jax.config.update("jax_default_matmul_precision", "highest")
