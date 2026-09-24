import os
import warnings

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
        If the data directory is missing.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(here, "data")
    if not os.path.isdir(candidate):
        raise FileNotFoundError(
            f"nyx cannot find its bundled data directory; it is expected at {candidate}"
        )
    return candidate + os.sep


ASSETS_PATH = _assets_path()


class NyxWarning(UserWarning):
    """Category for every warning nyx raises about a model or a fit."""


def configure(*, matmul_precision: str = "highest") -> None:
    """Apply nyx's recommended JAX settings.

    Parameters
    ----------
    matmul_precision : str
        Value for ``jax_default_matmul_precision``.
    """
    jax.config.update("jax_default_matmul_precision", matmul_precision)


_PRECISION_WARNED = False


def _warn_unless_highest_precision(stacklevel: int = 3) -> None:
    """Warn once per process if matmuls will run below full float32 precision.

    Parameters
    ----------
    stacklevel : int
        Passed through to :func:`warnings.warn`.
    """
    global _PRECISION_WARNED
    if _PRECISION_WARNED:
        return
    _PRECISION_WARNED = True

    current = jax.config.jax_default_matmul_precision
    try:
        resolved = None if current is None else jax.lax.Precision(current)
    except (TypeError, ValueError):  # pragma: no cover - unknown future value
        resolved = None
    if resolved is jax.lax.Precision.HIGHEST:
        return

    warnings.warn(
        f"jax_default_matmul_precision is {current!r}, not 'highest'. The render "
        "integrates over wavelength and sky position, so on GPU and TPU the "
        "lower-precision matmuls cost accuracy. Call "
        "nyx.configure() to set it, or set it yourself to silence this.",
        NyxWarning,
        stacklevel=stacklevel,
    )


_EXPORTS = {
    # model
    "Scene": "nyx.core",
    "Geometry": "nyx.core",
    "Observation": "nyx.core",
    "Parameter": "nyx.core",
    # parameters
    "dump_params": "nyx.core",
    "freeze": "nyx.core",
    "freeze_all": "nyx.core",
    "parameters_table": "nyx.core",
    "set_parameters": "nyx.core",
    "unfreeze": "nyx.core",
    "unfreeze_all": "nyx.core",
    # emitters
    "Airglow": "nyx.emitter",
    "BrightStars": "nyx.emitter",
    "Moon": "nyx.emitter",
    "PointSource": "nyx.emitter",
    "Stars": "nyx.emitter",
    "ZodiacalLight": "nyx.emitter",
    # atmosphere and instruments
    "SingleScattering": "nyx.atmosphere",
    "EffectiveApertureInstrument": "nyx.instrument",
    "EffectiveApertureTable": "nyx.instrument",
    "load_aperture_table": "nyx.instrument",
    # fitting
    "MultiTargetFit": "nyx.infer",
    "Optimizer": "nyx.infer",
    "parameter_correlation": "nyx.infer",
    "parameter_errors": "nyx.infer",
    "rescale_from_errors": "nyx.infer",
}

__all__ = ["ASSETS_PATH", "NyxWarning", "__version__", "configure", *sorted(_EXPORTS)]


def __getattr__(name: str) -> object:
    """Resolve a top-level export from the subpackage that defines it."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'nyx' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # resolve once
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
