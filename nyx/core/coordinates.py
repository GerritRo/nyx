from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

__all__ = [
    "altaz_to_offset",
    "cos_angular_separation_jax",
    "offset_to_altaz",
    "rotation_matrix_from_altaz",
    "safe_arcsin",
]


def rotation_matrix_from_altaz(az_rad: float, alt_rad: float) -> np.ndarray:
    """Build rotation matrix from an AltAz pointing direction.

    Parameters
    ----------
    az_rad, alt_rad : float
        Pointing direction in radians.

    Returns
    -------
    rotation : ndarray, shape (3, 3)
    """
    ca, sa = np.cos(az_rad), np.sin(az_rad)
    cd, sd = np.cos(alt_rad), np.sin(alt_rad)
    return np.array(
        [
            [cd * sa, cd * ca, sd],
            [ca, -sa, 0.0],
            [-sd * sa, -sd * ca, cd],
        ]
    )


def safe_arcsin(z: jax.Array) -> jax.Array:
    """``arcsin`` of a direction cosine, differentiable at ``|z| >= 1``.

    Parameters
    ----------
    z : jax.Array

    Returns
    -------
    jax.Array
    """
    at_pole = jnp.abs(z) >= 1.0
    safe_z = jnp.where(at_pole, 0.0, z)
    return jnp.where(at_pole, jnp.sign(z) * (jnp.pi / 2), jnp.arcsin(safe_z))


def altaz_to_offset(
    az: ArrayLike, alt: ArrayLike, rotation: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """Transform AltAz (az, alt) to offset frame (lon, lat) using *rotation*.

    Parameters
    ----------
    az, alt : array-like
        AltAz coordinates in radians.
    rotation : array-like, shape (3, 3)
        Rotation matrix from :func:`rotation_matrix_from_altaz`.

    Returns
    -------
    lon, lat : jax.Array
        Offset frame coordinates in radians.
    """
    p = jnp.stack(
        [
            jnp.cos(alt) * jnp.sin(az),
            jnp.cos(alt) * jnp.cos(az),
            jnp.sin(alt),
        ],
        axis=-1,
    )
    rotation = jnp.asarray(rotation)
    p_local = jnp.einsum("ij,...j->...i", rotation, p)
    lon = jnp.arctan2(p_local[..., 1], p_local[..., 0])
    lat = safe_arcsin(p_local[..., 2])
    return lon, lat


def offset_to_altaz(
    lon: ArrayLike, lat: ArrayLike, rotation: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    """Transform offset frame (lon, lat) to AltAz; inverse of :func:`altaz_to_offset`.

    Parameters
    ----------
    lon, lat : array-like
        Offset frame coordinates in radians.
    rotation : array-like, shape (3, 3)
        Rotation matrix from :func:`rotation_matrix_from_altaz`.

    Returns
    -------
    az, alt : jax.Array
        AltAz coordinates in radians.
    """
    p_local = jnp.stack(
        [
            jnp.cos(lat) * jnp.cos(lon),
            jnp.cos(lat) * jnp.sin(lon),
            jnp.sin(lat),
        ],
        axis=-1,
    )
    rotation = jnp.asarray(rotation)
    p = jnp.einsum("ij,...j->...i", rotation.T, p_local)

    eps = jnp.finfo(p.dtype).eps
    horiz_sq = p[..., 0] ** 2 + p[..., 1] ** 2
    at_pole = horiz_sq < 4 * eps

    safe_pz = jnp.where(at_pole, 0.0, p[..., 2])
    alt = jnp.where(
        at_pole,
        jnp.sign(p[..., 2]) * (jnp.pi / 2),
        jnp.arcsin(safe_pz),
    )

    safe_px = jnp.where(at_pole, 0.0, p[..., 0])
    safe_py = jnp.where(at_pole, 1.0, p[..., 1])
    az = jnp.where(at_pole, 0.0, jnp.arctan2(safe_px, safe_py))

    return az, alt


def cos_angular_separation_jax(
    az1: jax.Array, alt1: jax.Array, az2: jax.Array, alt2: jax.Array
) -> jax.Array:
    """JAX-compatible cosine of great-circle angular separation.

    Parameters
    ----------
    az1, alt1, az2, alt2 : jax.Array
        Coordinates in radians.

    Returns
    -------
    jax.Array
    """
    return jnp.sin(alt1) * jnp.sin(alt2) + jnp.cos(alt1) * jnp.cos(alt2) * jnp.cos(az1 - az2)
