"""Migrate a nyx instrument file to format 2.0.

Format 1.x stored the focal-plane geometry as an explicit
``(n_pixels, 2, grid_dim)`` table of sample coordinates.  Those coordinates
are windows onto a single lattice, but float32 storage hides that: the
samples miss the best-fit lattice by ~1e-5 of a step, so nyx had to refit
and snap them at every load.  Format 2.0 stores the lattice itself -- an
origin, a step, and one integer node offset per pixel -- which is exact,
smaller, and loads without a fit.

Everything else (the response tables, the bandpass, any metadata) is copied
across unchanged, so the migration is lossless apart from replacing the
coordinate table with the lattice it encodes.

Usage::

    python scripts/migrate_instrument.py old.h5 new.h5
    python scripts/migrate_instrument.py old.h5 new.h5 --in-place-backup
"""

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nyx.instrument._interpolation import LATTICE_TOL, PixelLattice  # noqa: E402
from nyx.instrument.io import FORMAT_VERSION  # noqa: E402


def migrate(src, dst, tol: float = LATTICE_TOL) -> dict:
    """Rewrite the instrument file *src* to *dst* in format 2.0.

    Returns a small report: the fitted lattice parameters and the residual
    by which the stored coordinates missed the lattice.
    """
    src, dst = Path(src), Path(dst)
    with h5py.File(src, "r") as f:
        version = str(f.attrs.get("nyx_instrument_version", ""))
        if not version:
            raise ValueError(f"{src} is not a nyx instrument file")
        if version.startswith("2."):
            raise ValueError(f"{src} is already format {version}")
        grid = f["grid"][:]
        values = f["values"][:]
        lattice = PixelLattice.from_grid(grid, values, tol=tol)

        rebuilt = np.asarray(lattice.grid, dtype=np.float64)
        step = np.asarray(lattice.step, dtype=np.float64)
        residual = float(np.max(np.abs(rebuilt - grid) / step[None, :, None]))

        dst.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dst, "w") as out:
            for key, value in f.attrs.items():
                out.attrs[key] = value
            out.attrs["nyx_instrument_version"] = FORMAT_VERSION
            for name in f:
                if name != "grid":  # replaced by the lattice
                    f.copy(name, out)
            lat = out.create_group("lattice")
            lat.create_dataset("origin", data=np.asarray(lattice.origin, dtype=np.float64))
            lat.create_dataset("step", data=step)
            lat.create_dataset("offset", data=np.asarray(lattice.offset, dtype=np.int32))

    return {
        "origin": np.asarray(lattice.origin, dtype=np.float64),
        "step": step,
        "n_pixels": lattice.n_pixels,
        "grid_shape": lattice.grid_shape,
        "lattice_shape": lattice.shape,
        "residual_steps": residual,
        "from_version": version,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("src", help="instrument file in format 1.x")
    parser.add_argument("dst", help="output file in format 2.0")
    parser.add_argument(
        "--in-place-backup",
        action="store_true",
        help="after writing dst, move it over src, keeping src as src.bak",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=LATTICE_TOL,
        help="lattice alignment tolerance, in units of one response-grid step",
    )
    args = parser.parse_args(argv)

    report = migrate(args.src, args.dst, tol=args.tol)
    print(f"{args.src} (format {report['from_version']}) -> {args.dst} (format {FORMAT_VERSION})")
    print(f"  pixels          {report['n_pixels']}")
    print(f"  response grid   {report['grid_shape'][0]} x {report['grid_shape'][1]}")
    print(f"  lattice         {report['lattice_shape'][0]} x {report['lattice_shape'][1]} nodes")
    print(f"  origin [rad]    {report['origin'][0]:.12e}, {report['origin'][1]:.12e}")
    print(f"  step   [rad]    {report['step'][0]:.12e}, {report['step'][1]:.12e}")
    print(f"  coordinates reproduced to {report['residual_steps']:.2e} of a step")

    if args.in_place_backup:
        backup = Path(args.src).with_suffix(Path(args.src).suffix + ".bak")
        shutil.move(args.src, backup)
        shutil.move(args.dst, args.src)
        print(f"  moved into place; original kept at {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
