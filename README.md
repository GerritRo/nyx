# nyx

![Python](https://img.shields.io/badge/python-3.12-blue)
[![License](https://img.shields.io/badge/license-BSD--3--Clause-green)](licenses/LICENSE.rst)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Differentiable Simulation of Night Sky Background in Imaging Atmospheric Cherenkov Telescopes**

nyx models the night sky background for IACT telescopes, enabling parameter estimation through HMC or gradient-based optimization.

## Features

- Fully differentiable computation for gradient-based optimization and Bayesian inference using JAX
- Zodiacal light, airglow, lunar brightness, and stellar catalogs are supported as emitters
- Effective aperture instrument models with spectral response

## Installation

```bash
pip install git+https://github.com/GerritRo/nyx/
```

For development:
```bash
git clone https://github.com/GerritRo/nyx.git
cd nyx
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Documentation

Full documentation and examples are available at: **https://gerritro.github.io/nyx/**

### Building Documentation Locally

```bash
pip install -e ".[docs]"
cd docs && make html
```

## License

BSD-3-Clause License - see [LICENSE](licenses/LICENSE.rst) for details.

## Citation

If you use nyx in your research, please cite this repository.

## Data Attribution

nyx bundles and downloads scientific datasets produced by third parties
(Pickles 1998, Leinert et al. 1998, the ROLO lunar model, ESO SkyCalc,
Gaia DR3, STScI CALSPEC, and the SVO Filter Profile Service). These datasets
carry their own citation and acknowledgement requirements. If you publish 
results obtained with nyx, cite the datasets you used as described in
[licenses/DATA.rst](licenses/DATA.rst).