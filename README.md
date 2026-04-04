# ravex

**RAdial VElocity eXplorer**

`ravex` is a Python package for simulating and analyzing radial-velocity (RV) observations of planetary systems.

It is designed for research-oriented RV workflows, including synthetic multi-planet time-series generation, signal injection and recovery, mass-precision forecasting, and detectability studies in period–mass space.

---

## Overview

`ravex` provides a compact framework to:

- simulate Keplerian RV signals for one or more planets,
- generate synthetic observing cadences,
- inject realistic noise into RV measurements,
- estimate RV precision for specific instruments,
- evaluate signal recovery with generalized Lomb–Scargle periodograms,
- track detection significance as a function of the number of observations,
- build injection–recovery detectability maps,
- accelerate large detectability grids through multiprocessing.

The package is currently organized around the `MultiPlanetSystem` class and a small set of supporting utility functions.

---

## Current features

### RV simulation
- Multi-planet Keplerian RV simulations
- Support for specifying planets through orbital period or semi-major axis
- Optional BJD conversion from JD when observatory location and target coordinates are provided
- True-anomaly computation and RV model evaluation
- Per-planet phased RV views for visualization and fitting

### Synthetic observations
- Random observing-date generation over a user-defined time span
- Gaussian RV noise injection
- Flexible handling of scalar or per-point RV uncertainties

### Recovery and detectability analysis
- GLS-based periodic signal recovery
- FAP-to-sigma conversion helpers
- Bootstrap-based FAP estimation
- Detection-growth curves
- Detectability tracking as a function of the number of observations
- Injection–recovery detectability maps in period vs. minimum-mass space
- Parallelized detectability-map computation

### Fitting and forecasting
- Mass-precision forecasting through repeated synthetic campaigns
- Recovery of sinusoidal amplitudes from time series
- Model interpolation in orbital phase for fitting workflows

### Instrumental utilities
- Empirical CARMENES VIS RV error estimator
- Empirical MAROON-X SERVAL RV error estimator

### Plotting and persistence
- Detection-growth plotting utilities
- Detectability-map plotting utilities
- CSV save/load helpers for `precision_tracker` outputs

---

## Installation

From the repository root:

```bash
/usr/bin/python3 -m pip install -e .


```

This installs `ravex` in editable mode, which is convenient during active development.

---

## Dependencies

The current version depends on:

- `numpy`
- `scipy`
- `astropy`
- `matplotlib`
- `pandas`

These dependencies are declared in `pyproject.toml`.

---

## Package layout

```text
ravex/
├── pyproject.toml
├── README.md
└── src/
    └── ravex/
        ├── __init__.py
        └── core.py
```

---

## Public API

At the moment, the main public entry points are:

```python
from ravex import MultiPlanetSystem
from ravex import carm_error
from ravex import maroonx_serval_error
from ravex import plot_detection_growth_strict
from ravex import plot_detectability_map
from ravex import save_precision_tracker_to_csv
from ravex import load_precision_tracker_from_csv
```

Advanced or internal helpers remain available through `ravex.core` if needed.

---

## Notes

- `ravex` is currently focused on research use and active development, not on polished end-user packaging.
- Some APIs may still evolve as the project gains examples, tests, and more documentation.
- The present implementation is best suited for synthetic RV studies, detectability forecasts, and injection–recovery experiments.

---

## Contributing

Contributions, suggestions, and issue reports are welcome.

During this early stage, the most useful contributions are likely to be:

- bug reports,
- API feedback,
- documentation improvements,
- example notebooks,
- validation against benchmark RV use cases.

---

## Citation

Citation metadata and a software DOI will be added in a future release.

The planned workflow is to archive versioned GitHub releases through Zenodo so that ravex can be cited through a DOI.

---

## License

This project is distributed under the BSD 3-Clause License.

A LICENSE file is included in the root of the repository.


---
## Authors

- **Lead developer:** Francisco J. Pozuelos (IAA-CSIC)
- **Contributor:** Roberto Varas (IAA-CSIC)

---

## AI-assisted development

Parts of the development of `ravex` were carried out with the assistance of **ChatGPT (GPT-5.4 Thinking)**.

AI-assisted contributions included, among others:

- code review,
- debugging,
- refactoring suggestions,
- optimization of computational routines,
- documentation drafting and polishing,
- package-structure and repository organization support.

The scientific design, methodological choices, validation, and final decisions regarding the code and its use remain the responsibility of the lead developer.

---

## Status

Current development stage: **v0.1.0**

