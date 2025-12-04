# SWOT Balanced Extraction

This repository provides the workflow for processing, analyzing, and performing the balanced extraction on data from NASA’s Surface Water and Ocean Topography (SWOT) mission. 
It includes tools for KaRIn and Nadir data processing, spectral model fitting, Gaussian-process–based balanced SSH extraction, synthetic SWOT data generation, and diagnostics for mesoscale and submesoscale ocean dynamics.

The frontend of the module is in Python and the transforms used for the covariance matrices are performed in Julia for faster processing. 
The Julia trasforms are integrated using the ```/src/JWS_SWOT_toolbox/julia_bridge.py``` which requires the ```juliacall``` package be installed and the ```FFTW, Interpolations, LinearAlgebra, ProgressMeter``` packages installed in the connected Julia environment. 

---

## Installation
### Python Package Installation
Install the main Python package from the root of the repository: 
```bash
pip install -e .
```

The core functionality resides in the Python module:

```
src/jws_swot_toolbox/
```

### Julia Installation

For optimal performance, the covariance matrix transforms are executed in Julia. This requires a specific setup to integrate Julia with the Python environment.

1.  **Install Julia.**
2.  **Install Python Bridge:** The integration is handled by the `juliacall` Python package, which must be installed.
3.  **Install Julia Packages:** The following Julia packages are required in the connected Julia environment:
    * `FFTW`
    * `Interpolations`
    * `LinearAlgebra`
    * `ProgressMeter`

The Python-to-Julia bridge is configured in:

```
/src/jws_swot_tools/julia_bridge.py
```

and the transform functions are called from: 

```
/src/jws_swot_tools/transforms_julia.jl
```

---

## Paper and Citation

A full description of the methodology is available in the following paper:

> Skinner, J. W., Callies, J., Lawrence, A., and Zhang, X. (2025). **Isolating Balanced Ocean Dynamics in SWOT Data**. *Submitted to JGR Oceans.*
> [arXiv:2512.03258](https://arxiv.org/abs/2512.03258)

The scripts used to create the figures in the paper are in the folder ```balanced_extraction_paper/```. 

