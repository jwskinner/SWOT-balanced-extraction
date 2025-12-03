# SWOT Balanced Extraction

This repository provides the workflow for processing, analyzing, and performing the balanced extraction on data from NASA’s Surface Water and Ocean Topography (SWOT) mission. 
It includes tools for KaRIn and Nadir data processing, spectral model fitting, Gaussian-process–based balanced SSH extraction, synthetic SWOT data generation, and diagnostics for mesoscale and submesoscale ocean dynamics.

For a full description of the code: 

The frontend of the module is in Python and the transforms used for the covariance matrices are performed in Julia for faster processing. 
The Julia trasforms are integrated using the ```/src/JWS_SWOT_toolbox/julia_bridge.py``` which requires the ```juliacall``` package be installed and the ```FFTW, Interpolations, LinearAlgebra, Printf, ProgressMeter``` packages installed in the connected Julia environment. 

The core functionality is in the Python module:

```
src/JWS_SWOT_toolbox/
```

and the scripts used to create the figures in the above paper are in ```balanced_extraction_paper/```


---


## Installation

```bash
pip install -e .
```

---

## Paper

A full description of the method is in:

Skinner, J. W., Callies, J., Lawrence, A., and Zhang, X., Isolating Balanced Ocean Dynamics
in SWOT Data (submitted to JGR Oceans 2025).

