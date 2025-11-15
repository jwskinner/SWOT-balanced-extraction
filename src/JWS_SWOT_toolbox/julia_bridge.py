import os
from juliacall import Main as julia_functions

_julia_file = os.path.join(os.path.dirname(__file__), "transforms_julia.jl")
if not os.path.exists(_julia_file):
    raise FileNotFoundError(f"Missing {_julia_file}")
julia_functions.include(_julia_file)