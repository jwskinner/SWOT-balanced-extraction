from setuptools import setup, find_packages

setup(
    name="JWS_SWOT_toolbox",
    version="0.1",
    description="Tools for SWOT data analysis for balanced extraction paper",
    author="Jack W. Skinner",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "xarray",
        "matplotlib",
        "netCDF4",
        "scipy",
        "xrft",
    ],
)