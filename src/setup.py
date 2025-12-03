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
        "scipy",
        "matplotlib",
        "netCDF4",
        "xrft",
        "pandas",
        "earthaccess",
        "h5py",
        "pyproj",
        "cmocean",
        "cartopy",
        "juliacall",
        "ipython",
    ],
)