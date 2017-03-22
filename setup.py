"""
numpy.distutils.core.setup seems to handle subpackages better than
setuptools.setup
"""
from setuptools import setup, find_packages

install_requires = [
    'numpy',
    'xarray>=0.9',
    'matplotlib',
    'dask',
    'toolz',
    'cloudpickle',
    ]

setup(name='pyspectra',
      version='1.1.0',
      description='Spectra base class and tools based on DataArray',
      author='MetOcean Solutions Ltd.',
      author_email='r.guedes@metocean.co.nz',
      install_requires=install_requires,
      url='http://www.metocean.co.nz/',
      packages=['pyspectra'],
      )

