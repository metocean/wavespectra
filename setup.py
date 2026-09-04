import os

# NumPy 1.23 imports distutils.msvccompiler, absent from Setuptools' bundled
# distutils. Use the stdlib implementation on Python 3.11 and earlier; Python
# 3.12+ removed stdlib distutils and requires updating the build tooling.
os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration


def configuration(parent_package="", top_path=None):
    config = Configuration("", parent_package, top_path)
    config.add_extension(
        "wavespectra.specpart",
        sources=[
            "wavespectra/specpart/specpart.pyf",
            "wavespectra/specpart/specpart.f90",
        ],
    )
    config.add_data_files(
        "LICENSE.txt",
        "wavespectra/core/attributes.yml",
        "wavespectra/output/ww3.yml",
    )
    return config


setup(configuration=configuration)
