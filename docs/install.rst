.. image:: _static/MO_Horiz_Primary_rgb.png
   :width: 150 px
   :align: right

============
Installation
============

Requirements
------------

wavespectra supports Python 3.8, 3.9, and 3.10. Python 3.9 is recommended
for this legacy build because spectral partitioning uses ``numpy.distutils``
and a compiled Fortran extension.

A Fortran compiler is required on Linux. On Debian or Ubuntu:

.. code:: bash

   sudo apt install gcc gfortran

Install from sources
--------------------
The source code is hosted at Github_:

.. code:: bash

   git clone https://github.com/metocean/wavespectra.git
   cd wavespectra

Create and activate a Python 3.9 environment:

.. code:: bash

   uv venv --python 3.9
   source .venv/bin/activate

For a non-editable installation with all optional dependencies:

.. code:: bash

   uv sync --no-editable --all-extras

The ``metocean`` extra installs ``cfjson`` from GitHub over HTTPS. To omit that
dependency, install the public optional and test dependencies instead:

.. code:: bash

   uv sync --no-editable --extra extra --extra test

Run the tests:

.. code:: bash

   pytest

For an editable installation, first install the legacy build tools:

.. code:: bash

   uv pip install "setuptools<65" wheel "numpy>=1.23.5,<2.0" "pip==22.3.1"

   SETUPTOOLS_ENABLE_FEATURES="legacy-editable" .venv/bin/pip install -e ".[extra,test,metocean]" --no-build-isolation --disable-pip-version-check

Running tests
-------------

.. code:: bash

   pytest -v

   pytest -v tests/core
   pytest -v tests/core/test_wave_stats.py
   pytest -v tests/core/test_wave_stats.py::TestSpecArray

.. _Github: https://github.com/metocean/wavespectra
.. _development mode: https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs
