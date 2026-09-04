wavespectra
===========
Python library for ocean wave spectra.

Main contents:
--------------
- SpecArray_: extends xarray's `DataArray`_ with methods to manipulate wave spectra and calculate spectral statistics.
- SpecDataset_: wrapper around `SpecArray`_ with methods for saving spectra in different formats.

Documentation:
--------------
The documentation is hosted on ReadTheDocs at https://wavespectra-metocean.readthedocs.io.

Install:
--------
The source code is hosted on GitHub at https://github.com/metocean/wavespectra.

Install from sources
~~~~~~~~~~~~~~~~~~~~
Clone the repository and navigate to its root directory:

.. code:: bash

   git clone https://github.com/metocean/wavespectra.git
   cd wavespectra

Create a Python 3.9 environment:

.. code:: bash

   uv venv --python 3.9
   source .venv/bin/activate

Install the Fortran compiler required to build spectral partitioning. On Debian
or Ubuntu:

.. code:: bash

   sudo apt install gcc gfortran

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

Run the tests with ``pytest`` after an editable installation.

Windows installation
~~~~~~~~~~~~~~~~~~~~
Currently there are some issues with the Fortran version of spectral partitioning in Windows systems. A workaround is described in:
https://github.com/metocean/wavespectra/issues/28#issuecomment-712503611

Code structure:
---------------
The two main classes SpecArray_ and SpecDataset_ are defined as `xarray accessors`_. The accessors are registered on xarray's DataArray_ and Dataset_ respectively as a new namespace called `spec`.

To use methods in the accessor classes simply import the classes into your code and they will be available to your xarray.Dataset or xarray.DataArray instances through the `spec` attribute, e.g.

.. code:: ipython

   import datetime
   import numpy as np
   import xarray as xr

   from wavespectra.specarray import SpecArray
   from wavespectra.specdataset import SpecDataset

   coords = {'time': [datetime.datetime(2017,01,n+1) for n in range(2)],
             'freq': [0.05,0.1],
             'dir': np.arange(0,360,120)}
   efth = xr.DataArray(data=np.random.rand(2,2,3),
                       coords=coords,
                       dims=('time','freq', 'dir'),
                       name='efth')

   In [1]: efth
   Out[1]:
   <xarray.DataArray (time: 2, freq: 2, dir: 3)>
   array([[[ 0.100607,  0.328229,  0.332708],
           [ 0.532   ,  0.665938,  0.177731]],

          [[ 0.469371,  0.002963,  0.627179],
           [ 0.004523,  0.682717,  0.09766 ]]])
   Coordinates:
     * freq     (freq) float64 0.05 0.1
     * dir      (dir) int64 0 120 240
     * time     (time) datetime64[ns] 2017-01-01 2017-01-02

   In [2]: efth.spec
   Out[2]:
   <SpecArray (time: 2, freq: 2, dir: 3)>
   array([[[ 0.100607,  0.328229,  0.332708],
           [ 0.532   ,  0.665938,  0.177731]],

          [[ 0.469371,  0.002963,  0.627179],
           [ 0.004523,  0.682717,  0.09766 ]]])
   Coordinates:
     * freq     (freq) float64 0.05 0.1
     * dir      (dir) int64 0 120 240
     * time     (time) datetime64[ns] 2017-01-01 2017-01-02

   In [3]: efth.spec.hs()
   Out[3]:
   <xarray.DataArray 'hs' (time: 2)>
   array([ 10.128485,   9.510618])
   Coordinates:
     * time     (time) datetime64[ns] 2017-01-01 2017-01-02
   Attributes:
       standard_name: sea_surface_wave_significant_height
       units: m

SpecDataset provides a wrapper around the methods in SpecArray. For instance, these produce same result:

.. code:: ipython

   In [4]: dset = efth.to_dataset(name='efth')

   In [5]: tm01 = dset.spec.tm01()

   In [6]: tm01.identical(dset.efth.spec.tm01())
   Out[6]: True

Data requirements:
------------------
SpecArray_ methods require DataArray_ to have the following attributes:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- wave frequency coordinate in :math:`Hz` named as `freq` (required).
- wave direction coordinate in :math:`degree` (coming from) named as `dir` (optional for 1D, required for 2D spectra).
- wave energy density data in :math:`m2/Hz/degree` (2D) or :math:`m2/Hz` (1D) named as `efth`

SpecDataset_ methods require xarray's Dataset_ to have the following attributes:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- spectra DataArray named as `efth`, complying with the above specifications

Examples:
---------

Define and plot spectra history from example SWAN_ spectra file:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from wavespectra import read_swan

   dset = read_swan('/source/wavespectra/tests/sample_files/swanfile.spec')
   spec_hist = dset.isel(lat=0, lon=0).sel(freq=slice(0.05,0.2)).spec.oned().T
   spec_hist.plot.contourf(levels=10)

.. _SpecArray: https://github.com/metocean/wavespectra/blob/master/wavespectra/specarray.py
.. _SpecDataset: https://github.com/metocean/wavespectra/blob/master/wavespectra/specdataset.py
.. _DataArray: http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html
.. _Dataset: http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html
.. _readspec: https://github.com/metocean/wavespectra/blob/master/wavespectra/readspec.py
.. _xarray accessors: http://xarray.pydata.org/en/stable/internals.html?highlight=accessor
.. _SWAN: http://swanmodel.sourceforge.net/online_doc/swanuse/node50.html
.. _Python package index: https://pypi.python.org/pypi/wavespectra
.. _wavespectra: https://github.com/metocean/wavespectra
.. _development mode: https://pip.pypa.io/en/latest/reference/pip_install/#editable-installs
