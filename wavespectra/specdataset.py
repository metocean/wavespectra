"""Wrapper around the xarray dataset."""
import os
import re
import sys
import types
import warnings

import six
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.specarray import SpecArray

here = os.path.dirname(os.path.abspath(__file__))


class Plugin(type):
    """Add all the export functions at class creation time."""

    def __new__(cls, name, bases, dct):
        modules = [
            __import__(
                "wavespectra.output.{}".format(os.path.splitext(fname)[0]),
                fromlist=["*"],
            )
            for fname in os.listdir(os.path.join(here, "output"))
            if fname.endswith(".py")
        ]
        for module in modules:
            for module_attr in dir(module):
                function = getattr(module, module_attr)
                if isinstance(function, types.FunctionType) and module_attr.startswith(
                    "to_"
                ):
                    dct[function.__name__] = function
        return type.__new__(cls, name, bases, dct)


@xr.register_dataset_accessor("spec")
@six.add_metaclass(Plugin)
class SpecDataset(object):
    """Wrapper around the xarray dataset.
    
    Plugin functions defined in wavespectra/output/<module>
    are attached as methods in this accessor class.

    """

    def __init__(self, xarray_dset):
        self.dset = xarray_dset
        self._wrapper()
        self._load_defaults()
        self.supported_dims = [
            attrs.TIMENAME,
            attrs.SITENAME,
            attrs.LATNAME,
            attrs.LONNAME,
            attrs.FREQNAME,
            attrs.DIRNAME,
        ]

    def __getattr__(self, attr):
        return getattr(self.dset, attr)

    def __repr__(self):
        return re.sub(r"<.+>", "<{}>".format(self.__class__.__name__), str(self.dset))

    def _wrapper(self):
        """Wraper around SpecArray methods.

        Allows calling public SpecArray methods from SpecDataset.
        For example:
            self.spec.hs() becomes equivalent to self.efth.spec.hs()

        """
        for method_name in dir(self.dset[attrs.SPECNAME].spec):
            if not method_name.startswith("_"):
                method = getattr(self.dset[attrs.SPECNAME].spec, method_name)
                setattr(self, method_name, method)

    def _load_defaults(self):
        """Load wind and depth values as defaults for the partition method.
        Allows runnig ds.spec.partition() directly or with keyword args.
        """
        try:
            assert self.partition.__code__.co_varnames[1:4] == (
                "wsp_darr",
                "wdir_darr",
                "dep_darr",
            )
            self.partition.__func__.__defaults__ = (
                self.dset[attrs["WSPDNAME"]],
                self.dset[attrs["WDIRNAME"]],
                self.dset[attrs["DEPNAME"]],
            ) + self.partition.__func__.__defaults__[3:]
        except:
            warnings.warn("Cannot load defaults for partition algorithm")

    def _check_and_stack_dims(self):
        """Ensure dimensions are suitable for dumping in some ascii formats.

        Returns:
            Dataset object with site dimension and with no grid dimensions

        Note:
            Grid is converted to site dimension which can be iterated over
            Site is defined if not in dataset and not a grid
            Dimensions are checked to ensure they are supported for dumping
        """
        dset = self.dset.load().copy(deep=True)

        unsupported_dims = set(dset[attrs.SPECNAME].dims) - set(self.supported_dims)
        if unsupported_dims:
            raise NotImplementedError(
                "Dimensions {} are not supported by {} method".format(
                    unsupported_dims, sys._getframe().f_back.f_code.co_name
                )
            )

        # If grid reshape into site, if neither define fake site dimension
        if set(("lon", "lat")).issubset(dset.dims):
            dset = dset.stack(site=("lat", "lon"))
        elif "site" not in dset.dims:
            dset = dset.expand_dims("site")

        return dset

    def sel_lonlat(self, lon, lat, method=None):
        """Select site based on longitude and latitude.

        Args:
            - lon (float): longitude of the site.
            - lat (float): latitude of the site.
            - method (string): Method to use for inexact matches (None or 'nearest').

        Returns:
            - Dataset for the site defined by lon and lat.
        """
        if method not in (None, "nearest"):
            raise ValueError(
                "Invalid method. Expecting None or nearest. Got {}".format(method)
            )
        lons = self.dset[attrs.LONNAME].values
        lats = self.dset[attrs.LATNAME].values
        xdist0 = abs(lons % 360 - lon % 360)
        xdist = xr.ufuncs.minimum(xdist0, 360 - xdist0)
        ydist = abs(lats - lat)
        dist2 = xdist ** 2 + ydist ** 2
        isite = [int(dist2.argmin())]
        if (method is None) and (dist2[isite] > 0):
            raise ValueError(
                "lon={:f}, lat={:f} not found. Use method='nearest' to get lon={:f}, lat={:f}".format(
                    lon, lat, lons[isite][0], lats[isite][0]
                )
            )
        indexersdict = {
            k: isite
            for k in {attrs.LONNAME, attrs.LATNAME, attrs.SITENAME}.intersection(
                self.dset.dims
            )
        }
        return self.dset.isel(indexersdict)


if __name__ == "__main__":
    from wavespectra.input.swan import read_swan

    here = os.path.dirname(os.path.abspath(__file__))
    ds = read_swan(os.path.join(here, "../tests/sample_files/swanfile.spec"))
    # ds.spec.to_octopus('/tmp/test.oct')
    # ds.spec.to_swan('/tmp/test.swn')
    # ds.spec.to_netcdf('/tmp/test.nc')
