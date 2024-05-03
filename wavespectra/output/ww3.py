"""Native WAVEWATCH3 output plugin."""
import os

import numpy as np
import xarray as xr
import yaml

# from wavespectra import __version__
from wavespectra.core.attributes import attrs
from wavespectra.core.misc import R2D

MAPPING = {
    "time": attrs.TIMENAME,
    "frequency": attrs.FREQNAME,
    "direction": attrs.DIRNAME,
    "station": attrs.SITENAME,
    "efth": attrs.SPECNAME,
    "longitude": attrs.LONNAME,
    "latitude": attrs.LATNAME,
    "wnddir": attrs.WDIRNAME,
    "wnd": attrs.WSPDNAME,
}

VAR_ATTRIBUTES = yaml.load(
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ww3.yml")),
    Loader=yaml.Loader,
)

TIME_UNITS = VAR_ATTRIBUTES["time"].pop("units")


def to_ww3(self, filename, ncformat="NETCDF4", compress=None):
    """Save spectra in native WW3 netCDF format.
    Args:
        - filename (str): name of output WW3 netcdf file.
        - ncformat (str): netcdf format for output, see options in native
          to_netcdf method.
        - compress (bool): if False avoids compression; if True output is
          compressed and chunked, has no effect for NETCDF3.
    """

    other = self.copy(deep=True)

    # Converting to degree
    other[attrs.SPECNAME] *= R2D

    # frequency bounds
    df = np.hstack((0, np.diff(other[attrs.FREQNAME]) / 2))
    other["frequency1"] = other[attrs.FREQNAME] - df
    df = np.hstack((np.diff(other[attrs.FREQNAME]) / 2, 0))
    other["frequency2"] = other[attrs.FREQNAME] + df

    # Direction in going-to convention
    other[attrs.DIRNAME] = (other[attrs.DIRNAME] + 180) % 360.
    other[attrs.DIRNAME] = other[attrs.DIRNAME].astype(np.float64)

    # Reorder direction to fit ww3 convention (anti-clockwise from east)
    other = other.sortby((90-other[attrs.DIRNAME])%360)

    # station_name variable
    arr = np.array([[c for c in f"{s:06.0f}"] + [""] * 10 for s in other.site.values], dtype="|S1")
    other["station_name"] = xr.DataArray(
                                data=arr,
                                coords={
                                    "site": other.site,
                                    "string16": [np.nan for i in range(16)]
                                    },
                                dims=("site", "string16"),
                            )

    # Renaming variables
    mapping = {v: k for k, v in MAPPING.items() if v in self.variables}
    other = other.rename(mapping)

    # Setting attributes
    for var_name, var_attrs in VAR_ATTRIBUTES.items():
        if var_name in other:
            other[var_name].attrs = var_attrs

    # for "time" variable
    if "time" in other:
        other.time.encoding["units"] = TIME_UNITS
        other.time.encoding['dtype'] = 'float32'
        times = other.time.to_index().to_pydatetime()

        if len(times) > 1:
            hours = round((times[1] - times[0]).total_seconds() / 3600)
            other.attrs.update({"field_type": f"{hours}-hourly"})

        other.attrs.update(
            {
            "start_date": f"{min(times):%Y-%m-%d %H:%M:%S}",
            "stop_date": f"{max(times):%Y-%m-%d %H:%M:%S}",
            }
        )

    # for "latitude" variable
    if "latitude" in other.dims:
        other.attrs.update(
            {
            "southernmost_latitude": other.latitude.values.min(),
            "northernmost_latitude": other.latitude.values.max(),
            "latitude_resolution": (other.latitude[1] - other.latitude[0]).values,
            "westernmost_longitude": other.longitude.values.min(),
            "easternmost_longitude": other.longitude.values.max(),
            "longitude_resolution": (other.longitude[1] - other.longitude[0]).values,
            }
        )

    # global attrs
    other.attrs.update(VAR_ATTRIBUTES["global"])
    other.attrs.update({"product_name": os.path.basename(filename)})

    ## compression / chunking / packing
    if compress == True:
        ## all encoding attrs will be used from the original data loaded
        pass
    elif compress == False:
        ## some encoding attrs are kept, others either changed or suppressed
        for dkey in list(other.coords) + list(other.data_vars):
            ## changing packing related attrs
            pack_attrs = ['scale_factor', 'add_offset']
            if all(attr in other[dkey].encoding for attr in pack_attrs):
                other[dkey].encoding['dtype'] = 'float32'
                other[dkey].encoding['scale_factor'] = 1.0
                other[dkey].encoding['add_offset'] = 0.0
                ## ensure adequate missing/fillvalues
                fillvalue =  9.96921e+36
                for akey in ['missing_value', '_FillValue']:
                    if akey not in other[dkey].encoding \
                    or other[dkey].encoding[akey] != fillvalue:
                        other[dkey].encoding[akey] = fillvalue

            ## removing compression/chunking attrs if any
            compress_attrs = [
                'complevel',
                'zlib',
                'shuffle',
                'fletcher32',
                'original_shape',
                'chunksizes',
                'contiguous']
            for popkey in compress_attrs:
                if popkey in other[dkey].encoding.keys():
                    other[dkey].encoding.pop(popkey)
    elif compress == None:
        ## for backwards compatibility
        if 'efth' in other and '_FillValue' not in other.efth.encoding:
            other.efth.encoding['_FillValue'] = 9.96921e+36

    # Dump file to disk
    other.to_netcdf(filename)
