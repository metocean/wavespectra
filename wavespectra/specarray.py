"""Spectra object based on DataArray to calculate spectral statistics.

References:
    - Cartwright and Longuet-Higgins (1956). The Statistical Distribution of the Maxima of a Random Function,
      Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, 237, 212-232.
    - Holthuijsen LH (2005). Waves in oceanic and coastal waters (page 82).
    - Longuet-Higgins (1975). On the joint distribution of the periods and amplitudes of sea waves,
      Journal of Geophysical Research, 80, 2688-2694, doi: 10.1029/JC080i018p02688.

"""
import copy
import inspect
import re
import types
import warnings
from collections import OrderedDict
from datetime import datetime
from itertools import product

import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs
from wavespectra.core.misc import D2R, GAMMA, R2D
from wavespectra.plot import _PlotMethods

try:
    from sympy import Symbol
    from sympy.parsing.sympy_parser import parse_expr
    from sympy.utilities.lambdify import lambdify
except ImportError:
    warnings.warn(
        'Cannot import sympy, install "extra" dependencies for full functionality'
    )

# TODO: dimension renaming and sorting in __init__ are not producing intended effect. They correctly modify xarray_obj
#       as defined in xarray.spec._obj but the actual xarray is not modified - and they lose their direct association

_ = np.newaxis


@xr.register_dataarray_accessor("spec")
class SpecArray:
    def __init__(self, xarray_obj, dim_map=None):
        """Define spectral attributes."""
        # # Rename spectra coordinates if not 'freq' and/or 'dir'
        # if 'dim_map' is not None:
        #     xarray_obj = xarray_obj.rename(dim_map)

        # # Ensure frequencies and directions are sorted
        # for dim in [attrs.FREQNAME, attrs.DIRNAME]:
        #     if dim in xarray_obj.dims and not self._strictly_increasing(xarray_obj[dim].values):
        #         xarray_obj = self._obj.sortby([dim])

        self._obj = xarray_obj
        self._non_spec_dims = set(self._obj.dims).difference(
            (attrs.FREQNAME, attrs.DIRNAME)
        )

        # Create attributes for accessor
        self.freq = xarray_obj.freq
        self.dir = xarray_obj.dir if attrs.DIRNAME in xarray_obj.dims else None

        self.df = (
            abs(self.freq[1:].values - self.freq[:-1].values)
            if len(self.freq) > 1
            else np.array((1.0,))
        )
        self.dd = (
            abs(self.dir[1].values - self.dir[0].values)
            if self.dir is not None and len(self.dir) > 1
            else 1.0
        )

        # df darray with freq dimension - may replace above one in the future
        if len(self.freq) > 1:
            df = np.diff(self.freq)
            dfarr_data = 0.5 * (np.hstack((df[0], df)) + np.hstack((df, df[-1])))
        else:
            dfarr_data = np.array((1.0,))

        self.dfarr = xr.DataArray(
            data=dfarr_data, coords={attrs.FREQNAME: self.freq}, dims=(attrs.FREQNAME)
        )

    def __repr__(self):
        return re.sub(r"<([^\s]+)", "<%s" % (self.__class__.__name__), str(self._obj))

    def _strictly_increasing(self, arr):
        """Check if array is sorted in increasing order."""
        return all(x < y for x, y in zip(arr, arr[1:]))

    def _collapse_array(self, arr, indices, axis):
        """Collapse ndim array [arr] along [axis] using [indices]."""
        magic_index = [np.arange(i) for i in indices.shape]
        magic_index = np.ix_(*magic_index)
        magic_index = magic_index[:axis] + (indices,) + magic_index[axis:]
        return arr[magic_index]

    def _twod(self, darray, dim=attrs.DIRNAME):
        """Ensure dir,freq dims are present so moment calculations won't break."""
        if dim not in darray.dims:
            spec_array = np.expand_dims(darray.values, axis=-1)
            spec_coords = OrderedDict((dim, darray[dim].values) for dim in darray.dims)
            spec_coords.update({dim: np.array((1,))})
            return xr.DataArray(
                spec_array,
                coords=spec_coords,
                dims=spec_coords.keys(),
                attrs=darray.attrs,
            )
        else:
            return darray

    def _interp_freq(self, fint):
        """Linearly interpolate spectra at frequency fint.

        Assumes self.freq.min() < fint < self.freq.max()

        Returns:
            DataArray with one value in frequency dimension (relative to fint)
            otherwise same dimensions as self._obj

        """
        assert (
            self.freq.min() < fint < self.freq.max()
        ), "Spectra must have frequencies smaller and greater than fint"
        ifreq = self.freq.searchsorted(fint)
        df = np.diff(self.freq.isel(freq=[ifreq - 1, ifreq]))[0]
        Sint = self._obj.isel(freq=[ifreq]) * (
            fint - self.freq.isel(freq=[ifreq - 1]).values
        ) + self._obj.isel(freq=[ifreq - 1]).values * (
            self.freq.isel(freq=[ifreq]).values - fint
        )
        freq_var = Sint.freq
        Sint = Sint.assign_coords(freq=xr.DataArray(data=[fint],
                                                    coords=freq_var.coords,
                                                    dims=freq_var.dims,
                                                    name=freq_var.name,
                                                    attrs=freq_var.attrs))
        return Sint / df

    def _peak(self, arr):
        """Returns indices of largest peaks along freq dim in a ND-array.

        Args:
            arr (SpecArray): 1D spectra (integrated over directions)

        Returns:
            ipeak (SpecArray): indices for slicing arr at the frequency peak

        Note:
            A peak is found when arr(ipeak-1) < arr(ipeak) < arr(ipeak+1)
            ipeak==0 does not satisfy above condition and is assumed to be
                missing_value in other parts of the code
        """
        ispeak = np.logical_and(
            xr.concat((arr.isel(freq=0), arr), dim=attrs.FREQNAME).diff(
                attrs.FREQNAME, n=1, label="upper"
            )
            > 0,
            xr.concat((arr, arr.isel(freq=-1)), dim=attrs.FREQNAME).diff(
                attrs.FREQNAME, n=1, label="lower"
            )
            < 0,
        )
        ipeak = arr.where(ispeak).fillna(0).argmax(dim=attrs.FREQNAME).astype(int)
        return ipeak

    def _product(self, dict_of_ids):
        """Dot product of a dictionary of ids used to construct arbitrary slicing dicts.

        Example:
            Input dictionary :: {'site': [0,1], 'time': [0,1]}
            Output iterator :: {'site': 0, 'time': 0}
                               {'site': 0, 'time': 1}
                               {'site': 1, 'time': 0}
                               {'site': 1, 'time': 1}

        """
        return (dict(zip(dict_of_ids, x)) for x in product(*iter(dict_of_ids.values())))

    def _inflection(self, fdspec, dfres=0.01, fmin=0.05):
        """Finds points of inflection in smoothed frequency spectra.

        Args:
            fdspec (ndarray): freq-dir 2D specarray
            dfres (float): used to determine length of smoothing window
            fmin (float): minimum frequency for looking for minima/maxima

        """
        if len(self.freq) > 1:
            sf = fdspec.sum(axis=1)
            nsmooth = int(dfres / self.df[0])  # Window size
            if nsmooth > 1:
                sf = np.convolve(sf, np.hamming(nsmooth), "same")  # Smoothed f-spectrum
            sf[(sf < 0) | (self.freq.values < fmin)] = 0
            diff = np.diff(sf)
            imax = np.argwhere(np.diff(np.sign(diff)) == -2) + 1
            imin = np.argwhere(np.diff(np.sign(diff)) == 2) + 1
        else:
            imax = 0
            imin = 0
        return imax, imin

    def _same_dims(self, other):
        """Check if another SpecArray has same non-spectral dims.

        Used to ensure consistent slicing

        """
        return set(other.dims) == self._non_spec_dims

    def _my_name(self):
        """Returns the caller's name."""
        return inspect.stack()[1][3]

    def _standard_name(self, varname):
        try:
            return attrs.ATTRS[varname]["standard_name"]
        except AttributeError:
            warnings.warn(
                f"Cannot set standard_name for variable {varname}. "
                "Ensure it is defined in attributes.yml"
            )
            return ""

    def _units(self, varname):
        try:
            return attrs.ATTRS[varname]["units"]
        except AttributeError:
            warnings.warn(
                f"Cannot set units for variable {varname}. "
                "Ensure it is defined in attributes.yml"
            )
            return ""

    @property
    def plot(self):
        """Access plotting functions.
        >>> d = DataArray([[1, 2], [3, 4]])
        For convenience just call this directly
        >>> d.plot()
        Or use it as a namespace to use xarray.plot functions as
        DataArray methods
        >>> d.plot.imshow()  # equivalent to xarray.plot.imshow(d)
        """
        return _PlotMethods(self._obj)

    def oned(self, skipna=True):
        """Returns the one-dimensional frequency spectra.

        Direction dimension is dropped after integrating.

        Args:
            - skipna (bool): choose it to skip nans when integrating spectra.
              This is the default behaviour for sum() in DataArray. Notice it
              converts masks, where the entire array is nan, into zero.

        """
        if self.dir is not None:
            return self.dd * self._obj.sum(dim=attrs.DIRNAME, skipna=skipna)
        else:
            return self._obj.copy(deep=True)

    def split(self, fmin=None, fmax=None, dmin=None, dmax=None):
        """Split spectra over freq and/or dir dims.

        Args:
            - fmin (float): lowest frequency to split spectra, by default the lowest.
            - fmax (float): highest frequency to split spectra, by default the highest.
            - dmin (float): lowest direction to split spectra over, by default min(dir).
            - dmax (float): highest direction to split spectra over, by default max(dir).

        Note:
            - spectra are interpolated at `fmin` / `fmax` if they are not present in self.freq

        """
        assert fmax > fmin if fmax and fmin else True, "fmax needs to be greater than fmin"

        # Slice frequencies
        other = self._obj.sel(freq=slice(fmin, fmax))

        # Slice directions
        if attrs.DIRNAME in other.dims and (dmin or dmax):
            tmp = self._obj.sortby([attrs.DIRNAME])
            if dmin and dmax and dmin > dmax:
                other = xr.concat(
                    [tmp.sel(dir=slice(dmin, None)), tmp.sel(dir=slice(None, dmax))],
                    dim="dir",
                )
            else:
                other = tmp.sel(dir=slice(dmin, dmax))

        # Interpolate at fmin
        if (fmin is not None) and (other.freq.min() > fmin) and (self.freq.min() <= fmin):
            other = xr.concat([self._interp_freq(fmin), other], dim=attrs.FREQNAME)

        # Interpolate at fmax
        if (fmax is not None) and (other.freq.max() < fmax) and (self.freq.max() >= fmax):
            other = xr.concat([other, self._interp_freq(fmax)], dim=attrs.FREQNAME)

        return other

    def to_energy(self, standard_name="sea_surface_wave_directional_energy_spectra"):
        """Convert from energy density (m2/Hz/degree) into wave energy spectra (m2)."""
        E = self._obj * self.dfarr * self.dd
        E.attrs.update(
            OrderedDict((("standard_name", standard_name), ("units", "m^{2}")))
        )
        return E.rename("energy")

    def hs(self, tail=True):
        """Spectral significant wave height Hm0.

        Args:
            - tail (bool): if True fit high-frequency tail before integrating spectra.

        """
        Sf = self.oned(skipna=False)
        E = (Sf * self.dfarr).sum(dim=attrs.FREQNAME)
        if tail and Sf.freq[-1] > 0.333:
            E += (
                0.25
                * Sf[{attrs.FREQNAME: -1}].drop(attrs.FREQNAME)
                * Sf.freq[-1].values
            )
        hs = 4 * np.sqrt(E)
        hs.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return hs.rename(self._my_name())

    def hmax(self, k=None, D=3600.):
        """Maximum wave height Hmax.

        Args:
            - k (float): factor to multiply hs by
            - D (float): time duration considered in the estimation (assumed stationary conditions)

        hmax is the most probable value of the maximum individual wave height
        for each sea state. Note that maximum wave height can be higher (but
        not by much since the probability density function is rather narrow).

        Note: k = 1.86  assumes N = 3*3600 / 10.8 (1000 crests)

        Reference:
            - Holthuijsen LH (2005). Waves in oceanic and coastal waters (page 82).

        """
        if not k:
            N = (
                D / self.tm02()
            ).round()  # N is the number of waves in a sea state
            k = np.sqrt(0.5 * np.log(N))
        hmax = k * self.hs()
        hmax.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return hmax.rename(self._my_name())

    def scale_by_hs(
        self,
        expr,
        inplace=True,
        hs_min=-np.inf,
        hs_max=np.inf,
        tp_min=-np.inf,
        tp_max=np.inf,
        dpm_min=-np.inf,
        dpm_max=np.inf,
    ):
        """Scale spectra using equation based on Significant Wave Height Hs.

        Args:
            - expr (str): expression to apply, e.g. '0.13*hs + 0.02'.
            - inplace (bool): use True to apply transformation in place, False otherwise.
            - hs_min, hs_max (float): Hs range over which expr is applied.
            - tp_min, tp_max (float): Tp range over which expr is applied.
            - dpm_min, dpm_max (float) Dpm range over which expr is applied.

        """
        func = lambdify(Symbol("hs"), parse_expr(expr.lower()), modules=["numpy"])
        hs = self.hs()
        k = (func(hs) / hs) ** 2
        scaled = k * self._obj
        if any(
            [
                abs(arg) != np.inf
                for arg in [hs_min, hs_max, tp_min, tp_max, dpm_min, dpm_max]
            ]
        ):
            tp = self.tp()
            dpm = self.dpm()
            scaled = scaled.where(
                (hs >= hs_min)
                & (hs <= hs_max)
                & (tp >= tp_min)
                & (tp <= tp_max)
                & (dpm >= dpm_min)
                & (dpm <= dpm_max)
            ).combine_first(self._obj)
        if inplace:
            self._obj.values = scaled.values
        else:
            return scaled

    def tp(self, smooth=True, mask=np.nan):
        """Peak wave period Tp.

        Args:
            - smooth (bool): True for the smooth wave period, False for simply the discrete
              period corresponding to the maxima in the direction-integrated spectra.
            - mask (float): value for missing data in output, if there is no peak in the spectra).

        """
        if len(self.freq) < 3:
            return None
        Sf = self.oned()
        ipeak = self._peak(Sf).load()
        fp = self.freq.astype("float64")[ipeak].drop("freq")
        if smooth:
            f1, f2, f3 = (self.freq[ipeak + i].values for i in [-1, 0, 1])
            e1, e2, e3 = (Sf.isel(freq=ipeak + i).values for i in [-1, 0, 1])
            s12 = f1 + f2
            q12 = (e1 - e2) / (f1 - f2)
            q13 = (e1 - e3) / (f1 - f3)
            qa = (q13 - q12) / (f3 - f2)
            qa = np.ma.masked_array(qa, qa >= 0)
            ind = ~qa.mask
            fpsmothed = (s12[ind] - q12[ind] / qa[ind]) / 2.0
            fp.values[ind] = fpsmothed
        tp = (1 / fp).where(ipeak > 0).fillna(mask).rename("tp")
        tp.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return tp

    def momf(self, mom=0):
        """Calculate given frequency moment."""
        fp = self.freq ** mom
        mf = self.dfarr * fp * self._obj
        return self._twod(mf.sum(dim=attrs.FREQNAME, skipna=False)).rename(
            f"mom{mom:d}"
        )

    def momd(self, mom=0, theta=90.0):
        """Calculate given directional moment."""
        cp = np.cos(np.radians(180 + theta - self.dir)) ** mom
        sp = np.sin(np.radians(180 + theta - self.dir)) ** mom
        msin = (self.dd * self._obj * sp).sum(dim=attrs.DIRNAME, skipna=False)
        mcos = (self.dd * self._obj * cp).sum(dim=attrs.DIRNAME, skipna=False)
        return msin, mcos

    def tm01(self):
        """Mean absolute wave period Tm01.

        True average period from the 1st spectral moment.

        """
        tm01 = self.momf(0).sum(dim=attrs.DIRNAME) / self.momf(1).sum(dim=attrs.DIRNAME)
        tm01.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return tm01.rename(self._my_name())

    def tm02(self):
        """Mean absolute wave period Tm02.

        Average period of zero up-crossings (Zhang, 2011).

        """
        tm02 = np.sqrt(
            self.momf(0).sum(dim=attrs.DIRNAME) / self.momf(2).sum(dim=attrs.DIRNAME)
        )
        tm02.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return tm02.rename(self._my_name())

    def dm(self):
        """Mean wave direction from the 1st spectral moment Dm."""
        moms, momc = self.momd(1)
        dm = np.arctan2(
            (moms * self.dfarr).sum(dim=attrs.FREQNAME, skipna=False),
            (momc * self.dfarr).sum(dim=attrs.FREQNAME, skipna=False),
        )
        dm = (270 - R2D * dm) % 360.0
        dm.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return dm.rename(self._my_name())

    def dp(self):
        """Peak wave direction Dp.

        Defined as the direction where the energy density of the
        frequency-integrated spectrum is maximum.

        """
        ipeak = self._obj.sum(dim=attrs.FREQNAME).argmax(dim=attrs.DIRNAME)
        template = self._obj.sum(dim=attrs.FREQNAME, skipna=False).sum(
            dim=attrs.DIRNAME, skipna=False
        )
        dp = self.dir.values[ipeak.values] * (0 * template + 1)
        dp.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return dp.rename(self._my_name())

    def dpm(self, mask=np.nan):
        """Peak wave direction Dpm.

        From WW3 Manual: peak wave direction, defined like the mean direction,
        using the frequency/wavenumber bin containing of the spectrum F(k)
        that contains the peak frequency only.

        Args:
            - mask (float): value for missing data in output.

        """
        ipeak = self._peak(self.oned())
        moms, momc = self.momd(1)
        moms_peak = self._collapse_array(
            moms.values, ipeak.values, axis=moms.get_axis_num(dim=attrs.FREQNAME)
        )
        momc_peak = self._collapse_array(
            momc.values, ipeak.values, axis=momc.get_axis_num(dim=attrs.FREQNAME)
        )
        dpm = np.arctan2(moms_peak, momc_peak) * (
            ipeak * 0 + 1
        )  # Cheap way to turn dp into appropriate DataArray
        dpm = (270 - R2D * dpm) % 360.0
        dpm.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return dpm.where(ipeak > 0).fillna(mask).rename(self._my_name())

    def dspr(self):
        """Directional wave spreading Dspr.

        The one-sided directional width of the spectrum.

        """
        moms, momc = self.momd(1)
        # Manipulate dimensions so calculations work
        moms = self._twod(moms, dim=attrs.DIRNAME)
        momc = self._twod(momc, dim=attrs.DIRNAME)
        mom0 = self._twod(self.momf(0), dim=attrs.FREQNAME).spec.oned(skipna=False)

        dspr = (
            2
            * R2D ** 2
            * (1 - ((moms.spec.momf() ** 2 + momc.spec.momf() ** 2) ** 0.5 / mom0))
        ) ** 0.5
        dspr = dspr.sum(dim=attrs.FREQNAME, skipna=False).sum(
            dim=attrs.DIRNAME, skipna=False
        )
        dspr.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return dspr.rename(self._my_name())

    def crsd(self, theta=90.0):
        """Add description."""
        cp = np.cos(D2R * (180 + theta - self.dir))
        sp = np.sin(D2R * (180 + theta - self.dir))
        crsd = (self.dd * self._obj * cp * sp).sum(dim=attrs.DIRNAME)
        crsd.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return crsd.rename(self._my_name())

    def swe(self):
        """Spectral width parameter by Cartwright and Longuet-Higgins (1956).

        Represents the range of frequencies where the dominant energy exists.

        Reference:
            - Cartwright and Longuet-Higgins (1956). The statistical distribution
              of maxima of a random function. Proc. R. Soc. A237, 212-232.

        """
        swe = (
            1.0
            - self.momf(2).sum(dim=attrs.DIRNAME) ** 2
            / (
                self.momf(0).sum(dim=attrs.DIRNAME)
                * self.momf(4).sum(dim=attrs.DIRNAME)
            )
        ) ** 0.5
        swe = swe.where(swe >= 0.001, 1.0)
        swe.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return swe.rename(self._my_name())

    def sw(self, mask=np.nan):
        """Spectral width parameter by Longuet-Higgins (1975).

        Represents energy distribution over entire frequency range.

        Args:
            - mask (float): value for missing data in output

        Reference:
            - Longuet-Higgins (1975). On the joint distribution of the periods and
              amplitudes of sea waves. JGR, 80, 2688-2694.

        """
        sw = (
            self.momf(0).sum(dim=attrs.DIRNAME)
            * self.momf(2).sum(dim=attrs.DIRNAME)
            / self.momf(1).sum(dim=attrs.DIRNAME) ** 2
            - 1.0
        ) ** 0.5
        sw.attrs.update(
            OrderedDict(
                (
                    ("standard_name", self._standard_name(self._my_name())),
                    ("units", self._units(self._my_name())),
                )
            )
        )
        return sw.where(self.hs() >= 0.001).fillna(mask).rename(self._my_name())

    def celerity(self, depth=False, freq=None):
        """Wave celerity C.

        Args:
            - depth (float): depths for calculating C, if not provided
              the deep water approximation is returned.
            - freq (ndarray): frequencies for calculating C, by default
              calculate from self.freq.

        Returns;
            - C: ndarray of same shape as freq with wave celerity for each frequency.

        """
        if freq is None:
            freq = self.freq
        if depth:
            ang_freq = 2 * np.pi * freq
            return ang_freq / wavenuma(ang_freq, depth)
        else:
            return 1.56 / freq

    def wavelen(self, depth=False):
        """Wavelength L.

        Args:
            - depth (float): depths for calculating L, if not provided
              the deep water approximation is returned.

        Returns;
            - L: ndarray of same shape as freq with wavelength for each frequency.

        """
        if depth:
            ang_freq = 2 * np.pi * self.freq
            return 2 * np.pi / wavenuma(ang_freq, depth)
        else:
            return 1.56 / self.freq ** 2

    def partition(
        self,
        wsp_darr=None,
        wdir_darr=None,
        dep_darr=None,
        agefac=1.7,
        wscut=0.3333,
        hs_min=0.001,
        nearest=False,
        max_swells=5,
        fortran_code=True,
    ):
        """Partition wave spectra using WW3 watershed algorithm.

        Args:
            - wsp_darr (DataArray): wind speed (m/s).
            - wdir_darr (DataArray): Wind direction (degree).
            - dep_darr (DataArray): Water depth (m).
            - agefac (float): Age factor.
            - wscut (float): Wind speed cutoff.
            - hs_min (float): minimum Hs for assigning swell partition.
            - nearest (bool): if True, wsp, wdir and dep are allowed to be taken from the.
              nearest point if not matching positions in SpecArray (slower).
            - max_swells (int): maximum number of swells to extract
            - fortran_code (bool): Try to use fortran partition code instead of (~30x slower) pure python

        Returns:
            - part_spec (SpecArray): partitioned spectra with one extra dimension
              representig partition number.

        Note:
            - All input DataArray objects must have same non-spectral
              dimensions as SpecArray.
        References:
            - Hanson, Jeffrey L., et al. "Pacific hindcast performance of three numerical
              wave models." Journal of Atmospheric and Oceanic Technology 26.8 (2009): 1614-1633.

        TODO:
            - We currently loop through each spectrum to calculate the partitions which
              is slow. Ideally we should handle the problem in a multi-dimensional way.

        """
        # Assert spectral dims are present in spectra and non-spectral dims are present in winds and depths
        assert attrs.FREQNAME in self._obj.dims and attrs.DIRNAME in self._obj.dims, (
            "partition requires E(freq,dir) but freq|dir "
            "dimensions not in SpecArray dimensions (%s)" % (self._obj.dims)
        )
        for darr in (wsp_darr, wdir_darr, dep_darr):
            # Conditional below aims at allowing wsp, wdir, dep to be DataArrays within the SpecArray. not working yet
            if isinstance(darr, str):
                darr = getattr(self, darr)
            assert set(darr.dims) == self._non_spec_dims, (
                "%s dimensions (%s) need matching non-spectral dimensions "
                "in SpecArray (%s) for consistent slicing"
                % (darr.name, set(darr.dims), self._non_spec_dims)
            )

        if fortran_code:
            try:
                from wavespectra.specpart import specpart
            except ImportError:
                fortran_code = False

        if not fortran_code:
            from wavespectra.core import specpartpy as specpart

        # Initialise output - one SpecArray for each partition
        all_parts = [0 * self._obj.load() for i in range(1 + max_swells)]

        # Predefine these for speed
        dirs = self.dir.values
        freqs = self.freq.values
        dfarr = self.dfarr.values
        ndir = len(dirs)
        nfreq = len(freqs)
        wsp_darr.load()
        wdir_darr.load()
        dep_darr.load()

        # Slice each possible 2D, freq-dir array out of the full data array
        slice_ids = {dim: range(self._obj[dim].size) for dim in self._non_spec_dims}
        for slice_dict in self._product(slice_ids):
            spectrum = self._obj[slice_dict].values
            if nearest:
                slice_dict_nearest = {
                    key: self._obj[key][val].values for key, val in slice_dict.items()
                }
                wsp = float(wsp_darr.sel(method="nearest", **slice_dict_nearest))
                wdir = float(wdir_darr.sel(method="nearest", **slice_dict_nearest))
                dep = float(dep_darr.sel(method="nearest", **slice_dict_nearest))
            else:
                wsp = float(wsp_darr[slice_dict])
                wdir = float(wdir_darr[slice_dict])
                dep = float(dep_darr[slice_dict])

            Up = agefac * wsp * np.cos(D2R * (dirs - wdir))
            windbool = np.tile(Up, (nfreq, 1)) > np.tile(
                self.celerity(dep, freqs)[:, _], (1, ndir)
            )

            part_array = specpart.partition(spectrum)

            ipeak = 1  # values from specpart.partition start at 1
            part_array_max = part_array.max()
            partitions_hs_swell = np.zeros(
                part_array_max + 1
            )  # +1 because zeroth index is used for windseas
            while ipeak <= part_array_max:
                part_spec = np.where(
                    part_array == ipeak, spectrum, 0.0
                )  # Current partition

                # Assign new partition if multiple valleys and satisfying some conditions
                imax, imin = self._inflection(part_spec, dfres=0.01, fmin=0.05)
                if len(imin) > 0:
                    # TODO: swap part_spec and part_spec_new and deal with more than one imin value ?
                    part_spec_new = part_spec.copy()
                    part_spec_new[imin[0].squeeze() :, :] = 0
                    newpart = part_spec_new > 0
                    if newpart.sum() > 20:
                        part_spec[newpart] = 0
                        part_array_max += 1
                        part_array[newpart] = part_array_max
                        partitions_hs_swell = np.append(partitions_hs_swell, 0)

                # Group sea partitions and sort swells by hs
                W = np.dot((part_spec*windbool).sum(1), dfarr) / np.dot(part_spec.sum(1), dfarr)
                if W > wscut:
                    part_array[part_array == ipeak] = 0  # mark as windsea
                else:
                    partitions_hs_swell[ipeak] = hs(part_spec, freqs, dirs)

                ipeak += 1

            num_swells = min(max_swells, sum(partitions_hs_swell > hs_min))

            sorted_swells = np.flipud(partitions_hs_swell[1:].argsort() + 1)
            parts = np.concatenate(([0], sorted_swells[:num_swells]))
            for ind, part in enumerate(parts):
                all_parts[ind][slice_dict] = np.where(part_array == part, spectrum, 0.0)

        # Concatenate partitions along new axis
        part_coord = xr.DataArray(
            data=range(len(all_parts)),
            coords={"part": range(len(all_parts))},
            dims=("part",),
            name="part",
            attrs=OrderedDict(
                (("standard_name", "spectral_partition_number"), ("units", ""))
            ),
        )
        return xr.concat(all_parts, dim=part_coord)

    def dpw(self):
        """Wave spreading at the peak wave frequency."""
        raise NotImplementedError(
            "Wave spreading at the peak wave frequency method not defined"
        )

    def stats(self, stats, fmin=None, fmax=None, dmin=None, dmax=None, names=None):
        """Calculate multiple spectral stats into a Dataset.

        Args:
            - stats (list): strings specifying stats to be calculated.
              (dict): keys are stats names, vals are dicts with kwargs to use with corresponding method.
            - fmin (float): lower frequencies for splitting spectra before
              calculating stats.
            - fmax (float): upper frequencies for splitting spectra before calculating stats.
            - dmin (float): lower directions for splitting spectra before calculating stats.
            - dmax (float): upper directions for splitting spectra before calculating stats.
            - names (list): strings to rename each stat in output Dataset (not working as expected though).

        Returns:
            - Dataset with all spectral statistics specified.

        Note:
            - All stats names must correspond to methods implemented in this class.
            - If names is provided, its length must correspond to the length of stats.
            - If names is provided and stats is a dict, stats must be OrderedDict.

        """
        if any((fmin, fmax, dmin, dmax)):
            spectra = self.split(fmin=fmin, fmax=fmax, dmin=dmin, dmax=dmax)
        else:
            spectra = self._obj

        if isinstance(stats, (list, tuple)):
            stats_dict = OrderedDict((s, {}) for s in stats)
        elif isinstance(stats, dict):
            stats_dict = stats
        else:
            raise ValueError("stats must be either a container or a dictionary")

        names = names or stats_dict.keys()
        if len(names) != len(stats_dict):
            raise ValueError(
                "length of names does not correspond to the number of stats"
            )

        params = list()
        for func, kwargs in stats_dict.items():
            try:
                stats_func = getattr(spectra.spec, func)
            except:
                raise OSError(
                    "%s is not implemented as a method in %s"
                    % (func, self.__class__.__name__)
                )
            if callable(stats_func):
                params.append(stats_func(**kwargs))
            else:
                raise OSError(
                    "%s attribute of %s is not callable"
                    % (func, self.__class__.__name__)
                )

        return xr.merge(params).rename(dict(zip(stats_dict.keys(), names)))


def wavenuma(ang_freq, water_depth):
    """Chen and Thomson wavenumber approximation."""
    k0h = 0.10194 * ang_freq * ang_freq * water_depth
    D = [0, 0.6522, 0.4622, 0, 0.0864, 0.0675]
    a = 1.0
    for i in range(1, 6):
        a += D[i] * k0h ** i
    return (k0h * (1 + 1.0 / (k0h * a)) ** 0.5) / water_depth


def hs(spec, freqs, dirs, tail=True):
    """Significant wave height Hmo.

    Copied as a function so it can be used in a generic context.

    Args:
        - spec (ndarray): wave spectrum array
        - freqs (darray): wave frequency array
        - dirs (darray): wave direction array
        - tail (bool): if True fit high-frequency tail before integrating spectra

    """
    df = abs(freqs[1:] - freqs[:-1])
    if len(dirs) > 1:
        ddir = abs(dirs[1] - dirs[0])
        E = ddir * spec.sum(1)
    else:
        E = np.squeeze(spec)
    Etot = 0.5 * sum(df * (E[1:] + E[:-1]))
    if tail and freqs[-1] > 0.333:
        Etot += 0.25 * E[-1] * freqs[-1]
    return 4.0 * np.sqrt(Etot)
