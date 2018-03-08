"""
Spectra object based on DataArray to calculate spectral statistics.

Reference:
    Cartwright and Longuet-Higgins (1956). The Statistical Distribution of the Maxima of a Random Function,
        Proceedings of the Royal Society of London. Series A, Mathematical and Physical Sciences, 237, 212-232.
    Holthuijsen LH (2005). Waves in oceanic and coastal waters (page 82).
    Longuet-Higgins (1975). On the joint distribution of the periods and amplitudes of sea waves,
        Journal of Geophysical Research, 80, 2688-2694, doi: 10.1029/JC080i018p02688.

"""
import re
from collections import OrderedDict
from datetime import datetime
import numpy as np
import xarray as xr
import types
import copy
from itertools import product

from wavespectra.core.attributes import attrs
from wavespectra.core.misc import GAMMA, D2R, R2D

try:
    from sympy import Symbol
    from sympy.utilities.lambdify import lambdify
    from sympy.parsing.sympy_parser import parse_expr
except ImportError:
    print 'Warning: cannot import sympy, install "extra" dependencies for full functionality'

# TODO: dimension renaming and sorting in __init__ are not producing intended effect. They correctly modify xarray_obj
#       as defined in xarray.spec._obj but the actual xarray is not modified - and they loose their direct association
# TODO: Implement true_peak method for both tp() and dpm()

_ = np.newaxis

@xr.register_dataarray_accessor('spec')
class SpecArray(object):
    def __init__(self, xarray_obj, dim_map=None):
        """Define spectral attributes."""
        # # Rename spectra coordinates if not 'freq' and/or 'dir'
        # if 'dim_map' is not None:
        #     xarray_obj = xarray_obj.rename(dim_map)

        # # Ensure frequencies and directions are sorted
        # for dim in [attrs.FREQNAME, attrs.DIRNAME]:
        #     if dim in xarray_obj.dims and not self._strictly_increasing(xarray_obj[dim].values):
        #         xarray_obj = self.sort(xarray_obj, dims=[dim])

        self._obj = xarray_obj
        self._non_spec_dims = set(self._obj.dims).difference((attrs.FREQNAME, attrs.DIRNAME))

        # Create attributes for accessor
        self.freq = xarray_obj.freq
        self.dir = xarray_obj.dir if attrs.DIRNAME in xarray_obj.dims else None

        self.df = abs(self.freq[1:].values - self.freq[:-1].values) if len(self.freq) > 1 else np.array((1.0,))
        self.dd = abs(self.dir[1].values - self.dir[0].values) if self.dir is not None and len(self.dir) > 1 else 1.0

        # df darray with freq dimension - may replace above one in the future
        if len(self.freq) > 1:
            self.dfarr = xr.DataArray(data=np.hstack((1.0, np.full(len(self.freq)-2, 0.5), 1.0)) *\
                                           (np.hstack((0.0, np.diff(self.freq))) + np.hstack((np.diff(self.freq), 0.0))),
                                      coords={attrs.FREQNAME: self.freq},
                                      dims=(attrs.FREQNAME))
        else:
            self.dfarr = xr.DataArray(data=np.array((1.0,)),
                                      coords={attrs.FREQNAME: self.freq},
                                      dims=(attrs.FREQNAME))

    def __repr__(self):
        return re.sub(r'<([^\s]+)', '<%s' % (self.__class__.__name__), str(self._obj))

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
            return xr.DataArray(spec_array, coords=spec_coords, dims=spec_coords.keys(), attrs=darray.attrs)
        else:
            return darray

    def _interp_freq(self, fint):
        """Linearly interpolate spectra at frequency fint.

        Assumes self.freq.min() < fint < self.freq.max()

        Returns:
            DataArray with one value in frequency dimension (relative to fint)
            otherwise same dimensions as self._obj

        """
        assert self.freq.min() < fint < self.freq.max(), "Spectra must have frequencies smaller and greater than fint"
        ifreq = self.freq.searchsorted(fint)
        df = np.diff(self.freq.isel(freq=[ifreq-1, ifreq]))[0]
        Sint = self._obj.isel(freq=[ifreq]) * (fint - self.freq.isel(freq=[ifreq-1]).values) +\
            self._obj.isel(freq=[ifreq-1]).values * (self.freq.isel(freq=[ifreq]).values - fint)
        Sint.freq.values = [fint]
        return Sint/df

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
            xr.concat((arr.isel(freq=0), arr), dim=attrs.FREQNAME).diff(attrs.FREQNAME, 1) > 0,
            xr.concat((arr, arr.isel(freq=-1)), dim=attrs.FREQNAME).diff(attrs.FREQNAME, 1) < 0
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
        return (dict(zip(dict_of_ids, x)) for x in product(*dict_of_ids.itervalues()))

    def _inflection(self, fdspec, dfres=0.01, fmin=0.05):
        """Finds points of inflection in smoothed frequency spectra.

        Args:
            fdspec (ndarray): freq-dir 2D specarray
            dfres (float): used to determine length of smoothing window
            fmin (float): minimum frequency for looking for minima/maxima

        """
        if len(self.freq) > 1:
            sf = fdspec.sum(axis=1)
            nsmooth = int(dfres / self.df[0]) # Window size
            if nsmooth > 1:
                sf = np.convolve(sf, np.hamming(nsmooth), 'same') # Smoothed f-spectrum
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

    def sort(self, darray, dims, inplace=False):
        """Sort darray along dims list so that the respective coordinates are sorted."""
        other = darray.copy(deep=not inplace)
        dims = [dims] if not isinstance(dims, list) else dims
        for dim in dims:
            if dim in other.dims:
                if not self._strictly_increasing(darray[dim].values):
                    other = other.isel(**{dim: np.argsort(darray[dim]).values})
            else:
                raise Exception('Dimension %s not in DataArray' % (dim))
        return other

    def oned(self, skipna=True):
        """Returns the one-dimensional frequency spectra.

        Direction dimension is dropped after integrating
        Args:
            skipna (bool): if True (xarray's default) sum of all-nan array returns zero
                           which will result in some methods like self.hs dropping mask

        """
        if self.dir is not None:
            return self.dd * self._obj.sum(dim=attrs.DIRNAME, skipna=skipna)
        else:
            return self._obj.copy(deep=True)

    def split(self, fmin=None, fmax=None, dmin=None, dmax=None):
        """Split spectra over freq and/or dir dims.

        Args:
            fmin (float): lowest frequency to split spectra, by default the lowest
            fmax (float): highest frequency to split spectra, by default the highest
            dmin (float): lowest direction to split spectra over, by default min(dir)
            dmax (float): highest direction to split spectra over, by default max(dir)

        Note:
            spectra are interpolated at fmin / fmax if they are not present in self.freq

        """
        assert fmax > fmin if fmax else True, 'fmax needs to be greater than fmin'
        assert dmax > dmin if dmax else True, 'fmax needs to be greater than fmin'

        # Slice frequencies
        other = self._obj.sel(freq=slice(fmin, fmax))

        # Slice directions
        if attrs.DIRNAME in other.dims and (dmin or dmax):
            other = self.sort(other, dims=[attrs.DIRNAME]).sel(dir=slice(dmin, dmax))

        # Interpolate at fmin
        if (other.freq.min() > fmin) and (self.freq.min() <= fmin):
            other = xr.concat([self._interp_freq(fmin), other], dim=attrs.FREQNAME)

        # Interpolate at fmax
        if (other.freq.max() < fmax) and (self.freq.max() >= fmax):
            other = xr.concat([other, self._interp_freq(fmax)], dim=attrs.FREQNAME)

        return other

    def to_energy(self, standard_name='sea_surface_wave_directional_energy_spectra'):
        """Convert from energy density (m2/Hz/degree) into wave energy spectra (m2)."""
        E = self._obj * self.dfarr * self.dd
        E.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'm^{2}'))))
        return E.rename('energy')

    def hs(self, tail=True, standard_name='sea_surface_wave_significant_height'):
        """Spectral significant wave height Hm0.

        Args:
            tail (bool): if True fit high-frequency tail before integrating spectra
            standard_name (str): CF standard name for defining DataArray attribute

        """
        Sf = self.oned(skipna=False)
        E = (Sf * self.dfarr).sum(dim=attrs.FREQNAME)
        if tail and Sf.freq[-1] > 0.333:
            E += 0.25 * Sf[{attrs.FREQNAME: -1}] * Sf.freq[-1].values
        hs = 4 * np.sqrt(E)
        hs.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'm'))))
        return hs.rename('hs')

    def hmax(self, standard_name='sea_surface_wave_maximum_height'):
        """Maximum wave height Hmax.

        hmax is the most probably value of the maximum individual wave height
        for each sea state. Note that maximum wave height can be higher (but
        not by much since the probability density function is rather narrow).

        Reference:
            Holthuijsen LH (2005). Waves in oceanic and coastal waters (page 82)

        """
        if attrs.TIMENAME in self._obj.coords and self._obj.time.size > 1:
            dt = np.diff(self._obj.time).astype('timedelta64[s]').mean()
            N = (dt.astype(float)/self.tm02()).round() # N is the number of waves in a sea state
            k = np.sqrt(0.5*np.log(N))
        else:
            k = 1.86 # assumes N = 3*3600 / 10.8
        hs = self.hs()
        hs.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'm'))))
        return k * hs

    def scale_by_hs(self, expr, inplace=True, hs_min=-np.inf, hs_max=np.inf,
                    tp_min=-np.inf, tp_max=np.inf, dpm_min=-np.inf, dpm_max=np.inf):
        """Scale spectra using equation based on Significant Wave Height Hs.

        Args:
            expr (str): expression to apply, e.g. '0.13*hs + 0.02'
            inplace (bool): use True to apply transformation in place, False otherwise
            hs_min, hs_max (float): Hs range over which expr is applied
            tp_min, tp_max (float): Tp range over which expr is applied
            dpm_min, dpm_max (float) Dpm range over which expr is applied

        """
        func = lambdify(Symbol('hs'), parse_expr(expr.lower()), modules=['numpy'])
        hs = self.hs()
        k = (func(hs) / hs)**2
        scaled = k * self._obj
        if any([abs(arg) != np.inf for arg in [hs_min, hs_max, tp_min, tp_max, dpm_min, dpm_max]]):
            tp = self.tp()
            dpm = self.dpm()
            scaled = scaled.where(( (hs >= hs_min) & (hs <= hs_max) &
                                    (tp >= tp_min) & (tp <= tp_max) &
                                    (dpm >= dpm_min) & (dpm <= dpm_max)
                                  )).combine_first(self._obj)
        if inplace:
            self._obj.values = scaled.values
        else:
            return scaled

    def tp(self, smooth=True, mask=np.nan,
           standard_name='sea_surface_wave_period_at_variance_spectral_density_maximum'):
        """Peak wave period Tp.

        Args:
            smooth (bool): True for the smooth wave period, False for simply the discrete period
                           corresponding to the maxima in the direction-integrated spectra
            mask (float): value for missing data in output, if there is no peak in the spectra)
            standard_name (str): CF standard name for defining DataArray attribute

        """
        if len(self.freq) < 3:
            return None
        Sf = self.oned()
        ipeak = self._peak(Sf)
        if smooth:
            freq_axis = Sf.get_axis_num(dim=attrs.FREQNAME)
            if len(ipeak.dims) > 1:
                sig1 = np.empty(ipeak.shape)
                sig2 = np.empty(ipeak.shape)
                sig3 = np.empty(ipeak.shape)
                for dim in range(ipeak.shape[1]):
                    sig1[:, dim] = self.freq[ipeak[:, dim]-1].values
                    sig2[:, dim] = self.freq[ipeak[:, dim]+1].values
                    sig3[:, dim] = self.freq[ipeak[:, dim]].values
            else:
                sig1 = self.freq[ipeak-1].values
                sig2 = self.freq[ipeak+1].values
                sig3 = self.freq[ipeak].values
            e1 = self._collapse_array(Sf.values, ipeak.values-1, axis=freq_axis)
            e2 = self._collapse_array(Sf.values, ipeak.values+1, axis=freq_axis)
            e3 = self._collapse_array(Sf.values, ipeak.values, axis=freq_axis)
            p = sig1 + sig2
            q = (e1-e2) / (sig1-sig2)
            r = sig1 + sig3
            t = (e1-e3) / (sig1-sig3)
            a = (t-q) / (r-p)
            fp = (-q+p*a) / (2.*a)
            fp[a >= 0] = sig3[a >= 0]
        else:
            fp = self.freq.values[ipeak]

        tp = (1+0*ipeak) / fp # Cheap way to turn Tp into appropriate DataArray object
        tp.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 's'))))

        # Returns masked dataarray
        return tp.where(ipeak > 0).fillna(mask).rename('tp')

    def momf(self, mom=0):
        """Calculate given frequency moment."""
        fp = self.freq**mom
        mf = self.dfarr * fp * self._obj
        return self._twod(mf.sum(dim=attrs.FREQNAME, skipna=False)).rename('mom{:d}'.format(mom))

    def momd(self, mom=0, theta=90.):
        """Calculate given directional moment."""
        cp = np.cos(np.radians(180 + theta - self.dir))**mom
        sp = np.sin(np.radians(180 + theta - self.dir))**mom
        msin = (self.dd * self._obj * sp).sum(dim=attrs.DIRNAME, skipna=False)
        mcos = (self.dd * self._obj * cp).sum(dim=attrs.DIRNAME, skipna=False)
        return msin, mcos

    def tm01(self,
             standard_name='sea_surface_wave_mean_period_from_variance_spectral_density_first_frequency_moment'):
        """Mean absolute wave period Tm01.

        True average period from the 1st spectral moment.

        Args:
            standard_name (str): CF standard name for defining DataArray attribute

        """
        tm01 = self.momf(0).sum(dim=attrs.DIRNAME) / self.momf(1).sum(dim=attrs.DIRNAME)
        tm01.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 's'))))
        return tm01.rename('tm01')

    def tm02(self,
             standard_name='sea_surface_wave_mean_period_from_variance_spectral_density_second_frequency_moment'):
        """Mean absolute wave period Tm02.

        Average period of zero up-crossings (Zhang, 2011).

        Args:
            standard_name (str): CF standard name for defining DataArray attribute

        """
        tm02 = np.sqrt(self.momf(0).sum(dim=attrs.DIRNAME)/self.momf(2).sum(dim=attrs.DIRNAME))
        tm02.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 's'))))
        return tm02.rename('tm02')

    def dm(self, standard_name='sea_surface_wave_from_mean_direction'):
        """Mean wave direction from the 1st spectral moment Dm.

        Args:
            standard_name (str): CF standard name for defining DataArray attribute

        """
        moms, momc = self.momd(1)
        dm = np.arctan2(moms.sum(dim=attrs.FREQNAME, skipna=False), momc.sum(dim=attrs.FREQNAME, skipna=False))
        dm = (270 - R2D*dm) % 360.
        dm.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'degree'))))
        return dm.rename('dm')

    def dp(self,
           standard_name='sea_surface_wave_from_direction_at_variance_spectral_density_maximum'):
        """Peak wave direction Dp.

        Defined as the direction where the energy density of
        the frequency-integrated spectrum is maximum.

        Args:
            standard_name (str): CF standard name for defining DataArray attribute

        """
        ipeak = self._obj.sum(dim=attrs.FREQNAME).argmax(dim=attrs.DIRNAME)
        template = self._obj.sum(dim=attrs.FREQNAME, skipna=False).sum(dim=attrs.DIRNAME, skipna=False)
        dp = self.dir.values[ipeak.values] * (0 * template + 1)
        dp.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'degree'))))
        return dp.rename('dp')

    def dpm(self, mask=np.nan,
            standard_name='sea_surface_wave_from_direction_at_variance_spectral_density_maximum'):
        """Peak wave direction Dpm.

        From WW3 Manual: peak wave direction, defined like the mean direction, using the
        frequency/wavenumber bin containing of the spectrum F(k) that contains the peak frequency only.

        Args:
            mask (float): value for missing data in output
            standard_name (str): CF standard name for defining DataArray attribute

        """
        ipeak = self._peak(self.oned())
        moms, momc = self.momd(1)
        moms_peak = self._collapse_array(moms.values, ipeak.values, axis=moms.get_axis_num(dim=attrs.FREQNAME))
        momc_peak = self._collapse_array(momc.values, ipeak.values, axis=momc.get_axis_num(dim=attrs.FREQNAME))
        dpm = np.arctan2(moms_peak, momc_peak) * (ipeak*0+1) # Cheap way to turn dp into appropriate DataArray
        dpm = (270 - R2D*dpm) % 360.
        dpm.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'degree'))))
        return dpm.where(ipeak>0).fillna(mask).rename('dpm')

    def dspr(self, standard_name='sea_surface_wave_directional_spread'):
        """Directional wave spreading Dspr.

        The one-sided directional width of the spectrum.

        Args:
            standard_name (str): CF standard name for defining DataArray attribute

        """
        moms, momc = self.momd(1)
        # Manipulate dimensions so calculations work
        moms = self._twod(moms, dim=attrs.DIRNAME)
        momc = self._twod(momc, dim=attrs.DIRNAME)
        mom0 = self._twod(self.momf(0), dim=attrs.FREQNAME).spec.oned(skipna=False)

        dspr = (2 * R2D**2 * (1 - ((moms.spec.momf()**2 + momc.spec.momf()**2)**0.5 / mom0)))**0.5
        dspr = dspr.sum(dim=attrs.FREQNAME, skipna=False).sum(dim=attrs.DIRNAME, skipna=False)
        dspr.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'degree'))))
        return dspr.rename('dspr')

    def crsd(self, theta=90., standard_name='sea_surface_wave_update_missing_part_here'):
        """Add description for this method."""
        cp = np.cos(D2R * (180 + theta - self.dir))
        sp = np.sin(D2R * (180 + theta - self.dir))
        crsd = (self.dd * self._obj * cp * sp).sum(dim=attrs.DIRNAME)
        crsd.attrs.update(OrderedDict((('standard_name', standard_name), ('units', 'm2s'))))
        return crsd.rename('crsd')

    def swe(self, standard_name='sea_surface_wave_spectral_width'):
        """Spectral width parameter by Cartwright and Longuet-Higgins (1956).

        Represents the range of frequencies where the dominant energy exists.

        Args:
            standard_name (str): CF standard name for defining DataArray attribute

        Reference:
            Cartwright and Longuet-Higgins (1956). The statistical distribution
            of maxima of a random function. Proc. R. Soc. A237, 212-232.

        """
        swe = (1. - self.momf(2).sum(dim=attrs.DIRNAME)**2 / (self.momf(0).sum(dim=attrs.DIRNAME)*self.momf(4).sum(dim=attrs.DIRNAME)))**0.5
        swe.values[swe.values < 0.001] = 1.
        swe.attrs.update(OrderedDict((('standard_name', standard_name), ('units', ''))))
        return swe.rename('swe')

    def sw(self, mask=np.nan, standard_name='sea_surface_wave_spectral_width'):
        """Spectral width parameter by Longuet-Higgins (1975).

        Represents energy distribution over entire frequency range.

        Args:
            mask (float): value for missing data in output
            standard_name (str): CF standard name for defining DataArray attribute

        Reference:
            Longuet-Higgins (1975). On the joint distribution of the periods and
            amplitudes of sea waves. JGR, 80, 2688-2694.

        """
        sw = (self.momf(0).sum(dim=attrs.DIRNAME) * self.momf(2).sum(dim=attrs.DIRNAME) / self.momf(1).sum(dim=attrs.DIRNAME)**2 - 1.0)**0.5
        sw.attrs.update(OrderedDict((('standard_name', standard_name), ('units', ''))))
        return sw.where(self.hs() >= 0.001).fillna(mask).rename('sw')

    def celerity(self, depth=False, freq=None):
        """Wave celerity C.

        Args:
            depth (float): depths for calculating C, if not provided
                           the deep water approximation is returned
            freq (ndarray): frequencies for calculating C, by default
                            calculate from self.freq

        Returns;
            C: ndarray of same shape as freq with wave celerity for each frequency

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
            depth (float): depths for calculating L, if not provided
                           the deep water approximation is returned

        Returns;
            L: ndarray of same shape as freq with wavelength for each frequency

        """
        if depth:
            ang_freq = 2 * np.pi * self.freq
            return 2 * np.pi / wavenuma(ang_freq, depth)
        else:
            return 1.56 / self.freq**2

    def partition(self, wsp_darr, wdir_darr, dep_darr, agefac=1.7,
                  wscut=0.3333, hs_min=0.001, nearest=False):
        """Partition wave spectra using WW3 watershed algorithm.

        Args:
            wsp_darr (DataArray): wind speed (m/s)
            wdir_darr (DataArray): Wind direction (degree)
            dep_darr (DataArray): Water depth (m)
            agefac (float): Age factor
            wscut (float): Wind speed cutoff
            hs_min (float): minimum Hs for assigning swell partition
            nearest (bool): if True, wsp, wdir and dep are allowed to be taken from the
                            nearest point if not matching positions in SpecArray (slower)

        Returns:
            part_spec :: SpecArray object with one extra dimension representig partition number

        Note:
            All input DataArray objects must have same non-spectral dimensions as SpecArray

        TODO: We currently loop through each spectrum to calculate the partitions which is slow.
              Ideally we should handle the problem in a multi-dimensional way.

        """
        # Assert spectral dims are present in spectra and non-spectral dims are present in winds and depths
        assert attrs.FREQNAME in self._obj.dims and attrs.DIRNAME in self._obj.dims, ('partition requires E(freq,dir) but freq|dir '
            'dimensions not in SpecArray dimensions (%s)' % (self._obj.dims))
        for darr in (wsp_darr, wdir_darr, dep_darr):
            # Conditional below aims at allowing wsp, wdir, dep to be DataArrays within the SpecArray. not working yet
            if isinstance(darr, str):
                darr = getattr(self, darr)
            assert set(darr.dims) == self._non_spec_dims, ('%s dimensions (%s) need matching non-spectral dimensions '
                'in SpecArray (%s) for consistent slicing' % (darr.name, set(darr.dims), self._non_spec_dims))

        from wavespectra.specpart import specpart

        # Initialise output - one SpecArray for each partition
        all_parts = [0 * self._obj]

        # Predefine these for speed
        dirs = self.dir.values
        freqs = self.freq.values
        ndir = len(dirs)
        nfreq = len(freqs)

        # Slice each possible 2D, freq-dir array out of the full data array
        slice_ids = {dim: range(self._obj[dim].size) for dim in self._non_spec_dims}
        for slice_dict in self._product(slice_ids):
            specarr = self._obj[slice_dict]
            if nearest:
                slice_dict_nearest = {key: self._obj[key][val].values for key, val in slice_dict.items()}
                wsp = float(wsp_darr.sel(method='nearest', **slice_dict_nearest))
                wdir = float(wdir_darr.sel(method='nearest', **slice_dict_nearest))
                dep = float(dep_darr.sel(method='nearest', **slice_dict_nearest))
            else:
                wsp = float(wsp_darr[slice_dict])
                wdir = float(wdir_darr[slice_dict])
                dep = float(dep_darr[slice_dict])

            spectrum = specarr.values
            part_array = specpart.partition(spectrum)
            nparts = part_array.max()

            # Assign new partition if multiple valleys and satisfying some conditions
            for part in range(1, nparts+1):
                part_spec = np.where(part_array == part, spectrum, 0.) # Current partition only
                imax, imin = self._inflection(part_spec, dfres=0.01, fmin=0.05)
                if len(imin) > 0:
                    part_spec[imin[0].squeeze():, :] = 0
                    newpart = part_spec > 0
                    if newpart.sum() > 20:
                        nparts += 1
                        part_array[newpart] = nparts

            # Extend partitions list if any extra one has been detected (+1 because of sea)
            if len(all_parts) < nparts+1:
                for new_part_number in set(range(nparts+1)).difference(range(len(all_parts))):
                    all_parts.append(0 * self._obj)

            # Assign sea and swells partitions based on wind and wave properties
            sea = 0 * spectrum
            swells = list()
            hs_swell = list()
            Up = agefac * wsp * np.cos(D2R*(dirs - wdir))
            for part in range(1, nparts+1):
                part_spec = np.where(part_array == part, spectrum, 0.) # Current partition only
                W = part_spec[np.tile(Up, (nfreq, 1)) > \
                    np.tile(self.celerity(dep, freqs)[:, _], (1, ndir))].sum() / part_spec.sum()
                if W > wscut:
                    sea += part_spec
                else:
                    _hs = hs(part_spec, freqs, dirs)
                    if _hs > hs_min:
                        swells.append(part_spec)
                        hs_swell.append(_hs)
                        hs_swell = [h + np.random.rand()*1e-10 for h in hs_swell] # If two or more partitions with same Hs
            if len(swells) > 1:
                swells = [x for (y, x) in sorted(zip(hs_swell, swells), reverse=True)]

            # Updating partition SpecArrays for current slice
            all_parts[0][slice_dict] = sea
            for ind, swell in enumerate(swells):
                all_parts[ind+1][slice_dict] = swell

        # Concatenate partitions along new axis
        part_coord = xr.DataArray(data=range(len(all_parts)),
                                  coords={'part': range(len(all_parts))},
                                  dims=('part',),
                                  name='part',
                                  attrs=OrderedDict((('standard_name', 'spectral_partition_number'), ('units', '')))
                                 )
        return xr.concat(all_parts, dim=part_coord)

    def dpw(self):
        """Wave spreading at the peak wave frequency."""
        raise NotImplementedError('Wave spreading at the peak wave frequency method not defined')

    def stats(self, stats, fmin=None, fmax=None, dmin=None, dmax=None, names=None):
        """Calculate multiple spectral stats into a Dataset.

        Args:
            stats (list): strings specifying stats to be calculated
                  (dict): keys are stats names, vals are dicts with kwargs to use with corresponding method
            fmin (float): lower frequencies for splitting spectra before calculating stats
            fmax (float): upper frequencies for splitting spectra before calculating stats
            dmin (float): lower directions for splitting spectra before calculating stats
            dmax (float): upper directions for splitting spectra before calculating stats
            names (list): strings to rename each stat in output Dataset (not working as expected though)

        Returns:
            Dataset with all spectral statistics specified

        Note:
            All stats names must correspond to methods implemented in this class
            If names is provided, its length must correspond to the length of stats

        """
        if any((fmin, fmax, dmin, dmax)):
            spectra = self.split(fmin=fmin, fmax=fmax, dmin=dmin, dmax=dmax)
        else:
            spectra = self._obj

        if isinstance(stats, (list, tuple)):
            stats_dict = {s: {} for s in stats}
        elif isinstance(stats, dict):
            stats_dict = stats
        else:
            raise ValueError('stats must be either a container or a dictionary')

        names = names or stats_dict.keys()
        if len(names) != len(stats_dict):
            raise ValueError('length of names does not correspond to the number of stats')

        params = list()
        for name, (func, kwargs) in zip(names, stats_dict.items()):
            try:
                stats_func = getattr(spectra.spec, func)
            except:
                raise IOError('%s is not implemented as a method in %s' % (func, self.__class__.__name__))
            if callable(stats_func):
                params.append(stats_func(**kwargs).rename(name))
            else:
                raise IOError('%s attribute of %s is not callable' % (func, self.__class__.__name__))

        return xr.merge(params)

def wavenuma(ang_freq, water_depth):
    """Chen and Thomson wavenumber approximation."""
    k0h = 0.10194 * ang_freq * ang_freq * water_depth
    D = [0, 0.6522, 0.4622, 0, 0.0864, 0.0675]
    a = 1.0
    for i in range(1, 6):
        a += D[i] * k0h**i
    return (k0h * (1 + 1./(k0h*a))**0.5) / water_depth

def hs(spec, freqs, dirs, tail=True):
    """Significant wave height Hmo.

    Copied as a function so it can be used in a generic context.

    Args:
        spec (ndarray): wave spectrum array
        freqs (darray): wave frequency array
        dirs (darray): wave direction array
        tail (bool): if True fit high-frequency tail before integrating spectra

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
    return 4. * np.sqrt(Etot)

if __name__ == '__main__':
    pass
