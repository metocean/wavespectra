import os

import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.specdataset import SpecDataset


def prepare_reconstruction(spec_info, base_dset=None):
    """ Load parameters for spectral reconstruction.

    Arguments:
        spec_info: dictionary for updating reconstruction defaults. Optionally extra variables to keep or rename.
        - coordinates:
            - spectral: freq, dir
        - frequency spectrum:
            - jonswap/TMA: hs, tp, gamma, dpt
            - ochihubble: hs, tp, lam
        - directional distribution:
            - cos2s: dp, dspr
            - normal: dp, dspr
        base_dset: path or xarray dataset object

    Returns:
        ds: xarray dataset with parameters for spectral reconstruction. part dimension is used for concatenation
    """
    reconstruction_defaults = {
        "freq": np.arange(0.04, 1.0, 0.02),  # frequencies
        "dir": np.arange(0, 360, 10),  # directions
        "hs": "hs",  # significant wave height
        "tp": "tp",  # peak period
        "gamma": None,  # jonswap peak enhancement factor
        "dpt": None,  # water depth
        "lam": None,  # ochihubble peak enhancement factor
        "dp": "dp",  # peak direction
        "dspr": None,  # directional spread
    }  # fields used for reconstruction. It can be updated with fields of base_dset, numbers, or datarrays
    reconstruction_info = reconstruction_defaults.copy()
    reconstruction_info.update(spec_info)

    if base_dset is None:
        ds0 = xr.Dataset()
    elif isinstance(base_dset, str):
        if os.path.isfile(base_dset):
            ds0 = xr.open_dataset(base_dset)
        else:
            ds0 = xr.open_mfdataset(base_dset, combine="by_coords")
    else:
        ds0 = base_dset

    ds = ds0[[]]  # to keep metadata
    spc_coords = ("freq", "dir")
    for k, v in reconstruction_info.items():
        if isinstance(v, xr.DataArray) and k in spc_coords:
            ds[k] = v.values
        elif isinstance(v, (list, tuple)) and k not in spc_coords:
            ds[k] = xr.concat(
                [ds0[e] if isinstance(e, str) else xr.DataArray(e) for e in v],
                dim="part",
                coords="minimal",
            ).assign_coords({"part": range(len(v))})
        elif isinstance(v, str):
            ds[k] = ds0[v]
        elif v is None:
            if k in ds0:
                ds[k] = ds0[k]
        else:
            ds[k] = v
    return ds


def finite_depth(freqs, dpt):
    """Factors for modifiying JONSWAP spectra in shallow water (TMA spectrum)

    Arguments:
        freqs: frequencies
        dpt: water depth

    Returns:
        phi: factors between 0 and 1 for each frequency
    """
    w = 2 * np.pi * freqs
    whg = w * (dpt / 9.81) ** 0.5
    phi = w ** 0  # filled with ones
    phi[whg < 2] = 1 - 0.5 * (2 - whg[whg < 2]) ** 2
    phi[whg < 1] = 0.5 * whg[whg < 1] ** 2
    return phi


def calc_Sf_jonswap(freqs, hs, fp, gamma, dpt=None, sigmaA=0.07, sigmaB=0.09, alpha=-1):
    """ Reconstruct JONSWAP or TMA frequency spectra

    Arguments:
        freqs: frequencies
        hs: significant wave height
        fp: peak frequency
        gamma: jonswap peak enhancement factor
        dpt: water depth
        sigmaA, sigmaB: spectral width parameters
        alpha: normalization factor

    Returns:
        Sf: xarray dataarray with reconstructed frequency spectra
    """
    sigma = xr.where(freqs <= fp, sigmaA, sigmaB)
    r = np.exp(-((freqs - fp) ** 2.0) / (2 * sigma ** 2 * fp ** 2))
    Sf = 0.0617 * freqs ** (-5) * np.exp(-1.25 * (freqs / fp) ** (-4)) * gamma ** r
    if dpt is not None:
        Sf *= finite_depth(freqs, dpt)

    if alpha < 0:  # normalizing by integration
        alpha = (hs / Sf.spec.hs()) ** 2  # make sure m0=Hm0^2/16=int S(w)dw
    elif alpha == 0:  # original normalization for default values
        alpha = 5.061 * hs ** 2 * fp ** 4 * (1 - 0.287 * np.log(gamma))
    return (alpha * Sf).fillna(0)  # alpha>0 is applied directly


gamma_fun = (
    lambda x: np.sqrt(2.0 * np.pi / x)
    * ((x / np.exp(1.0)) * np.sqrt(x * np.sinh(1.0 / x))) ** x
)  # alternative to scipy.special.gamma


def calc_Sf_ochihubble(freqs, hs, fp, lam):
    """ Reconstruct OCHI-HUBBLE frequency spectra

    Arguments:
        freqs: frequencies
        hs: Significant wave height
        fp: peak frequency
        lam: ochihubble peak enhancement factor

    Returns:
        Sf: xarray dataarray with reconstructed frequency spectra
    """
    w = 2 * np.pi * freqs
    w0 = 2 * np.pi * fp
    B = xr.ufuncs.maximum(lam, 0.01) + 0.25
    A = 0.5 * np.pi * hs ** 2 * ((B * w0 ** 4) ** lam / gamma_fun(lam))
    a = xr.ufuncs.minimum((w0 / w) ** 4, 100.0)
    Sf = A * np.exp(-B * a) / (w ** (4.0 * B))
    return Sf.fillna(0)


def calc_Dth_cos2s(dirs, dp, dspr):
    """Cosine 2s spreading function.

    Arguments:
        dirs: direction coordinates
        dp: wave directions
        dspr: wave directional spreads

    Returns:
        Dth: normalized spreading

    Note:
        Function defined such that \int{Dth d\theta}=1*
    """
    th1 = 0.5 * np.deg2rad(dirs)
    th2 = 0.5 * np.deg2rad(dp)
    a = abs(
        np.cos(th1) * np.cos(th2) + np.sin(th1) * np.sin(th2)
    )  # cos(a-b) = cos(a)cos(b)+sin(a)sin(b)
    # Converting to cos2s spreading parameter
    # see Holthuijsen pag165
    s = (2./(dspr*np.pi/180)**2)-1
    Dth = a ** (2.0 * s)  # cos((dirs-dp)/2) ** (2*s)
    Dth /= Dth.sum("dir") * abs(dirs[1] - dirs[0])
    return Dth


def calc_Dth_normal(dirs, dp, dspr):
    """Normal distribution spreading

    Arguments:
        dirs: direction coordinates
        dp: wave directions
        dspr: wave directional spreads

    Returns:
        Dth: normalized spreading
    """
    ddif0 = abs(dirs % 360 - dp % 360)
    ddifmin = np.minimum(ddif0, 360 - ddif0)
    Dth = np.exp((-(ddifmin ** 2)) / (2 * dspr ** 2)) / (dspr * (2 * np.pi) ** 0.5)
    # TODO: wrapped normal but it's a bit pointless for real world dspr values
    Dth /= Dth.sum("dir") * abs(dirs[1] - dirs[0])
    return Dth


@xr.register_dataset_accessor("construct")
class SpecConstruct(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def Sf(self, stype=""):
        """ Wrapper for calc_Sf functions

        Arguments:
            stype: frequency spectrum type

        Returns:
            Sf: xarray dataarray with reconstructed frequency spectra
        """
        if not stype or stype == "jonswap":
            Sf = calc_Sf_jonswap(
                self._obj.freq,
                self._obj.hs,
                1 / self._obj.tp,
                self._obj.get("gamma", 3.3),
                self._obj.get("dpt", None),
            )
        elif stype == "ochihubble":
            Sf = calc_Sf_ochihubble(
                self._obj.freq, self._obj.hs, 1 / self._obj.tp, self._obj.lam
            )
        else:
            raise ValueError
        return Sf

    def Dth(self, dtype=""):
        """ Wrapper for calc_Dth functions

        Arguments:
            dtype: directionl distribution type

        Returns:
            Dth: normalized directional spreading
        """
        dspr = self._obj.get("dspr", 30)
        if not dtype or dtype == "cos2s":
            Dth = calc_Dth_cos2s(self._obj.dir, self._obj.dp, dspr)
        elif dtype == "normal":
            Dth = calc_Dth_normal(self._obj.dir, self._obj.dp, dspr)
        else:
            raise ValueError
        return Dth

    def efth(self, stype="", dtype="", sumdim="part"):
        """ Reconstruct directional spectra

        Arguments:
            stype: frequency spectrum type
            dtype: directionl distribution type
            sumdim: dimension to sum values
        Returns:
            efth: xarray dataarray with reconstructed frequency-direction spectra
        """
        efth = self.Sf(stype) * self.Dth(dtype)
        if sumdim in efth.coords:
            efth = efth.sum(dim=sumdim)
        return efth

    def to_dset(self, spec_info={}, **kwargs):
        """ Create wavespectra dataset

        Arguments:
            spec_info: dictionary for updating reconstruction defaults.

        Returns:
            ds: wavespectra dataset with reconstructed frequency-direction spectra
        """
        # TODO: Ensure that all arrays have wavespectra compatible names
        if spec_info:
            ds = prepare_reconstruction(spec_info, base_dset=self._obj)
        else:
            ds = self._obj.copy()
        ds[attrs.SPECNAME] = ds.construct.efth(**kwargs)
        set_spec_attributes(ds)
        return ds


if __name__ == "__main__":
    # Example1
    spec_info = {
        "hs": [1, 3],
        "tp": [5, 12],
        "gamma": 3.3,
        "dp": [10, 40],
        "dspr": [35, 25],
    }
    ds = prepare_reconstruction(spec_info).construct.to_dset()

    # # Example2
    # spec_info = {
    #     "hs": ["phs0", "phs1", "phs2"],
    #     "tp": ["ptp0", "ptp1", "ptp2"],
    #     "gamma": [1.0, 3.3, 3.3],
    #     "dp": ["pdir0", "pdir1", "pdir2"],
    #     "dspr": 30,
    # }
    # ds = xr.open_dataset(grdfile).construct.to_dset(spec_info)

    # # Example3
    # dstmp = xr.open_dataset(grdfile).isel(time=1, longitude=range(79, 82), latitude=62)
    # spec_info = {
    #     'hs': ["sea8hs", "sw8hs"],
    #     'tp': ["sea8tp", "sw8tp"],
    #     "lam": [1.54 * np.exp(-0.062 * dstmp.hs), 3.00],
    #     "dp": ["sea8dp", "sw8dp"],
    #     "dspr": [35, 25],
    # }
    # ds = dstmp.construct.to_dset(spec_info, stype="ochihubble", dtype="normal")
