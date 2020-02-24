import os

import numpy as np
import xarray as xr

from wavespectra.core.attributes import attrs, set_spec_attributes
from wavespectra.specdataset import SpecDataset


def prepare_reconstruction(spec_info, base_dset=None):
    """ Load parameters for spectral reconstruction

    Args:
        spec_info: dictionary for updating reconstruction defaults. Optionally extra variables to keep.
        base_dset: path or xarray dataset object

    Returns:
        ds: xarray dset with spectral parameters
    """
    reconstruction_defaults = {  # fields of base_dset, numbers, or datarrays
        "freq": np.arange(0.04, 1.0, 0.02),
        "dir": np.arange(0, 360, 10),
        "hs": "hs",
        "tp": "tp",
        "gamma": 3.3,
        "dp": "dp",
        "dspr": 20,
    }
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

    ds = ds0[[]]
    if "part" in ds0.coords:
        ds["part"] = ds0["part"]

    spc_coords = ("freq", "dir")
    for k, v in reconstruction_info.items():
        if isinstance(v, xr.DataArray) and k in spc_coords:
            ds[k] = v.values
        elif isinstance(v, (list, tuple)) and k not in spc_coords:
            if all(isinstance(e, str) for e in v):
                darr = ds0[v].to_array(dim="part")
            else:
                darr = xr.DataArray(data=v, dims="part")
            ds[k] = darr.assign_coords({"part": range(len(v))})
        elif isinstance(v, str):
            ds[k] = ds0[v]
        else:
            ds[k] = v
    return ds


def finite_depth(freqs, dpt):
    """Factors for modifiying JONSWAP spectra in shallow water (TMA spectrum)

    Args:
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


def calc_Sf(freqs, hs, fp, gamma=3.3, sigmaA=0.07, sigmaB=0.09, dpt=None, alpha=-1):
    """ Reconstruct JONSWAP or TMA frequency spectra

    Args:
        freqs: frequencies
        hs: Significant wave height
        fp: peak frequency
        gamma: peak enhancement factor
        sigmaA, sigmaB: spectral width parameters
        dpt: water depth
        alpha: normalization factor

    Returns:
        Sf: xarray dataarray with reconstructed spectra
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


def calc_Dth(dirs, dp, dspr):
    """Cosine 2s spreading function.
    
    Args:
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
    Dth = a ** (2.0 * dspr)  # cos((dirs-dp)/2) ** (2*dspr)
    Dth /= Dth.sum("dir") * abs(dirs[1] - dirs[0])
    return Dth


@xr.register_dataset_accessor("construct")
class SpecConstruct(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def Sf(self):
        """ Wrapper for calc_Sf
        """
        dpt = self._obj.dpt if "dpt" in self._obj else None
        Sf = calc_Sf(
            self._obj.freq, self._obj.hs, 1 / self._obj.tp, self._obj.gamma, dpt=dpt
        )
        return Sf

    def Dth(self):
        """ Wrapper for calc_Dth
        """
        Dth = calc_Dth(self._obj.dir, self._obj.dp, self._obj.dspr)
        return Dth

    def efth(self, sumdim="part"):
        """ Reconstruct directional spectra

        Args:
            ds: xarray dataset with reconstruction parameters
            sumdim: dimension to sum values
        Returns:
            efth: xarray dataarray with reconstructed spectra
        """
        efth = self.Sf() * self.Dth()
        if sumdim in efth.coords:
            efth = efth.sum(dim=sumdim)
        return efth

    def to_dset(self, spec_info={}):
        """ Create wavespectra dataset
        """
        # TODO: Ensure that all arrays have wavespectra compatible names
        ds = prepare_reconstruction(spec_info, base_dset=self._obj)
        ds[attrs.SPECNAME] = ds.construct.efth()
        set_spec_attributes(ds)
        return ds


if __name__ == "__main__":
    # Example1
    spec_info = {
        "hs": [1, 3],
        "tp": [5, 12],
        "gamma": 3.3,
        "dp": [10, 40],
        "dspr": [30, 15],
    }
    ds1 = prepare_reconstruction(spec_info).construct.to_dset()

    # Example2
    # spec_info = {
    #     'hs': ["phs0", "phs1", "phs2"],
    #     'tp': ["ptp0", "ptp1", "ptp2"],
    #     "gamma": [1.0, 3.3, 3.3],
    #     "dp": ["pdir0", "pdir1", "pdir2"],
    #     "dspr": 30,
    #     }
    # ds2 = xr.open_dataset(grdfile).isel(time=1, longitude=range(5, 8)).construct.to_dset(spec_info)
