"""Module for reading K2 data from different sources and writing k2sc FITS files.

    This module contains several "readers" that can read a file containing a K2 light
    curve, and return a properly initialised K2Data instance. The module also contains
    a writer class for writing FITS files containing the K2 data and the detrending
    time series.

    Copyright (C) 2016  Suzanne Aigrain

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import contextlib
import re
from collections import namedtuple
from datetime import UTC, datetime
from os.path import basename, splitext
from typing import ClassVar

import astropy.io.fits as pf
import numpy as np

from .core import M_NOTFINITE
from .k2data import K2Data

## ===  READERS  ===
## =================


class DataReader:
    extensions: ClassVar[list[str]] = []
    ndatasets: ClassVar[int | None] = 0
    fn_out_template: ClassVar[str | None] = None

    def __init__(self, fname):
        raise NotImplementedError

    @classmethod
    def read(cls, fname, **kwargs):
        raise NotImplementedError

    @classmethod
    def can_read(cls, fname):
        raise NotImplementedError

    @classmethod
    def is_extension_valid(cls, fname):
        return splitext(basename(str(fname)))[1].lower() in cls.extensions


class AMCReader(DataReader):
    extensions: ClassVar[list[str]] = [".bin", ".dat"]
    ndatasets: ClassVar[int | None] = None
    fn_out_template: ClassVar[str | None] = "EPIC_{:9d}_amc.fits"

    @classmethod
    def read(cls, fname, **kwargs):
        fname_str = str(fname)
        matches = re.findall(r"EPIC_([0-9]+)_", basename(fname_str))
        epic = int(matches[0]) if matches else int(kwargs.get("epic", 0))
        data = np.loadtxt(fname_str, skiprows=1)

        return K2Data(
            epic,
            time=data[:, 0],
            cadence=data[:, 1],
            quality=data[:, -1],
            fluxes=data[:, 2:-3:2].T,
            errors=data[:, 3:-3:2].T,
            x=data[:, -3],
            y=data[:, -2],
            campaign=kwargs.get("campaign", None),
        )

    @classmethod
    def can_read(cls, fname):
        ext_ok = cls.is_extension_valid(fname)
        if not ext_ok:
            return False
        try:
            with open(fname, "r", encoding="utf-8", errors="replace") as f:
                header = f.readline().lower().split()
                head_ok = all(
                    cn in header
                    for cn in ["dates", "cadences", "xpos", "ypos", "quality"]
                )
        except OSError:
            return False
        return ext_ok and head_ok


class MASTReader(DataReader):
    extensions: ClassVar[list[str]] = [".fits", ".fit"]
    ndatasets: ClassVar[int | None] = 1
    fn_out_template: ClassVar[str | None] = "EPIC_{:9d}_mast.fits"
    allowed_types: ClassVar[list[str]] = ["sap", "pdc"]
    fkeys: ClassVar[dict[str, str]] = {"sap": "sap_flux", "pdc": "pdcsap_flux"}

    @classmethod
    def read(cls, fname, sid=0, **kwargs):
        fname_str = str(fname)
        ftype = kwargs.get("type", "pdc").lower()
        assert ftype in cls.allowed_types, "Flux type must be either `sap` or `pdc`"
        fkey = cls.fkeys[ftype]

        try:
            epic = int(re.findall(r"ktwo([0-9]+)-c", basename(fname_str))[0])
        except IndexError:
            matches = re.findall(r"C.._([0-9]+)_smear", basename(fname_str))
            epic = int(matches[0]) if matches else int(kwargs.get("epic", 0))

        data = pf.getdata(fname_str, 1)
        phead = pf.getheader(fname_str, 0)
        dhead = pf.getheader(fname_str, 1)

        with contextlib.suppress(KeyError, ValueError):
            for h in (phead, dhead):
                if "CHECKSUM" in h:
                    h.remove("CHECKSUM")
            for k in ["CREATOR", "PROCVER", "FILEVER", "TIMVERSN"]:
                if k in phead:
                    phead.remove(k)

        try:
            campaign = phead["campaign"]
        except KeyError:
            campaign = kwargs.get("campaign", None)

        return K2Data(
            epic,
            time=data["time"],
            cadence=data["cadenceno"],
            quality=data["sap_quality"],
            fluxes=data[fkey],
            errors=data[fkey + "_err"],
            x=data["pos_corr1"],
            y=data["pos_corr2"],
            primary_header=phead,
            data_header=dhead,
            campaign=campaign,
        )

    @classmethod
    def can_read(cls, fname):
        ext_ok = cls.is_extension_valid(fname)
        if not ext_ok:
            return False
        try:
            h = pf.getheader(str(fname), 1)
            return any(isinstance(v, str) and v.upper() == "SAP_FLUX" for v in h.values())
        except (OSError, KeyError, IndexError, pf.VerifyError):
            return False


class SPLOXReader(DataReader):
    extensions: ClassVar[list[str]] = [".fits", ".fit"]
    ndatasets: ClassVar[int | None] = 6
    fn_out_template: ClassVar[str | None] = "STAR_{:09d}.fits"
    _cache = None
    _nstars = None

    @classmethod
    def read(cls, fname, sid=0, **kwargs):
        fname_str = str(fname)
        cache = namedtuple(
            "K2Cache",
            "fname objno nobj nexp time cadence quality fluxes errors x y header",
        )
        if not cls._cache or cls._cache.fname != fname_str:
            with pf.open(fname_str) as fin:
                data = fin[1].data
                nobj = fin[1].header["naxis2"]
                nexp = fin[2].header["naxis2"]
                fluxes = (1.0 + data["f"].reshape([nobj, nexp, -1])) * data["f_med"][
                    :, np.newaxis, :
                ]
                fluxes = np.swapaxes(fluxes, 1, 2)
                cls._nstars = nobj
                cls._cache = cache(
                    fname_str,
                    fin[1].data["objno"],
                    nobj,
                    nexp,
                    fin[2].data["mjd_obs"],
                    fin[2].data["cadence"],
                    np.zeros(nexp, dtype=np.int32),
                    fluxes,
                    np.zeros_like(fluxes),
                    data["x"],
                    data["y"],
                    fin[0].header,
                )

        return K2Data(
            cls._cache.objno[sid],
            time=cls._cache.time[:-1],
            cadence=cls._cache.cadence[:-1],
            quality=cls._cache.quality[:-1],
            fluxes=cls._cache.fluxes[sid, :, :-1],
            errors=cls._cache.errors[sid, :, :-1],
            x=cls._cache.x[sid, :-1],
            y=cls._cache.y[sid, :-1],
            primary_header=cls._cache.header,
            data_header=cls._cache.header,
            campaign=kwargs.get("campaign", None),
        )

    @classmethod
    def nstars(cls, fname):
        return pf.getval(str(fname), "naxis2", 1)

    @classmethod
    def can_read(cls, fname):
        ext_ok = cls.is_extension_valid(fname)
        if not ext_ok:
            return False
        try:
            h = pf.getheader(str(fname), 1)
            return any(isinstance(v, str) and v.upper() == "F_SCATTER" for v in h.values())
        except (OSError, KeyError, IndexError, pf.VerifyError):
            return False


## ===  WRITERS  ===
## =================


class FITSWriter:
    @classmethod
    def write(cls, fname, splits, data: K2Data, dtres):
        nm = np.asarray(data.nanmask, dtype=bool)
        n_full = int(nm.size)

        def unpack_float(arr) -> np.ndarray:
            arr_np = np.asarray(arr, dtype=np.float64)
            if arr_np.size == n_full and n_full != data.npoints:
                aup = arr_np.copy()
                aup[~nm] = np.nan
                return aup
            aup = np.full(n_full, np.nan, dtype=np.float64)
            aup[nm] = arr_np
            return aup

        def unpack_int(arr, raw_arr=None, fill_value: int = 0) -> np.ndarray:
            if raw_arr is not None and np.asarray(raw_arr).size == n_full:
                raw_np = np.asarray(raw_arr)
                if np.issubdtype(raw_np.dtype, np.floating):
                    aup = np.where(np.isfinite(raw_np), raw_np, fill_value).astype(
                        np.int32
                    )
                else:
                    aup = raw_np.astype(np.int32, copy=True)
            else:
                aup = np.full(n_full, fill_value, dtype=np.int32)

            arr_np = np.asarray(arr)
            if arr_np.size == data.npoints:
                if np.issubdtype(arr_np.dtype, np.floating):
                    aup[nm] = np.where(
                        np.isfinite(arr_np), arr_np, fill_value
                    ).astype(np.int32)
                else:
                    aup[nm] = arr_np.astype(np.int32)
            elif arr_np.size == n_full:
                if np.issubdtype(arr_np.dtype, np.floating):
                    aup = np.where(np.isfinite(arr_np), arr_np, fill_value).astype(
                        np.int32
                    )
                else:
                    aup = arr_np.astype(np.int32, copy=True)
            return aup

        def unpack_flags(arr) -> np.ndarray:
            arr_np = np.asarray(arr, dtype=np.uint8)
            if arr_np.size == n_full and n_full != data.npoints:
                aup = arr_np.copy()
                aup[~nm] |= M_NOTFINITE
                return aup
            aup = np.full(n_full, M_NOTFINITE, dtype=np.uint8)
            aup[nm] = arr_np
            return aup

        C = pf.Column
        cols = [
            C(name="time", format="D", array=unpack_float(data.time)),
            C(
                name="cadence",
                format="J",
                array=unpack_int(
                    data.cadence,
                    raw_arr=getattr(data, "raw_cadence", None),
                    fill_value=-1,
                ),
            ),
            C(
                name="quality",
                format="J",
                array=unpack_int(
                    data.quality,
                    raw_arr=getattr(data, "raw_quality", None),
                    fill_value=0,
                ),
            ),
            C(name="x", format="D", array=unpack_float(data.x)),
            C(name="y", format="D", array=unpack_float(data.y)),
        ]

        for i in range(data.nsets):
            sid = "" if data.nsets == 1 else f"_{i + 1}"
            cols.extend(
                [
                    C(
                        name=f"flux{sid}",
                        format="D",
                        array=unpack_float(data.fluxes[i]),
                    ),
                    C(
                        name=f"error{sid}",
                        format="D",
                        array=unpack_float(data.errors[i]),
                    ),
                    C(
                        name=f"mflags{sid}",
                        format="B",
                        array=unpack_flags(data.mflags[i]),
                    ),
                    C(
                        name=f"trtime{sid}",
                        format="D",
                        array=unpack_float(dtres[i].tr_time),
                    ),
                    C(
                        name=f"trposi{sid}",
                        format="D",
                        array=unpack_float(dtres[i].tr_position),
                    ),
                ]
            )

        data_header = data.data_header.copy() if data.data_header is not None else None
        hdu = pf.BinTableHDU.from_columns(pf.ColDefs(cols), header=data_header)
        hdu.header["extname"] = "K2SC"
        hdu.header["object"] = int(data.epic)
        hdu.header["epic"] = int(data.epic)
        hdu.header["splits"] = str(splits)
        for i in range(data.nsets):
            sid = "" if data.nsets == 1 else str(i + 1)
            hdu.header[f"cdpp{sid}r"] = float(dtres[i].cdpp_r)
            hdu.header[f"cdpp{sid}t"] = float(dtres[i].cdpp_t)
            hdu.header[f"cdpp{sid}c"] = float(dtres[i].cdpp_c)
            hdu.header[f"ap{sid}_warn"] = bool(dtres[i].warn)
        hdu.header["ker_name"] = dtres[0].detrender.kernel.name
        hdu.header["ker_pars"] = " ".join(dtres[0].detrender.kernel.names)
        hdu.header["ker_eqn"] = dtres[0].detrender.kernel.eq
        for i in range(data.nsets):
            hdu.header[f"ker_hps{i + 1}"] = str(dtres[i].detrender.tr_pv).replace(
                "\n", ""
            )
        if len(dtres) > 0 and hasattr(dtres[0], "pca_transform"):
            pca_tf = dtres[0].pca_transform
            hdu.header["ROLL_PCA"] = (
                pca_tf is not None,
                "Spatial GP fitted in PCA roll coordinates",
            )
            if pca_tf is not None:
                hdu.header["PCA_CX"] = (
                    float(pca_tf.center_xy[0]),
                    "Roll PCA x centre [pix]",
                )
                hdu.header["PCA_CY"] = (
                    float(pca_tf.center_xy[1]),
                    "Roll PCA y centre [pix]",
                )
                hdu.header["PCA_R11"] = (
                    float(pca_tf.rotation[0, 0]),
                    "Roll PCA rotation R[0,0]",
                )
                hdu.header["PCA_R12"] = (
                    float(pca_tf.rotation[0, 1]),
                    "Roll PCA rotation R[0,1]",
                )
                hdu.header["PCA_R21"] = (
                    float(pca_tf.rotation[1, 0]),
                    "Roll PCA rotation R[1,0]",
                )
                hdu.header["PCA_R22"] = (
                    float(pca_tf.rotation[1, 1]),
                    "Roll PCA rotation R[1,1]",
                )
                hdu.header["PCA_S1"] = (
                    float(pca_tf.scales_u[0]),
                    "Roll PCA u1 4-MAD scale [pix]",
                )
                hdu.header["PCA_S2"] = (
                    float(pca_tf.scales_u[1]),
                    "Roll PCA u2 4-MAD scale [pix]",
                )
            else:
                for _pca_key in (
                    "PCA_CX",
                    "PCA_CY",
                    "PCA_R11",
                    "PCA_R12",
                    "PCA_R21",
                    "PCA_R22",
                    "PCA_S1",
                    "PCA_S2",
                ):
                    hdu.header.pop(_pca_key, None)

        primary_header = (
            data.primary_header.copy() if data.primary_header is not None else None
        )
        primary_hdu = pf.PrimaryHDU(header=primary_header)
        now_str = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S")
        for h in (primary_hdu.header, hdu.header):
            h["origin"] = "SPLOX: Stars and Planets at Oxford"
            h["program"] = "k2SC v2.0"
            h["date"] = now_str

        hdu_list = pf.HDUList([primary_hdu, hdu])
        hdu_list.writeto(str(fname), overwrite=True)


readers = [AMCReader, MASTReader, SPLOXReader]


def select_reader(fname):
    for R in readers:
        if R.can_read(fname):
            return R
    return None
