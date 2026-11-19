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
    extensions: ClassVar[list[str]] = [".bin", ".dat", ".txt"]
    ndatasets: ClassVar[int | None] = None
    fn_out_template: ClassVar[str | None] = "EPIC_{:9d}_amc.fits"

    @classmethod
    def read(cls, fname, **kwargs):
        fname_str = str(fname)
        matches = re.findall(r"EPIC_0*([0-9]+)", basename(fname_str))
        epic = int(matches[0]) if matches else int(kwargs.get("epic", 0))
        data = np.loadtxt(fname_str, skiprows=1, ndmin=2)
        if data.size == 0 or data.shape[1] < 7:
            raise ValueError(
                f"AMC light-curve file {fname_str!r} must contain at least 1 row and 7 columns"
            )

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


_MAST_FILENAME_PATTERNS = (
    re.compile(r"ktwo0*([0-9]+)-c", re.IGNORECASE),
    re.compile(r"hlsp_k2sc_k2_llc_0*([0-9]+)-c", re.IGNORECASE),
    re.compile(r"kplr0*([0-9]+)-", re.IGNORECASE),
    re.compile(r"tess[0-9]+-s[0-9]+-0*([0-9]+)-", re.IGNORECASE),
    re.compile(r"C.._0*([0-9]+)_smear", re.IGNORECASE),
    re.compile(r"EPIC_0*([0-9]+)", re.IGNORECASE),
)
_MAST_OBJECT_PREFIX_RE = re.compile(
    r"(?:EPIC|KIC|TIC|KEPLERID)\s*[-_:]?\s*0*([0-9]+)", re.IGNORECASE
)
_SPATIAL_COLUMN_CANDIDATES = (
    ("pos_corr1", "pos_corr2"),
    ("mom_centr1", "mom_centr2"),
    ("centroid_col", "centroid_row"),
)
_MIN_SPATIAL_PAIRS = 15


def _resolve_mast_epic(fname_str: str, phead, dhead=None, fallback_epic=None) -> int:
    """Resolve target integer identifier from FITS headers first, then filename patterns."""
    for hdr in (phead, dhead):
        if hdr is None:
            continue
        for key in ("EPIC", "KEPLERID", "TICID"):
            if key in hdr:
                val = hdr[key]
                if isinstance(val, (int, np.integer)) and int(val) > 0:
                    return int(val)
                if isinstance(val, str):
                    s = val.strip()
                    if s.isdigit() and int(s) > 0:
                        return int(s)

    for hdr in (phead, dhead):
        if hdr is None or "OBJECT" not in hdr:
            continue
        obj_val = hdr["OBJECT"]
        if isinstance(obj_val, (int, np.integer)) and int(obj_val) > 0:
            return int(obj_val)
        obj_str = str(obj_val).strip()
        m_pref = _MAST_OBJECT_PREFIX_RE.search(obj_str)
        if m_pref:
            return int(m_pref.group(1))
        if obj_str.isdigit() and int(obj_str) > 0:
            return int(obj_str)
        m_long = re.search(r"([0-9]{5,})", obj_str)
        if m_long:
            return int(m_long.group(1))

    bname = basename(fname_str)
    for pattern in _MAST_FILENAME_PATTERNS:
        m = pattern.search(bname)
        if m:
            return int(m.group(1))

    if fallback_epic is not None:
        return int(fallback_epic)

    stem = splitext(bname)[0]
    digit_runs = re.findall(r"[0-9]+", stem)
    if digit_runs:
        long_runs = [d for d in digit_runs if len(d) >= 5]
        chosen = long_runs[0] if long_runs else max(digit_runs, key=len)
        return int(chosen)
    return 0


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

        with pf.open(fname_str) as hdul:
            phead = hdul[0].header.copy()
            dhead = hdul[1].header.copy()
            data = hdul[1].data

            with contextlib.suppress(KeyError, ValueError):
                for h in (phead, dhead):
                    if "CHECKSUM" in h:
                        h.remove("CHECKSUM")
                for k in ["CREATOR", "PROCVER", "FILEVER", "TIMVERSN"]:
                    if k in phead:
                        phead.remove(k)

            epic = _resolve_mast_epic(
                fname_str, phead, dhead, fallback_epic=kwargs.get("epic", None)
            )
            col_map = {c.lower(): c for c in data.columns.names}

            preferred_fkey = cls.fkeys[ftype]
            if preferred_fkey in col_map:
                fkey = preferred_fkey
            elif "flux" in col_map:
                fkey = "flux"
            else:
                alt_fkey = cls.fkeys["sap" if ftype == "pdc" else "pdc"]
                if alt_fkey in col_map:
                    fkey = alt_fkey
                else:
                    raise KeyError(
                        f"No supported flux column ({preferred_fkey!r} or 'flux') "
                        f"found in {fname_str!r}"
                    )

            time_arr = np.asarray(data[col_map["time"]], dtype=np.float64)
            flux_arr = np.asarray(data[col_map[fkey]], dtype=np.float64)

            err_key = f"{fkey}_err"
            if err_key in col_map:
                err_arr = np.asarray(data[col_map[err_key]], dtype=np.float64)
            elif "flux_err" in col_map:
                err_arr = np.asarray(data[col_map["flux_err"]], dtype=np.float64)
            else:
                err_arr = np.full(time_arr.shape, np.nan, dtype=np.float64)

            if "cadenceno" in col_map:
                cadence_arr = np.asarray(data[col_map["cadenceno"]])
            elif "cadence" in col_map:
                cadence_arr = np.asarray(data[col_map["cadence"]])
            else:
                cadence_arr = np.arange(time_arr.size, dtype=np.int32)

            if "sap_quality" in col_map:
                quality_arr = np.asarray(data[col_map["sap_quality"]])
            elif "quality" in col_map:
                quality_arr = np.asarray(data[col_map["quality"]])
            else:
                quality_arr = np.zeros(time_arr.size, dtype=np.int32)

            time_finite = np.isfinite(time_arr)
            n_time_finite = int(np.sum(time_finite))
            best_xy: tuple[np.ndarray, np.ndarray] | None = None
            best_count = -1
            for xk, yk in _SPATIAL_COLUMN_CANDIDATES:
                if xk in col_map and yk in col_map:
                    xc = np.asarray(data[col_map[xk]], dtype=np.float64)
                    yc = np.asarray(data[col_map[yk]], dtype=np.float64)
                    n_fin = int(np.sum(time_finite & np.isfinite(xc) & np.isfinite(yc)))
                    if n_fin >= _MIN_SPATIAL_PAIRS and n_fin >= 0.5 * n_time_finite:
                        best_xy = (xc, yc)
                        break
                    if n_fin > best_count:
                        best_count = n_fin
                        best_xy = (xc, yc)

            if best_xy is None:
                raise KeyError(
                    f"No supported spatial covariate columns (POS_CORR1/2, MOM_CENTR1/2, "
                    f"or CENTROID_COL/ROW) found in {fname_str!r}"
                )
            x_arr, y_arr = best_xy

        try:
            campaign = phead["campaign"]
        except KeyError:
            campaign = kwargs.get("campaign", None)

        return K2Data(
            epic,
            time=time_arr,
            cadence=cadence_arr,
            quality=quality_arr,
            fluxes=flux_arr,
            errors=err_arr,
            x=x_arr,
            y=y_arr,
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
            ttypes = {
                v.strip().upper()
                for k, v in h.items()
                if k.upper().startswith("TTYPE") and isinstance(v, str)
            }
            if "F_SCATTER" in ttypes or "TIME" not in ttypes:
                return False
            if ttypes & {"SAP_FLUX", "PDCSAP_FLUX"}:
                return True
            if "FLUX" in ttypes:
                return any(
                    {xk.upper(), yk.upper()} <= ttypes
                    for xk, yk in _SPATIAL_COLUMN_CANDIDATES
                )
            return False
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
            return any(
                isinstance(v, str) and v.upper() == "F_SCATTER" for v in h.values()
            )
        except (OSError, KeyError, IndexError, pf.VerifyError):
            return False


## ===  WRITERS  ===
## =================


def _format_fits_hps(pv, max_len: int = 68) -> str:
    """Format a hyperparameter vector compactly so FITS cards never exceed 80 columns."""
    arr = np.asarray(pv, dtype=np.float64).ravel()
    for brackets in (True, False):
        for prec in (7, 6, 5, 4):
            body = " ".join(f"{val:.{prec}g}" for val in arr)
            text = f"[{body}]" if brackets else body
            if len(text) <= max_len:
                return text
    return " ".join(f"{val:.3g}" for val in arr)


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
                    aup[nm] = np.where(np.isfinite(arr_np), arr_np, fill_value).astype(
                        np.int32
                    )
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
        hdu.header["EXTNAME"] = ("K2SC", "Extension name")
        hdu.header["OBJECT"] = (int(data.epic), "Target identifier")
        hdu.header["EPIC"] = (int(data.epic), "EPIC/Target identifier")
        splits_str = str(splits)
        hdu.header["SPLITS"] = (
            splits_str,
            "Spatial kernel split times" if len(splits_str) <= 38 else "",
        )
        for i in range(data.nsets):
            sid = "" if data.nsets == 1 else str(i + 1)
            hdu.header[f"cdpp{sid}r"] = float(dtres[i].cdpp_r)
            hdu.header[f"cdpp{sid}t"] = float(dtres[i].cdpp_t)
            hdu.header[f"cdpp{sid}c"] = float(dtres[i].cdpp_c)
            hdu.header[f"ap{sid}_warn"] = bool(dtres[i].warn)
        hdu.header["KER_NAME"] = (dtres[0].detrender.kernel.name, "GP kernel name")
        hdu.header["KER_PARS"] = (" ".join(dtres[0].detrender.kernel.names), "")
        hdu.header["KER_EQN"] = (dtres[0].detrender.kernel.eq, "")
        for i in range(data.nsets):
            hdu.header[f"KER_HPS{i + 1}"] = (
                _format_fits_hps(dtres[i].detrender.tr_pv),
                "",
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
            h["ORIGIN"] = ("SPLOX: Stars and Planets at Oxford", "Data origin")
            h["PROGRAM"] = ("k2SC v2.0", "Detrending pipeline")
            h["DATE"] = (now_str, "File creation date (UTC)")

        hdu_list = pf.HDUList([primary_hdu, hdu])
        hdu_list.writeto(str(fname), overwrite=True)


readers = [AMCReader, MASTReader, SPLOXReader]


def select_reader(fname):
    for R in readers:
        if R.can_read(fname):
            return R
    return None
