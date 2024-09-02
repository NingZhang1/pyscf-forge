#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

############ sys module ############

import numpy, scipy
import numpy as np
import ctypes

############ pyscf module ############

from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact

############ isdf backends ############

import pyscf.isdf.BackEnd.isdf_backend as BACKEND
from pyscf.isdf.BackEnd.isdf_memory_allocator import SimpleMemoryAllocator

USE_GPU = BACKEND.USE_GPU

NUM_THREADS = BACKEND.NUM_THREADS
FLOAT64 = BACKEND.FLOAT64
TENSORTy = BACKEND.TENSORTy
ToNUMPY = BACKEND._toNumpy
ToTENSOR = BACKEND._toTensor
MALLOC = BACKEND._malloc
ZEROS = BACKEND._zeros
MAX = BACKEND._maximum
ABS = BACKEND._absolute
DOT = BACKEND._dot
TAKE = BACKEND._take
EINSUM_IK_JK_IJK = BACKEND._einsum_ik_jk_ijk
CLEAN = BACKEND._clean
ADD_TRANSPOSE_ = BACKEND._add_transpose_

############ isdf utils ############

import pyscf.isdf.misc as misc
from pyscf.isdf.isdf_tools_local import aoR_Holder

# utils #


def _get_moR(aoR, mo_coeff):
    if isinstance(aoR, TENSORTy):
        return DOT(ToTENSOR(mo_coeff.T), aoR)
    else:
        assert isinstance(aoR, list)
        nmo = mo_coeff.shape[1]

        buffer = MALLOC(mo_coeff.shape, FLOAT64, gpu=USE_GPU)
        res = []
        for _aoR_ in aoR:
            nao_involved = _aoR_.aoR.shape[0]
            ngrids_involved = _aoR_.aoR.shape[1]
            mo_coeff_packed = MALLOC(
                (nao_involved, nmo), FLOAT64, buf=buffer, offset=0, gpu=USE_GPU
            )
            TAKE(mo_coeff, _aoR_.ao_involved, 0, out=mo_coeff_packed)

            moR = DOT(mo_coeff_packed.T, _aoR_.aoR)
            res.append(
                aoR_Holder(
                    moR,
                    ToTENSOR(np.arange(nmo).astype(np.int64)),
                    _aoR_.global_gridID_begin,
                    _aoR_.global_gridID_end,
                )
            )
        del buffer
        return res


def isdf_eri(mydf, mo_coeff=None, with_robust_fitting=None, AOPAIR_BLKSIZE=2e9):
    """
    Perform AO2MO transformation from ISDF with robust fitting with s4 symmetry
    """

    if with_robust_fitting is None:
        with_robust_fitting = mydf.with_robust_fitting

    t1 = (logger.process_clock(), logger.perf_counter())

    if mo_coeff is None:
        IsAOERI = True
    else:
        mo_coeff = ToTENSOR(mo_coeff)
        IsAOERI = False

    # fetch moR #

    if IsAOERI:
        moRg = mydf.aoRg
        moR = mydf.aoR
        nmo = mydf.nao
    else:
        moRg = _get_moR(mydf.aoRg, mo_coeff)
        nmo = mo_coeff.shape[1]
        if with_robust_fitting:
            moR = _get_moR(mydf.aoR, mo_coeff)
        else:
            moR = None

    nnmo = nmo * (nmo + 1) // 2
    res = ZEROS((nnmo, nnmo), cpu=not USE_GPU)

    # allocate buffer #

    GRID_BUNCHIZE = int(AOPAIR_BLKSIZE // nnmo // 5 * 8)
    if with_robust_fitting:
        GRID_BUNCHIZE = min(GRID_BUNCHIZE, mydf.ngrids)
    else:
        GRID_BUNCHIZE = min(GRID_BUNCHIZE, mydf.naux)

    size_max = (nmo**2 * GRID_BUNCHIZE) + nnmo * GRID_BUNCHIZE
    size_max += nnmo * GRID_BUNCHIZE * 2

    buffer = SimpleMemoryAllocator(size_max, FLOAT64, gpu=USE_GPU)

    row_indices, col_indices = np.tril_indices(nmo)
    row_indices = ToTENSOR(np.array(row_indices, dtype=np.int64))
    col_indices = ToTENSOR(np.array(col_indices, dtype=np.int64))
    indices = row_indices * nmo + col_indices

    # perform the contraction bunch by bunch #

    # V term #

    if with_robust_fitting:
        V = mydf.V
        for p0, p1 in lib.prange(0, mydf.naux, GRID_BUNCHIZE):
            moPairRgBra = buffer.malloc((nnmo, p1 - p0), name="moPairRg")
            moPairRg2 = buffer.malloc((nmo, nmo, p1 - p0), name="moPairRg2")
            EINSUM_IK_JK_IJK(moRg[:, p0:p1], moRg[:, p0:p1], out=moPairRg2)
            moPairRg2 = moPairRg2.reshape((nmo * nmo, p1 - p0))
            TAKE(moPairRg2, indices, 0, out=moPairRgBra)
            buffer.free(count=1)
            moPairRVKet = buffer.malloc((nnmo, p1 - p0), name="moPairRV")
            CLEAN(moPairRVKet)
            for q0, q1 in lib.prange(0, mydf.ngrids, GRID_BUNCHIZE):
                moPairRKet = buffer.malloc((nnmo, q1 - q0), name="moPairRKet")
                moPairR2 = buffer.malloc((nmo, nmo, q1 - q0), name="moPairR2")
                EINSUM_IK_JK_IJK(moR[:, q0:q1], moR[:, q0:q1], out=moPairR2)
                moPairR2 = moPairR2.reshape((nmo * nmo, q1 - q0))
                TAKE(moPairR2, indices, 0, out=moPairRKet)
                buffer.free(count=1)
                DOT(moPairRKet, V[p0:p1, q0:q1].T, c=moPairRVKet, beta=1)
                buffer.free(count=1)

            DOT(moPairRgBra, moPairRVKet.T, c=res, beta=1)
            buffer.free(count=2)

        ADD_TRANSPOSE_(res)

    # W term #

    sign = 1 if not with_robust_fitting else -1

    W = mydf.W
    for p0, p1 in lib.prange(0, mydf.naux, GRID_BUNCHIZE):
        moPairRgBra = buffer.malloc((nnmo, p1 - p0), name="moPairRg")
        moPairRg2 = buffer.malloc((nmo, nmo, p1 - p0), name="moPairRg2")
        EINSUM_IK_JK_IJK(moRg[:, p0:p1], moRg[:, p0:p1], out=moPairRg2)
        moPairRg2 = moPairRg2.reshape((nmo * nmo, p1 - p0))
        TAKE(moPairRg2, indices, 0, out=moPairRgBra)
        buffer.free(count=1)
        moPairRgWKet = buffer.malloc((nnmo, p1 - p0), name="moPairRgWKet")
        CLEAN(moPairRgWKet)
        for q0, q1 in lib.prange(0, mydf.naux, GRID_BUNCHIZE):
            moPairRgKet = buffer.malloc((nnmo, q1 - q0), name="moPairRg")
            moPairRg2 = buffer.malloc((nmo, nmo, q1 - q0), name="moPairRg2")
            EINSUM_IK_JK_IJK(moRg[:, q0:q1], moRg[:, q0:q1], out=moPairRg2)
            moPairRg2 = moPairRg2.reshape((nmo * nmo, q1 - q0))
            TAKE(moPairRg2, indices, 0, out=moPairRgKet)
            buffer.free(count=1)
            DOT(moPairRgKet, W[p0:p1, q0:q1].T, c=moPairRgWKet, beta=1)
            buffer.free(count=1)

        DOT(moPairRgBra, moPairRgWKet.T, c=res, beta=1, alpha=sign)
        buffer.free(count=2)

    del buffer

    t2 = (logger.process_clock(), logger.perf_counter())

    misc._benchmark_time(t1, t2, "isdf_eri", mydf, mydf.rank)

    return ToNUMPY(res) * mydf.ngrids / mydf.cell.vol


def isdf_eri_ovov(
    mydf,
    mo_coeff_o: np.ndarray = None,
    mo_coeff_v: np.ndarray = None,
    with_robust_fitting=None,
    AOPAIR_BLKSIZE=2e9,
):
    """
    Perform AO2MO transformation from ISDF for specific orbital types (ovov), for MP2 calculation

    Args:
        mydf       : ISDF objects.
        mo_coeff_o : Molecular orbital coefficients for occupied orbitals
        mo_coeff_v : Molecular orbital coefficients for virtual  orbitals

    Return:
        eri : ovov part of MO-ERI

    """
    if with_robust_fitting is None:
        with_robust_fitting = mydf.with_robust_fitting

    t1 = (logger.process_clock(), logger.perf_counter())

    if with_robust_fitting:
        moR_o = _get_moR(mydf.aoR, mo_coeff_o)
        moR_v = _get_moR(mydf.aoR, mo_coeff_v)
    else:
        moR_o = None
        moR_v = None

    moRg_o = _get_moR(mydf.aoRg, mo_coeff_o)
    moRg_v = _get_moR(mydf.aoRg, mo_coeff_v)

    nmo_o = mo_coeff_o.shape[1]
    nmo_v = mo_coeff_v.shape[1]

    nmo_ov = nmo_o * nmo_v

    res = ZEROS((nmo_ov, nmo_ov), cpu=not USE_GPU)

    # allocate buffer #

    GRID_BUNCHIZE = int(AOPAIR_BLKSIZE // nmo_ov // 5 * 8)
    if with_robust_fitting:
        GRID_BUNCHIZE = min(GRID_BUNCHIZE, mydf.ngrids)
    else:
        GRID_BUNCHIZE = min(GRID_BUNCHIZE, mydf.naux)

    size_max = nmo_ov * GRID_BUNCHIZE * 4
    buffer = SimpleMemoryAllocator(size_max, FLOAT64, gpu=USE_GPU)

    # perform the contraction bunch by bunch #

    # V term #

    if with_robust_fitting:
        V = mydf.V
        for p0, p1 in lib.prange(0, mydf.naux, GRID_BUNCHIZE):
            moPairRgOVBra = buffer.malloc((nmo_o, nmo_v, p1 - p0), name="moPairRgOVBra")
            EINSUM_IK_JK_IJK(moRg_o[:, p0:p1], moRg_v[:, p0:p1], out=moPairRgOVBra)
            moPairRgOVBra = moPairRgOVBra.reshape((nmo_ov, p1 - p0))
            moPairRVOVKet = buffer.malloc((nmo_ov, p1 - p0), name="moPairRVOVKet")
            CLEAN(moPairRVOVKet)
            for q0, q1 in lib.prange(0, mydf.ngrids, GRID_BUNCHIZE):
                moPairRKet = buffer.malloc((nmo_o, nmo_v, q1 - q0), name="moPairRKet")
                EINSUM_IK_JK_IJK(moR_o[:, q0:q1], moR_v[:, q0:q1], out=moPairRKet)
                moPairRKet = moPairRKet.reshape((nmo_ov, q1 - q0))
                DOT(moPairRKet, V[p0:p1, q0:q1].T, c=moPairRVOVKet, beta=1)
                buffer.free(count=1)

            DOT(moPairRgOVBra, moPairRVOVKet.T, c=res, beta=1)
            buffer.free(count=2)

        ADD_TRANSPOSE_(res)

    # W term #

    sign = 1 if not with_robust_fitting else -1

    W = mydf.W
    for p0, p1 in lib.prange(0, mydf.naux, GRID_BUNCHIZE):
        moPairRgOVBra = buffer.malloc((nmo_o, nmo_v, p1 - p0), name="moPairRgOVBra")
        EINSUM_IK_JK_IJK(moRg_o[:, p0:p1], moRg_v[:, p0:p1], out=moPairRgOVBra)
        moPairRgOVBra = moPairRgOVBra.reshape((nmo_ov, p1 - p0))
        moPairRgOVKet = buffer.malloc((nmo_ov, p1 - p0), name="moPairRgOVKet")
        CLEAN(moPairRgOVKet)
        for q0, q1 in lib.prange(0, mydf.naux, GRID_BUNCHIZE):
            moPairRgKet = buffer.malloc((nmo_o, nmo_v, q1 - q0), name="moPairRgKet")
            EINSUM_IK_JK_IJK(moRg_o[:, q0:q1], moRg_v[:, q0:q1], out=moPairRgKet)
            moPairRgKet = moPairRgKet.reshape((nmo_ov, q1 - q0))
            DOT(moPairRgKet, W[p0:p1, q0:q1].T, c=moPairRgOVKet, beta=1)
            buffer.free(count=1)

        DOT(moPairRgOVBra, moPairRgOVKet.T, c=res, beta=1, alpha=sign)
        buffer.free(count=2)

    del buffer

    t2 = (logger.process_clock(), logger.perf_counter())

    misc._benchmark_time(t1, t2, "isdf_eri_ovov", mydf, mydf.rank)

    return (
        ToNUMPY(res.reshape((nmo_o, nmo_v, nmo_o, nmo_v))) * mydf.ngrids / mydf.cell.vol
    )


def get_eri(
    mydf,
    kpts=None,
    compact=getattr(__config__, "pbc_df_ao2mo_get_eri_compact", True),
    with_robust_fitting=None,
    AOPAIR_BLKSIZE=2e9,
):
    cell = mydf.cell
    nao = cell.nao_nr()
    kptijkl = _format_kpts(kpts)
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(
            cell,
            "isdf_ao2mo: momentum conservation not found in " "the given k-points %s",
            kptijkl,
        )
        return numpy.zeros((nao, nao, nao, nao))

    if gamma_point(kptijkl):

        eri = isdf_eri(
            mydf, with_robust_fitting=with_robust_fitting, AOPAIR_BLKSIZE=AOPAIR_BLKSIZE
        )

        if compact:
            return eri
        else:
            return ao2mo.restore(1, eri, nao)
    else:
        raise NotImplementedError("k-point dependent ISDF AO2MO is not implemented yet")


def general(
    mydf,
    mo_coeffs,
    kpts=None,
    compact=getattr(__config__, "pbc_df_ao2mo_general_compact", True),
    with_robust_fitting=None,
    AOPAIR_BLKSIZE=2e9,
):
    from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri

    warn_pbc2d_eri(mydf)
    cell = mydf.cell
    nao = cell.nao_nr()
    kptijkl = _format_kpts(kpts)
    kpti, kptj, kptk, kptl = kptijkl
    if isinstance(mo_coeffs, numpy.ndarray) and mo_coeffs.ndim == 2:
        mo_coeffs = (mo_coeffs,) * 4
    mo_coeffs = [numpy.asarray(mo, order="F") for mo in mo_coeffs]
    if not _iskconserv(cell, kptijkl):
        lib.logger.warn(
            cell,
            "isdf_ao2mo: momentum conservation not found in " "the given k-points %s",
            kptijkl,
        )
        return numpy.zeros([mo.shape[1] for mo in mo_coeffs])

    allreal = not any(numpy.iscomplexobj(mo) for mo in mo_coeffs)

    if gamma_point(kptijkl) and allreal:
        if (
            iden_coeffs(mo_coeffs[0], mo_coeffs[1])
            and iden_coeffs(mo_coeffs[0], mo_coeffs[2])
            and iden_coeffs(mo_coeffs[0], mo_coeffs[3])
        ):
            ### MO-ERI ###

            eri = isdf_eri(
                mydf,
                ToNUMPY(mo_coeffs[0]).copy(),
                with_robust_fitting=False,
                AOPAIR_BLKSIZE=AOPAIR_BLKSIZE,
            )

            if compact:
                return eri
            else:
                return ao2mo.restore(1, eri, nao)
        else:
            #### ovov MO-ERI ####

            if iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and iden_coeffs(
                mo_coeffs[1], mo_coeffs[3]
            ):
                eri_ovov = isdf_eri_ovov(
                    mydf,
                    ToNUMPY(mo_coeffs[0]).copy(),
                    ToNUMPY(mo_coeffs[1]).copy(),
                    with_robust_fitting=with_robust_fitting,
                    AOPAIR_BLKSIZE=AOPAIR_BLKSIZE,
                )
                if compact:
                    print("compact is not supported in general with ov ov mode")
                    return eri_ovov
                else:
                    return eri_ovov
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError("k-point dependent ISDF AO2MO is not implemented yet")
