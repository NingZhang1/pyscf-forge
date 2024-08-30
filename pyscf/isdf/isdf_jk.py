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

import copy
import numpy as np
import numpy
import ctypes
from functools import partial

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point

############ isdf backends ############

import pyscf.isdf.BackEnd.isdf_backend as BACKEND

USE_GPU = BACKEND.USE_GPU

NUM_THREADS = BACKEND.NUM_THREADS
FLOAT64 = BACKEND.FLOAT64
ToNUMPY = BACKEND._toNumpy
ToTENSOR = BACKEND._toTensor
MAX = BACKEND._maximum
ABS = BACKEND._absolute
DOT = BACKEND._dot

############ isdf utils ############

import pyscf.isdf.misc as misc

##################################################
#
# only Gamma Point
#
##################################################


def _get_j_dm(mydf, dm, use_mpi=False):
    assert not use_mpi, "ISDF does not support MPI"

    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    # funcs #

    EINSUM_IJ_IJ_J = BACKEND._einsum_ij_ij_j
    EINSUM_IJ_J_IJ = BACKEND._einsum_ij_j_ij
    FFTN = partial(BACKEND._fftn, threads=NUM_THREADS)
    IFFTN = partial(BACKEND._ifftn, threads=NUM_THREADS)
    REAL = BACKEND._real
    IMAG = BACKEND._imag

    # step 0 info #

    buffer = mydf.buffer
    mesh = tuple(mydf.mesh)

    #### step 1. get density value on real space ####

    buf1 = buffer.malloc((mydf.nao, mydf.ngrids), dtype=FLOAT64, name="buf1")
    DOT(dm, mydf.aoR, c=buf1)
    rhoR = EINSUM_IJ_IJ_J(mydf.aoR, buf1)

    #### step 2. get potential ####

    rhoR = rhoR.reshape(mesh)
    rhoG = IFFTN(rhoR, s=mesh, axes=(0, 1, 2), overwrite_input=False).reshape(-1)
    rhoG *= mydf.coul_G
    rhoR = FFTN(
        rhoG.reshape(*mesh), s=mesh, axes=(0, 1, 2), overwrite_input=False
    ).reshape(-1)
    imag = IMAG(rhoR)
    rhoR = REAL(rhoR)
    norm = MAX(ABS((imag)))
    if norm > 1e-10:
        misc._warn(mydf, mydf.rank, "in _get_j_dm imag norm of rhoR %e", norm)

    #### step 3. get J ####

    J = EINSUM_IJ_J_IJ(mydf.aoR, rhoR, out=buf1)
    J = DOT(mydf.aoR, J.T)

    t2 = (logger.process_clock(), logger.perf_counter())

    misc._benchmark_time(t1, t2, "get_j_dm", mydf, mydf.rank)

    buffer.free(count=1)

    return J * (mydf.ngrids / mydf.cell.vol)


def _get_k_dm(mydf, dm, use_mpi=False):

    # if mydf.with_robust_fitting:
    #     misc._warn(
    #         mydf,
    #         mydf.rank,
    #         "ISDF with robust fitting is constructed but get_k use W only",
    #     )

    assert not use_mpi, "ISDF does not support MPI"

    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    # funcs #

    CWISE_MUL = BACKEND._cwise_mul

    #### step 0 info ####

    buffer = mydf.buffer

    #### step 1. get density matrix values on IPs ####

    buf1 = buffer.malloc((mydf.nao, mydf.naux), dtype=FLOAT64, name="buf1")
    dm_RgRg = DOT(dm, mydf.aoRg, c=buf1)
    buf2 = buffer.malloc((mydf.naux, mydf.naux), dtype=FLOAT64, name="buf2")
    dm_RgRg = DOT(mydf.aoRg.T, dm_RgRg, c=buf2)

    #### step 2. get K matrix ####

    tmp = CWISE_MUL(dm_RgRg, mydf.W, out=dm_RgRg)
    buf1 = buf1.reshape(mydf.naux, mydf.nao)
    tmp = DOT(tmp, mydf.aoRg.T, c=buf1)
    K = DOT(mydf.aoRg, tmp)

    t2 = (logger.process_clock(), logger.perf_counter())

    misc._benchmark_time(t1, t2, "get_k_dm", mydf, mydf.rank)

    buffer.free(count=2)

    return K * (mydf.ngrids / mydf.cell.vol)


def _get_k_dm_wrf(mydf, dm, use_mpi=False):

    assert (
        mydf.with_robust_fitting
    ), "ISDF without robust fitting is constructed but get_k_wrf is called"
    assert not use_mpi, "ISDF does not support MPI"

    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    # funcs #

    CWISE_MUL = BACKEND._cwise_mul
    ADD_T_ = BACKEND._add_transpose_

    #### step 0 info ####

    buffer = mydf.buffer

    #### step 1. get density matrix values on real space ####

    buf1 = buffer.malloc((mydf.nao, mydf.ngrids), dtype=FLOAT64, name="buf1")
    buf2 = buffer.malloc((mydf.naux, mydf.ngrids), dtype=FLOAT64, name="buf2")

    density_RgR = DOT(dm, mydf.aoR, c=buf1)
    density_RgR = DOT(mydf.aoRg.T, density_RgR, c=buf2)

    #### step 2. get K term1 and term2 ####

    tmp = CWISE_MUL(mydf.V, density_RgR, out=density_RgR)
    buf3 = BACKEND._malloc(
        (mydf.naux, mydf.nao), dtype=FLOAT64, buf=buf1, gpu=USE_GPU
    )  # reuse the buffer
    K = DOT(tmp, mydf.aoR.T, c=buf3)
    K = DOT(mydf.aoRg, K)
    # K += K.T
    K = ADD_T_(K)

    buffer.free(count=2)

    K *= mydf.ngrids / mydf.cell.vol
    K -= _get_k_dm(mydf, dm, use_mpi)

    t2 = (logger.process_clock(), logger.perf_counter())

    misc._benchmark_time(t1, t2, "get_k_dm_wrf", mydf, mydf.rank)

    return K


# DRIVER #


def get_jk_dm(
    mydf,
    dm,
    hermi=1,
    kpt=np.zeros(3),
    kpts_band=None,
    with_j=True,
    with_k=True,
    omega=None,
    use_mpi=False,
    **kwargs
):
    """JK for given k-point"""

    assert not use_mpi, "ISDF does not support MPI"

    dm = ToTENSOR(dm, cpu=not USE_GPU)

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1 or dm.shape[0] == 2
    else:
        assert dm.ndim == 2
        dm = dm.reshape(1, dm.shape[0], dm.shape[1])

    nset = dm.shape[0]

    if hasattr(mydf, "kmesh"):
        assert np.prod(mydf.kmesh) == 1

    #### perform the calculation ####

    if "exxdiv" in kwargs:
        exxdiv = kwargs["exxdiv"]
    else:
        exxdiv = None

    # vj = vk = None
    vj = np.zeros(dm.shape, dtype=np.float64)
    vk = np.zeros(dm.shape, dtype=np.float64)

    if kpts_band is not None and abs(kpt - kpts_band).sum() > 1e-9:
        raise NotImplementedError("ISDF does not support kpts_band != kpt")

    log = logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (logger.process_clock(), logger.perf_counter())

    j_real = gamma_point(kpt)
    k_real = gamma_point(kpt) and not np.iscomplexobj(ToNUMPY(dm))

    assert j_real
    assert k_real

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))

    log.debug1("max_memory = %d MB (%d in use)", max_memory, mem_now)

    for iset in range(nset):

        if with_j:
            vj[iset] = ToNUMPY(_get_j_dm(mydf, dm[iset], use_mpi))
        if with_k:
            if mydf.with_robust_fitting:
                vk[iset] = ToNUMPY(_get_k_dm_wrf(mydf, dm[iset], use_mpi))
            else:
                vk[iset] = ToNUMPY(_get_k_dm(mydf, dm[iset], use_mpi))
            if exxdiv == "ewald":
                print("WARNING: ISDF does not support ewald")

    dm = ToNUMPY(dm)

    ##### the following code is added to deal with _ewald_exxdiv_for_G0 #####

    from pyscf.pbc.df.df_jk import (
        _format_dms,
        _format_kpts_band,
        _format_jks,
        _ewald_exxdiv_for_G0,
    )

    kpts = kpt.reshape(1, 3)
    kpts = np.asarray(kpts)
    dm_kpts = dm.copy()
    dm_kpts = lib.asarray(dm_kpts, order="C")
    dms = _format_dms(dm_kpts, kpts)
    nset, nkpts, nao = dms.shape[:3]
    if nset > 2:
        logger.warn(
            mydf,
            "nset > 2, please confirm what you are doing, for RHF nset == 1, for UHF nset == 2",
        )
    assert nkpts == 1

    kpts_band, _ = _format_kpts_band(kpts_band, kpts), kpts_band
    nband = len(kpts_band)
    assert nband == 1

    if is_zero(kpts_band) and is_zero(kpts):
        vk = vk.reshape(nset, nband, nao, nao)
    else:
        raise NotImplementedError("ISDF does not support kpts_band != 0")

    if exxdiv == "ewald":
        _ewald_exxdiv_for_G0(mydf.cell, kpts, dms, vk, kpts_band=kpts_band)

    vk = vk.reshape(nset, nao, nao)

    t1 = log.timer("sr jk", *t1)

    return vj, vk
