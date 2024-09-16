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

import copy, sys
import ctypes
import numpy as np
import h5py
from functools import partial

############ pyscf module ############

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks

############ isdf backends ############

import pyscf.isdf.BackEnd.isdf_backend as BACKEND

USE_GPU = BACKEND.USE_GPU

NUM_THREADS = BACKEND.NUM_THREADS
FLOAT64 = BACKEND.FLOAT64
ToNUMPY = BACKEND._toNumpy
ToTENSOR = BACKEND._toTensor
ZEROS = BACKEND._zeros
MAX = BACKEND._maximum
ABS = BACKEND._absolute
DOT = BACKEND._dot
CLEAN = BACKEND._clean
MALLOC = BACKEND._malloc
TAKE = BACKEND._take
CWISE_MUL = BACKEND._cwise_mul

ADD_INDEXED_SUBMAT_ = BACKEND._add_to_indexed_submatrix_
COPY_INDEXED_SUBMAT = BACKEND._copy_indexed_submatrix
INDEX_ADD = BACKEND._index_add

############ isdf utils ############

import pyscf.isdf.misc as misc

from pyscf.isdf.isdf_tools_local import (
    _range_partition,
    _pack_aoR_holder,
)

from pyscf.isdf._isdf_local_K_kernel import (
    _build_V_local_bas_kernel,
    _build_W_local_bas_kernel,
    _get_V_W,
    _get_dm_RgR,
    _get_half_K,
)

############ GLOBAL PARAMETER ############

J_MAX_GRID_BUNCHSIZE = 8192

##################################################
#
# only Gamma Point
#
##################################################

### ls = linear scaling ###
### wrt = with robust fitting ###


### NOTE: it is assumed that aoR is stored in all nodes not distributely ###


def _half_J(mydf, dm, use_mpi=False):
    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast
        from pyscf.isdf.isdf_tools_mpi import reduce2 as mpi_reduce
    else:
        comm_size = 1

    # buffer #

    buffer = mydf.buffer_cpu  ## only valid for CPU now ##
    mesh = tuple(mydf.mesh)

    # funcs #

    EINSUM_IJ_IJ_J = BACKEND._einsum_ij_ij_j
    FFTN = partial(BACKEND._fftn, threads=NUM_THREADS)
    IFFTN = partial(BACKEND._ifftn, threads=NUM_THREADS)
    REAL = BACKEND._real
    IMAG = BACKEND._imag

    # split task #

    atmID_begin, atmID_end = _range_partition(
        mydf.first_natm, mydf.rank, comm_size, use_mpi
    )

    rhoR = buffer.malloc((mydf.ngrids,), dtype=FLOAT64, name="rhoR")
    CLEAN(rhoR)

    # get the rhoR #

    for atmID in range(atmID_begin, atmID_end):
        aoR = mydf.aoR[atmID].aoR
        gridID_begin = mydf.aoR[atmID].global_gridID_begin
        ngrids_involved = aoR.shape[1]
        ao_involved = mydf.aoR[atmID].ao_involved
        nao_involved = aoR.shape[0]

        ## pack dm ##

        if nao_involved == mydf.nao:
            dm_packed = dm
        else:
            dm_packed = buffer.malloc(
                (nao_involved, nao_involved), dtype=FLOAT64, name="dm_packed"
            )
            # dm_packed[:] = dm[ao_involved][:, ao_involved]
            COPY_INDEXED_SUBMAT(dm, ao_involved, ao_involved, out=dm_packed)

        rhoR_local = buffer.malloc((ngrids_involved,), dtype=FLOAT64, name="rhoR_local")
        # CLEAN(rhoR_local)

        for p0, p1 in lib.prange(0, ngrids_involved, J_MAX_GRID_BUNCHSIZE):
            tmp1 = buffer.malloc((nao_involved, p1 - p0), dtype=FLOAT64, name="tmp1")
            DOT(dm_packed, aoR[:, p0:p1], c=tmp1)
            EINSUM_IJ_IJ_J(tmp1, aoR[:, p0:p1], out=rhoR_local[p0:p1])
            buffer.free(count=1)

        rhoR[gridID_begin : gridID_begin + ngrids_involved] = rhoR_local

        # free buf #

        if nao_involved == mydf.nao:
            buffer.free(count=1)
        else:
            buffer.free(count=2)

    # get the potential #

    if use_mpi:
        rhoR = ToTENSOR(mpi_reduce(rhoR, root=0))

    if mydf.rank == 0:
        rhoR_new = ZEROS((mydf.ngrids,), dtype=FLOAT64)
        rhoR_new[mydf.gridID_ordering] = rhoR
        rhoR = rhoR_new
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
            misc._warn(mydf, mydf.rank, "in _half_J imag norm of rhoR %e", norm)
        # rhoR_new = ZEROS((mydf.ngrids,), dtype=FLOAT64)
        rhoR_new = ToTENSOR(ToNUMPY(rhoR[mydf.gridID_ordering]).copy())
        half_J = rhoR_new
    else:
        half_J = None

    # reduce and bcast if use_mpi #

    if use_mpi:
        half_J = ToTENSOR(bcast(half_J, root=0))

    return half_J


def _get_j_dm_local(mydf, dm, use_mpi=False):
    ####### preprocess #######

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast
        from pyscf.isdf.isdf_tools_mpi import reduce2 as mpi_reduce
    else:
        comm_size = 1

    assert isinstance(mydf.aoRg, list)
    if mydf.aoR is not None:
        assert isinstance(mydf.aoR, list)

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    ####### buffer #######

    buffer = mydf.buffer_cpu  ## only valid for CPU now ##
    buffer.free_all()

    # funcs #

    EINSUM_IJ_J_IJ = BACKEND._einsum_ij_j_ij

    ####### Start the calculation ########

    ### get the half J ###

    t1 = (logger.process_clock(), logger.perf_counter())

    J_half = _half_J(
        mydf, dm, use_mpi
    )  # J_half should always be allocated rather than allocated from buffer
    buffer.free_all()

    t2 = (logger.process_clock(), logger.perf_counter())

    misc._benchmark_time(t1, t2, "J_half", mydf, mydf.rank)

    ### get the full J ###

    J = ZEROS((mydf.nao, mydf.nao), dtype=FLOAT64)

    # (1) split the task #

    atmID_begin, atmID_end = _range_partition(
        mydf.first_natm, mydf.rank, comm_size, use_mpi
    )

    # (2) get the full J #

    for atmID in range(atmID_begin, atmID_end):
        aoR = mydf.aoR[atmID].aoR
        gridID_begin = mydf.aoR[atmID].global_gridID_begin
        ngrids_involved = aoR.shape[1]
        ao_involved = mydf.aoR[atmID].ao_involved
        nao_involved = aoR.shape[0]

        J_local = buffer.malloc(
            (nao_involved, nao_involved), dtype=FLOAT64, name="J_local"
        )
        CLEAN(J_local)

        for p0, p1 in lib.prange(0, ngrids_involved, J_MAX_GRID_BUNCHSIZE):
            tmp1 = buffer.malloc((nao_involved, p1 - p0), dtype=FLOAT64, name="tmp1")

            EINSUM_IJ_J_IJ(
                aoR[:, p0:p1], J_half[gridID_begin + p0 : gridID_begin + p1], out=tmp1
            )

            DOT(tmp1, aoR[:, p0:p1].T, c=J_local, beta=1)

            buffer.free(count=1)

        # final addition #

        if nao_involved == mydf.nao:
            # NOTE: it is assumed that the orbid is ordered #
            J += J_local
        else:
            ADD_INDEXED_SUBMAT_(J, ao_involved, ao_involved, J_local)

        buffer.free(count=1)

    buffer.free_all()

    #### reduce everything ####

    if use_mpi:
        J = mpi_reduce(J, root=0)
        J = bcast(J, root=0)
        J = ToTENSOR(J)

    return J * mydf.ngrids / mydf.cell.vol


def _get_k_dm_local(mydf, dm, direct=None, with_robust_fitting=None, use_mpi=False):
    ####### preprocess #######

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast
        from pyscf.isdf.isdf_tools_mpi import reduce2 as mpi_reduce
    else:
        comm_size = 1

    assert isinstance(mydf.aoRg, list)
    if mydf.aoR is not None:
        assert isinstance(mydf.aoR, list)

    if len(dm.shape) == 3:
        assert dm.shape[0] <= 4  # at most 4!
    else:
        assert len(dm.shape) == 2
        dm = dm.reshape(1, *dm.shape)

    nset, nao = dm.shape[:2]

    if with_robust_fitting is None:
        with_robust_fitting = mydf.with_robust_fitting
    if direct is None:
        direct = mydf.direct

    ####### buffer #######

    buffer = mydf.buffer_cpu  ## only valid for CPU now ##
    buffer.free_all()
    # buffer_fft = mydf.buffer_fft

    # info used in direct mode #

    if direct:
        # group_gridID = mydf.partition_group_2_gridID
        # grid_ordering = mydf.gridID_ordering
        mesh = mydf.mesh
        coul_G = mydf.coul_G
        coul_G = ToTENSOR(coul_G, cpu=True).reshape(*mesh)
        coul_G = ToTENSOR(ToNUMPY(coul_G[:, :, : mesh[2] // 2 + 1].reshape(-1)).copy())
    else:
        # group_gridID = None
        # grid_ordering = None
        mesh = None
        coul_G = None

    # funcs #

    # (1) split the tasks #

    group = mydf.group
    group_begin, group_end = _range_partition(len(group), mydf.rank, comm_size, use_mpi)

    # (2) build K #

    K = ZEROS((nset, nao, nao), dtype=FLOAT64)
    if with_robust_fitting:
        K_V = ZEROS((nset, nao, nao), dtype=FLOAT64)
    else:
        K_V = None

    bunchsize = mydf._build_V_K_bunchsize

    IP_begin_id = 0
    for group_id in range(group_begin, group_end):
        # pack aoRg #

        aoRg_unpacked = [mydf.aoRg[atm_id] for atm_id in group[group_id]]
        nIP_involved = sum(
            [aoRg.aoR.shape[1] for aoRg in aoRg_unpacked if aoRg is not None]
        )
        assert nIP_involved == mydf.IP_segment[group_id + 1] - mydf.IP_segment[group_id]
        # IP_begin_id = mydf.IP_segment[group_id]

        packed_buf_aoRg = buffer.malloc(
            (mydf.nao, nIP_involved), dtype=FLOAT64, name="packed_buf_aoRg"
        )
        CLEAN(packed_buf_aoRg)
        aoRg_packed = _pack_aoR_holder(aoRg_unpacked, mydf.nao, out_buf=packed_buf_aoRg)

        ## pack dm ##

        ao_involved = aoRg_packed.ao_involved
        nao_involved = ao_involved.shape[0]

        if nao_involved == mydf.nao:
            dm_packed = dm
        else:
            dm_packed = buffer.malloc(
                (nset, nao_involved, mydf.nao), dtype=FLOAT64, name="dm_packed"
            )
            for i in range(nset):
                TAKE(dm[i], ao_involved, 0, out=dm_packed[i])

        for p0, p1 in lib.prange(0, nIP_involved, bunchsize):
            # if direct build W first #
            V_tmp, W_tmp = _get_V_W(
                mydf,
                group_id,
                p0,
                p1,
                coul_G,
                IP_begin_id,
                direct,
                with_robust_fitting,
            )
            for i in range(nset):
                ## build dm_RgRg ##
                ## (1) build dm_RgAO ##
                dm_RgAO = buffer.malloc((p1 - p0, nao), dtype=FLOAT64, name="dm_RgAO")
                DOT(aoRg_packed.aoR[:, p0:p1].T, dm_packed[i], c=dm_RgAO)
                ## (2) contract with aoRg ##
                dm_RgRg = buffer.malloc(
                    (p1 - p0, mydf.naux), dtype=FLOAT64, name="dm_RgRg"
                )
                dm_RgRg = _get_dm_RgR(
                    dm_RgRg,
                    dm_RgAO,
                    mydf.aoRg,
                    buffer,
                )
                # cwise dot #
                CWISE_MUL(W_tmp, dm_RgRg, out=dm_RgRg)
                # contract with aoRg ket #
                half_K = buffer.malloc((p1 - p0, nao), dtype=FLOAT64, name="half_K")
                half_K = _get_half_K(
                    half_K,
                    dm_RgRg,
                    mydf.aoRg,
                    buffer,
                )
                # final contraction #
                if nao_involved == mydf.nao:
                    DOT(aoRg_packed.aoR[:, p0:p1], half_K, c=K[i], beta=1)
                else:
                    tmp_ddot_res = buffer.malloc(
                        (nao_involved, nao), dtype=FLOAT64, name="tmp_ddot_res"
                    )
                    DOT(aoRg_packed.aoR[:, p0:p1], half_K, c=tmp_ddot_res)
                    INDEX_ADD(K[i], 0, ao_involved, tmp_ddot_res)
                    buffer.free(count=1)
                if with_robust_fitting:
                    buffer.free(count=2)
                    ## (2) contract with aoR ##
                    dm_RgR = buffer.malloc(
                        (p1 - p0, mydf.ngrids), dtype=FLOAT64, name="dm_RgR"
                    )
                    dm_RgR = _get_dm_RgR(
                        dm_RgR,
                        dm_RgAO,
                        mydf.aoR,
                        buffer,
                    )
                    # cwise_dot #
                    CWISE_MUL(V_tmp, dm_RgR, out=dm_RgR)
                    # contract with aoR ket #
                    half_K = buffer.malloc((p1 - p0, nao), dtype=FLOAT64, name="half_K")
                    half_K = _get_half_K(
                        half_K,
                        dm_RgR,
                        mydf.aoR,
                        buffer,
                    )
                    # final contraction #
                    if nao_involved == mydf.nao:
                        DOT(aoRg_packed.aoR[:, p0:p1], half_K, c=K_V[i], beta=1)
                    else:
                        tmp_ddot_res = buffer.malloc(
                            (nao_involved, nao), dtype=FLOAT64, name="tmp_ddot_res"
                        )
                        DOT(aoRg_packed.aoR[:, p0:p1], half_K, c=tmp_ddot_res)
                        INDEX_ADD(K_V[i], 0, ao_involved, tmp_ddot_res)
                        buffer.free(count=1)
                    buffer.free(count=3)
                else:
                    buffer.free(count=3)
            if direct:
                buffer.free(count=2)  # both V and W

        buffer.free_all()
        IP_begin_id += nIP_involved

    if use_mpi:
        K = mpi_reduce(K, root=0)
        if with_robust_fitting:
            K_V = mpi_reduce(K_V, root=0)
        if rank == 0:
            for i in range(nset):
                K[i] = (K[i] + K[i].T) / 2.0
            if with_robust_fitting:
                for i in range(nset):
                    K_V[i] = K_V[i] + K_V[i].T
                K = K_V - K
        comm.barrier()
        K = bcast(K, root=0)
        K = ToTENSOR(K)
    else:
        for i in range(nset):
            K[i] = (K[i] + K[i].T) / 2.0  # make it symmetric
        if with_robust_fitting:
            for i in range(nset):
                K_V[i] = K_V[i] + K_V[i].T
            K = K_V - K

    return K * mydf.ngrids / mydf.cell.vol


def _get_k_dm_local_outcore(
    mydf, dm, direct=None, with_robust_fitting=None, use_mpi=False
):
    ####### preprocess #######

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast
        from pyscf.isdf.isdf_tools_mpi import reduce2 as mpi_reduce
    else:
        comm_size = 1

    assert isinstance(mydf.aoRg, list)
    if mydf.aoR is not None:
        assert isinstance(mydf.aoR, list)

    if len(dm.shape) == 3:
        assert dm.shape[0] <= 4  # at most 4!
    else:
        assert len(dm.shape) == 2
        dm = dm.reshape(1, *dm.shape)

    nset, nao = dm.shape[:2]

    if with_robust_fitting is None:
        with_robust_fitting = mydf.with_robust_fitting
    if direct is None:
        direct = mydf.direct
    assert not with_robust_fitting
    assert direct

    ####### buffer #######

    buffer = mydf.buffer_cpu  ## only valid for CPU now ##
    buffer.free_all()

    # info used in direct mode #

    if direct:
        mesh = mydf.mesh
        coul_G = mydf.coul_G
        coul_G = ToTENSOR(coul_G, cpu=True).reshape(*mesh)
        coul_G = ToTENSOR(ToNUMPY(coul_G[:, :, : mesh[2] // 2 + 1].reshape(-1)).copy())
    else:
        mesh = None
        coul_G = None

    # funcs #

    # (1) split the tasks #

    group = mydf.group
    group_begin, group_end = _range_partition(len(group), mydf.rank, comm_size, use_mpi)

    # (2) build K #

    K = ZEROS((nset, nao, nao), dtype=FLOAT64)
    if with_robust_fitting:
        K_V = ZEROS((nset, nao, nao), dtype=FLOAT64)
    else:
        K_V = None

    bunchsize = mydf._build_V_K_bunchsize

    W_file_handle = h5py.File(mydf.W, "r")

    IP_begin_id = 0
    # ISTEP = 0
    for group_id in range(group_begin, group_end):

        # buf for load W #

        buf1 = buffer.malloc((bunchsize, mydf.naux), dtype=FLOAT64, name="buf1")
        buf2 = buffer.malloc((bunchsize, mydf.naux), dtype=FLOAT64, name="buf2")

        # pack aoRg #

        aoRg_unpacked = [mydf.aoRg[atm_id] for atm_id in group[group_id]]
        nIP_involved = sum(
            [aoRg.aoR.shape[1] for aoRg in aoRg_unpacked if aoRg is not None]
        )
        assert nIP_involved == mydf.IP_segment[group_id + 1] - mydf.IP_segment[group_id]
        # IP_begin_id = mydf.IP_segment[group_id]

        packed_buf_aoRg = buffer.malloc(
            (mydf.nao, nIP_involved), dtype=FLOAT64, name="packed_buf_aoRg"
        )
        CLEAN(packed_buf_aoRg)
        aoRg_packed = _pack_aoR_holder(aoRg_unpacked, mydf.nao, out_buf=packed_buf_aoRg)

        ## pack dm ##

        ao_involved = aoRg_packed.ao_involved
        nao_involved = ao_involved.shape[0]

        if nao_involved == mydf.nao:
            dm_packed = dm
        else:
            dm_packed = buffer.malloc(
                (nset, nao_involved, mydf.nao), dtype=FLOAT64, name="dm_packed"
            )
            for i in range(nset):
                TAKE(dm[i], ao_involved, 0, out=dm_packed[i])

        def load(bunch_range):
            nonlocal buf1, buf2
            istep, (p0, p1) = bunch_range
            buf2, buf1 = buf1, buf2
            W_tmp = MALLOC((p1 - p0, mydf.naux), dtype=FLOAT64, buf=buf1, offset=0)

            W_file_handle["W"].read_direct(
                ToNUMPY(W_tmp),
                dest_sel=np.s_[:, :],
                source_sel=np.s_[p0 + IP_begin_id : p1 + IP_begin_id],
            )
            return W_tmp

        bunch_range = list(enumerate(lib.prange(0, nIP_involved, bunchsize)))

        # for p0, p1 in lib.prange(0, nIP_involved, bunchsize):
        for istep, W_tmp in enumerate(lib.map_with_prefetch(load, bunch_range)):
            p0, p1 = bunch_range[istep][1]
            for i in range(nset):
                ## build dm_RgRg ##
                ## (1) build dm_RgAO ##
                dm_RgAO = buffer.malloc((p1 - p0, nao), dtype=FLOAT64, name="dm_RgAO")
                DOT(aoRg_packed.aoR[:, p0:p1].T, dm_packed[i], c=dm_RgAO)
                ## (2) contract with aoRg ##
                dm_RgRg = buffer.malloc(
                    (p1 - p0, mydf.naux), dtype=FLOAT64, name="dm_RgRg"
                )
                dm_RgRg = _get_dm_RgR(
                    dm_RgRg,
                    dm_RgAO,
                    mydf.aoRg,
                    buffer,
                )
                # cwise dot #
                CWISE_MUL(W_tmp, dm_RgRg, out=dm_RgRg)
                # contract with aoRg ket #
                half_K = buffer.malloc((p1 - p0, nao), dtype=FLOAT64, name="half_K")
                half_K = _get_half_K(
                    half_K,
                    dm_RgRg,
                    mydf.aoRg,
                    buffer,
                )
                # final contraction #
                if nao_involved == mydf.nao:
                    DOT(aoRg_packed.aoR[:, p0:p1], half_K, c=K[i], beta=1)
                else:
                    tmp_ddot_res = buffer.malloc(
                        (nao_involved, nao), dtype=FLOAT64, name="tmp_ddot_res"
                    )
                    DOT(aoRg_packed.aoR[:, p0:p1], half_K, c=tmp_ddot_res)
                    INDEX_ADD(K[i], 0, ao_involved, tmp_ddot_res)
                    buffer.free(count=1)
                buffer.free(count=3)
        buffer.free_all()
        IP_begin_id += nIP_involved

    if use_mpi:
        K = mpi_reduce(K, root=0)
        if with_robust_fitting:
            K_V = mpi_reduce(K_V, root=0)
        if rank == 0:
            for i in range(nset):
                K[i] = (K[i] + K[i].T) / 2.0
            if with_robust_fitting:
                for i in range(nset):
                    K_V[i] = K_V[i] + K_V[i].T
                K = K_V - K
        comm.barrier()
        K = bcast(K, root=0)
        K = ToTENSOR(K)
    else:
        for i in range(nset):
            K[i] = (K[i] + K[i].T) / 2.0  # make it symmetric
        if with_robust_fitting:
            for i in range(nset):
                K_V[i] = K_V[i] + K_V[i].T
            K = K_V - K

    W_file_handle.close()

    return K * mydf.ngrids / mydf.cell.vol


#### driver ####


def get_jk_dm_local(
    mydf,
    dm,
    hermi=1,
    kpt=np.zeros(3),
    kpts_band=None,
    with_j=True,
    with_k=True,
    omega=None,
    **kwargs
):
    """JK"""

    #### preprocess ####

    if dm.ndim == 2:
        dm = dm.reshape(1, *dm.shape)

    assert dm.ndim == 3

    # direct = mydf.direct
    # with_robust_fitting = mydf.with_robust_fitting
    use_mpi = mydf.use_mpi

    if len(dm.shape) == 3:
        assert dm.shape[0] <= 4
        ## NOTE: 1 for RHF 2 for UHF 3/4 for GHF

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, bcast

        dm = bcast(dm, root=0)

    dm = ToTENSOR(dm, cpu=True)
    nset, nao = dm.shape[:2]

    if mydf._force_translation_sym:
        assert mydf.natm % np.prod(mydf._T_mesh) == 0
        from pyscf.isdf.isdf_tools_Tsym import symmetrize_mat

        dm = symmetrize_mat(dm, mydf._T_mesh)

    #### perform the calculation ####

    if "exxdiv" in kwargs:
        exxdiv = kwargs["exxdiv"]
        kwargs.pop("exxdiv")
    else:
        exxdiv = None

    assert exxdiv in ["ewald", None]

    vj = vk = None

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

    vj = None
    vk = None

    if with_j:
        vj = np.zeros((nset, nao, nao))
        for iset in range(nset):
            vj[iset] = ToNUMPY(_get_j_dm_local(mydf, dm[iset], use_mpi=use_mpi))
    if with_k:
        if not mydf.outcore:
            vk = ToNUMPY(_get_k_dm_local(mydf, dm, use_mpi=use_mpi))
        else:
            vk = ToNUMPY(_get_k_dm_local_outcore(mydf, dm, use_mpi=use_mpi))
        # vk = np.zeros_like(vj)

    dm = ToNUMPY(dm)

    if (not use_mpi or (use_mpi and rank == 0)) and vk is not None:
        kpts = kpt.reshape(1, 3)
        kpts = np.asarray(kpts)
        dm_kpts = dm.reshape(-1, dm.shape[0], dm.shape[1]).copy()
        dm_kpts = lib.asarray(dm_kpts, order="C")
        dms = _format_dms(dm_kpts, kpts)
        nset, nkpts, nao = dms.shape[:3]

        assert nset <= 4
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

        vk = vk[:, 0, :, :]

    if vk is None:
        vk = np.zeros((nset, nao, nao))

    if use_mpi:
        if rank == 0 and mydf._force_translation_sym:
            from pyscf.isdf.isdf_tools_Tsym import symmetrize_mat

            vj = symmetrize_mat(vj, mydf._T_mesh)
            vk = symmetrize_mat(vk, mydf._T_mesh)
        vj = bcast(vj, root=0)
        vk = bcast(vk, root=0)
    else:
        if mydf._force_translation_sym:
            from pyscf.isdf.isdf_tools_Tsym import symmetrize_mat

            vj = symmetrize_mat(vj, mydf._T_mesh)
            vk = symmetrize_mat(vk, mydf._T_mesh)

    # vk = np.zeros_like(vj)

    t1 = log.timer("get_jk_dm_local", *t1)

    return vj, vk
