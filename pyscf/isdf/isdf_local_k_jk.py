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
REAL = BACKEND._real
IMAG = BACKEND._imag
MAX = BACKEND._maximum
ABS = BACKEND._absolute
DOT = BACKEND._dot
CLEAN = BACKEND._clean
TAKE = BACKEND._take
CWISE_MUL = BACKEND._cwise_mul
INDEX_COPY = BACKEND._index_copy

ADD_INDEXED_SUBMAT_ = BACKEND._add_to_indexed_submatrix_
COPY_INDEXED_SUBMAT = BACKEND._copy_indexed_submatrix
INDEX_ADD = BACKEND._index_add
CAST_TO_COMPLEX = BACKEND._cast_to_complex

############ isdf utils ############

import pyscf.isdf.misc as misc

from pyscf.isdf.isdf_tools_local import (
    _range_partition,
    _pack_aoR_holder,
)

from pyscf.isdf._isdf_local_K_kernel import (
    _build_V_local_bas_kernel,
    _build_W_local_bas_k_kernel,
    _get_dm_RgR_k,
    _get_half_K_k,
    _final_contraction_k,
)

from pyscf.isdf.isdf_local_jk import (
    J_MAX_GRID_BUNCHSIZE,
)

from pyscf.isdf.isdf_tools_Tsym import (
    _1e_operator_k2gamma,
    _1e_operator_gamma2k,
    pack_JK,
)

### wrt = with robust fitting ###

############ subroutines ############


def _preprocess_dm(mydf, _dm):

    dm = copy.deepcopy(_dm)

    # transform dm to gamma-space #

    if len(dm.shape) == 3:
        return _1e_operator_k2gamma(mydf.cell.nao_nr(), mydf.kmesh, dm)
    else:
        dm_res = []
        for i in range(dm.shape[0]):
            dm_res.append(_1e_operator_k2gamma(mydf.cell.nao_nr(), mydf.kmesh, dm[i]))
        return ToTENSOR(np.asarray(dm_res))


def _get_j_dm_k_local(mydf, dm, use_mpi=False):

    t1 = (logger.process_clock(), logger.perf_counter())

    #### preprocess ####

    # transform dm in k-space to real-space #

    dm = _preprocess_dm(mydf, dm)

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast
        from pyscf.isdf.isdf_tools_mpi import reduce2 as mpi_reduce

        dm = ToTENSOR(bcast(dm, root=0))
    else:
        comm_size = 1

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    nao = dm.shape[0]
    assert mydf.nao == nao

    # buffer #

    buffer = mydf.buffer_cpu

    # funcs #

    EINSUM_IJ_J_IJ = BACKEND._einsum_ij_j_ij
    EINSUM_IJ_IJ_J = BACKEND._einsum_ij_ij_j
    FFTN = partial(BACKEND._fftn, threads=NUM_THREADS)
    IFFTN = partial(BACKEND._ifftn, threads=NUM_THREADS)
    REAL = BACKEND._real
    IMAG = BACKEND._imag

    rhoR_prim = buffer.malloc((mydf.ngridPrim,), dtype=FLOAT64, name="rhoR_prim")
    CLEAN(rhoR_prim)

    #### step 1. get density value on real space grid and IPs ####

    # split task #

    assert len(mydf.aoR) == mydf.natmPrim or len(mydf.aoR) == mydf.natm
    assert len(mydf.aoR) == mydf.first_natm or len(mydf.aoR) == mydf.natm

    atm_id_begin, atm_id_end = _range_partition(
        mydf.first_natm, mydf.rank, comm_size, use_mpi
    )

    for atm_id in range(
        atm_id_begin, atm_id_end
    ):  # only the potential within the ref cell is needed

        aoR_holder = mydf.aoR[atm_id]
        ngrids_involved = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        ao_involved = aoR_holder.ao_involved
        gridID_begin = aoR_holder.global_gridID_begin
        aoR = aoR_holder.aoR

        if nao_involved == mydf.nao:
            dm_packed = ToTENSOR(dm)
        else:
            dm_packed = buffer.malloc(
                (nao_involved, nao_involved), dtype=FLOAT64, name="dm_packed"
            )
            # dm_packed[:] = dm[ao_involved][:, ao_involved]
            COPY_INDEXED_SUBMAT(ToTENSOR(dm), ao_involved, ao_involved, out=dm_packed)

        rhoR_local = buffer.malloc((ngrids_involved,), dtype=FLOAT64, name="rhoR_local")

        for p0, p1 in lib.prange(0, ngrids_involved, J_MAX_GRID_BUNCHSIZE):
            tmp1 = buffer.malloc((nao_involved, p1 - p0), dtype=FLOAT64, name="tmp1")
            DOT(dm_packed, aoR[:, p0:p1], c=tmp1)
            EINSUM_IJ_IJ_J(tmp1, aoR[:, p0:p1], out=rhoR_local[p0:p1])
            buffer.free(count=1)

        rhoR_prim[gridID_begin : gridID_begin + ngrids_involved] = rhoR_local

        # free buf #

        if nao_involved == mydf.nao:
            buffer.free(count=1)
        else:
            buffer.free(count=2)

    if use_mpi:
        rhoR_prim = ToTENSOR(mpi_reduce(rhoR_prim, root=0))

    #### step 2. construct the potential ####

    if mydf.rank == 0:

        meshPrim = tuple(mydf.meshPrim)

        # coulGPrim #

        if not hasattr(mydf, "coulG_prim"):
            mydf.coulG_prim = tools.get_coulG(mydf.prim_cell, mesh=mydf.prim_cell.mesh)

        rhoR_prim_new = ZEROS((mydf.ngridPrim,), dtype=FLOAT64)
        rhoR_prim_new[mydf.gridID_ordering_prim] = rhoR_prim
        rhoR_prim = rhoR_prim_new
        rhoR_prim = rhoR_prim.reshape(meshPrim)
        rhoG = IFFTN(
            rhoR_prim, s=meshPrim, axes=(0, 1, 2), overwrite_input=False
        ).reshape(-1)
        rhoG *= mydf.coulG_prim
        rhoR_prim = FFTN(
            rhoG.reshape(*meshPrim), s=meshPrim, axes=(0, 1, 2), overwrite_input=False
        ).reshape(-1)
        imag = IMAG(rhoR_prim)
        rhoR_prim = REAL(rhoR_prim)
        norm = MAX(ABS((imag)))
        if norm > 1e-10:
            misc._warn(mydf, mydf.rank, "in _half_J imag norm of rhoR_prim %e", norm)
        # rhoR_new = ZEROS((mydf.ngrids,), dtype=FLOAT64)
        rhoR_new = ToTENSOR(ToNUMPY(rhoR_prim[mydf.gridID_ordering_prim]).copy())
        half_J = rhoR_new
    else:
        half_J = None

    if use_mpi:
        half_J = ToTENSOR(bcast(half_J, root=0))

    buffer.free_all()

    #### step 3. get J , using translation symmetry ####

    J_Res = ZEROS((mydf.naoPrim, mydf.nao), dtype=FLOAT64)

    ntask = mydf.natmPrim * np.prod(mydf.kmesh)

    # split task #

    task_begin, task_end = _range_partition(ntask, mydf.rank, comm_size, use_mpi)

    for task_id in range(task_begin, task_end):
        box_loc1 = task_id // mydf.natmPrim
        box_loc2 = task_id % mydf.natmPrim

        box_x = box_loc1 // (mydf.kmesh[1] * mydf.kmesh[2])
        box_y = box_loc1 % (mydf.kmesh[1] * mydf.kmesh[2]) // mydf.kmesh[2]
        box_z = box_loc1 % mydf.kmesh[2]

        aoR_holder_bra = mydf._aoR_ref_unit(box_loc2, box_x, box_y, box_z)
        aoR_holder_ket = mydf.aoR[box_loc2]
        permutation_ket = mydf._get_permutation_column_aoR(
            box_x, box_y, box_z, box_loc2
        )
        J_tmp = half_J[mydf.grid_segment[box_loc2] : mydf.grid_segment[box_loc2 + 1]]

        assert aoR_holder_ket.aoR.shape[1] == J_tmp.shape[0]

        # perform the calculation #

        nao_bra = aoR_holder_bra.aoR.shape[0]
        nao_ket = aoR_holder_ket.aoR.shape[0]

        aoR_J_res = buffer.malloc(
            aoR_holder_bra.aoR.shape, dtype=FLOAT64, name="aoR_J_res"
        )
        EINSUM_IJ_J_IJ(aoR_holder_bra.aoR, J_tmp, out=aoR_J_res)
        ddot_res = buffer.malloc((nao_bra, nao_ket), dtype=FLOAT64, name="ddot_res")
        DOT(aoR_J_res, aoR_holder_ket.aoR.T, c=ddot_res)

        # pack col res #

        pack_col_res = buffer.malloc(
            (nao_bra, mydf.nao), dtype=FLOAT64, name="pack_col_res"
        )
        CLEAN(pack_col_res)
        INDEX_COPY(pack_col_res, 1, permutation_ket, ddot_res)
        INDEX_ADD(J_Res, 0, aoR_holder_bra.ao_involved, pack_col_res)

        buffer.free(count=3)

    t2 = (logger.process_clock(), logger.perf_counter())

    J = J_Res
    if use_mpi:
        J = ToTENSOR(mpi_reduce(J_Res, root=0))

    if mydf.rank == 0:
        J *= mydf.ngrids / mydf.cell.vol
        J = _1e_operator_gamma2k(mydf.cell.nao_nr(), mydf.kmesh, J)

    if use_mpi:
        J = ToTENSOR(bcast(J, root=0))

    t2 = (logger.process_clock(), logger.perf_counter())

    misc._benchmark_time(mydf, mydf.rank, "_get_j_dm_k_local", t1, t2)

    buffer.free_all()

    J = CAST_TO_COMPLEX(J)

    return J


## utils to split indices ##


def _get_ref_cell_aoR_indices(ao_involved, kmesh, naoPrim):

    from itertools import product

    res = {}

    # ao_involved is assumend to be sorted #

    for ix, iy, iz in product(range(kmesh[0]), range(kmesh[1]), range(kmesh[2])):
        target_box_x = (kmesh[0] - ix) % kmesh[0]
        target_box_y = (kmesh[1] - iy) % kmesh[1]
        target_box_z = (kmesh[2] - iz) % kmesh[2]

        ao_begin = (
            target_box_x * kmesh[1] * kmesh[2] + target_box_y * kmesh[2] + target_box_z
        ) * naoPrim
        ao_end = ao_begin + naoPrim

        pos = [
            loc
            for loc, aoID in enumerate(ao_involved)
            if aoID >= ao_begin and aoID < ao_end
        ]

        if len(pos) == 0:
            res[(ix, iy, iz)] = [None, None, None]
        else:
            loc_begin = pos[0]
            loc_end = pos[-1] + 1
            assert loc_end - loc_begin == len(pos)
            ao_idx = ToTENSOR(ToNUMPY(ao_involved[loc_begin:loc_end]).copy() - ao_begin)
            res[(ix, iy, iz)] = [loc_begin, loc_end, ao_idx]

    return res


def _get_k_dm_k_local(mydf, dm, direct=None, with_robust_fitting=None, use_mpi=False):

    t1 = (logger.process_clock(), logger.perf_counter())

    ####### preprocess #######

    # transform dm in k-space to real-space #

    dm = _preprocess_dm(mydf, dm)

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast
        from pyscf.isdf.isdf_tools_mpi import reduce2 as mpi_reduce

        dm = ToTENSOR(bcast(dm, root=0))
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
    buffer_fft = mydf.buffer_fft
    # buffer.clean()

    # info used in direct mode #

    if direct:
        group_gridID = mydf.partition_group_2_gridID
        grid_ordering = mydf.gridID_ordering
        mesh = mydf.mesh
        coul_G = mydf.coul_G
        coul_G = ToTENSOR(coul_G, cpu=True).reshape(*mesh)
        coul_G = ToTENSOR(ToNUMPY(coul_G[:, :, : mesh[2] // 2 + 1].reshape(-1)).copy())
    else:
        group_gridID = None
        grid_ordering = None
        mesh = None
        coul_G = None

    # funcs #

    # (1) split the tasks #

    group = mydf.group
    group_begin, group_end = _range_partition(len(group), mydf.rank, comm_size, use_mpi)

    # (2) build K #

    nao_prim = mydf.naoPrim

    K = ZEROS((nset, nao_prim, nao), dtype=FLOAT64)
    if with_robust_fitting:
        K_V = ZEROS((nset, nao_prim, nao), dtype=FLOAT64)
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

        packed_buf_aoRg = buffer.malloc(
            (nao, nIP_involved), dtype=FLOAT64, name="packed_buf_aoRg"
        )
        CLEAN(packed_buf_aoRg)
        aoRg_packed = _pack_aoR_holder(aoRg_unpacked, nao, out_buf=packed_buf_aoRg)
        # aoRg_packed = _pack_aoR_holder(aoRg_unpacked, nao)
        # aoRg_packed = aoRg_packed.toFull(nao) # NOTE: for debugging!

        ## pack dm ##

        ao_involved = aoRg_packed.ao_involved
        nao_involved = ao_involved.shape[0]

        if nao_involved == nao:
            dm_packed = dm
        else:
            dm_packed = buffer.malloc(
                (nset, nao_involved, nao), dtype=FLOAT64, name="dm_packed"
            )
            CLEAN(dm_packed)
            for i in range(nset):
                TAKE(dm[i], ao_involved, 0, out=dm_packed[i])

        segment_map = _get_ref_cell_aoR_indices(ao_involved, mydf.kmesh, mydf.naoPrim)

        for p0, p1 in lib.prange(0, nIP_involved, bunchsize):
            # if direct build W first #
            if direct:
                V_tmp = _build_V_local_bas_kernel(
                    mydf.aux_basis,
                    group_id,
                    p0,
                    p1,
                    buffer,
                    buffer_fft,
                    group_gridID,
                    grid_ordering,
                    mesh,
                    coul_G,
                )
                W_tmp = buffer.malloc((p1 - p0, mydf.naux), dtype=FLOAT64, name="W_tmp")
                W_tmp = _build_W_local_bas_k_kernel(
                    V_tmp, mydf.aux_basis, mydf.kmesh, W_tmp
                )
            else:
                if with_robust_fitting:
                    V_tmp = mydf.V[IP_begin_id + p0 : IP_begin_id + p1, :]
                else:
                    V_tmp = None
                W_tmp = mydf.W[IP_begin_id + p0 : IP_begin_id + p1, :]
            for i in range(nset):
                ## build dm_RgRg ##
                ## (1) build dm_RgAO ##
                dm_RgAO = buffer.malloc((p1 - p0, nao), dtype=FLOAT64, name="dm_RgAO")
                DOT(aoRg_packed.aoR[:, p0:p1].T, dm_packed[i], c=dm_RgAO)
                ## (2) contract with aoRg ##
                dm_RgRg = buffer.malloc(
                    (p1 - p0, mydf.naux), dtype=FLOAT64, name="dm_RgRg"
                )
                dm_RgRg = _get_dm_RgR_k(
                    mydf, dm_RgRg, dm_RgAO, mydf.aoRg, mydf.kmesh, True, buffer
                )
                # cwise dot #
                CWISE_MUL(W_tmp, dm_RgRg, out=dm_RgRg)
                # contract with aoRg ket #
                half_K = buffer.malloc((p1 - p0, nao), dtype=FLOAT64, name="half_K")
                half_K = _get_half_K_k(
                    mydf, half_K, dm_RgRg, mydf.aoRg, mydf.kmesh, True, buffer
                )
                # final contraction #
                _final_contraction_k(
                    K[i], aoRg_packed, p0, p1, half_K, segment_map, mydf.kmesh, buffer
                )
                if with_robust_fitting:
                    buffer.free(count=2)
                    ## (2) contract with aoR ##
                    dm_RgR = buffer.malloc(
                        (p1 - p0, mydf.ngrids), dtype=FLOAT64, name="dm_RgR"
                    )
                    dm_RgR = _get_dm_RgR_k(
                        mydf, dm_RgR, dm_RgAO, mydf.aoR, mydf.kmesh, False, buffer
                    )
                    # cwise_dot #
                    CWISE_MUL(V_tmp, dm_RgR, out=dm_RgR)
                    # contract with aoR ket #
                    half_K = buffer.malloc((p1 - p0, nao), dtype=FLOAT64, name="half_K")
                    half_K = _get_half_K_k(
                        mydf, half_K, dm_RgR, mydf.aoR, mydf.kmesh, False, buffer
                    )
                    # final contraction #
                    _final_contraction_k(
                        K_V[i],
                        aoRg_packed,
                        p0,
                        p1,
                        half_K,
                        segment_map,
                        mydf.kmesh,
                        buffer,
                    )
                    buffer.free(count=3)
                else:
                    buffer.free(count=3)

            if direct:
                buffer.free(count=2)  # both V and W #

        buffer.free_all()
        IP_begin_id += nIP_involved

    if use_mpi:
        K = ToTENSOR(mpi_reduce(K, root=0))
        if with_robust_fitting:
            K_V = ToTENSOR(mpi_reduce(K_V, root=0))
        if rank == 0:
            ## expand ##
            K = [pack_JK(K[i], mydf.kmesh, nao_prim) for i in range(nset)]
            K = ToTENSOR(ToNUMPY(K))
            for i in range(nset):
                K[i] = (K[i] + K[i].T) / 2.0
            if with_robust_fitting:
                K_V = [pack_JK(K_V[i], mydf.kmesh, nao_prim) for i in range(nset)]
                K_V = ToTENSOR(ToNUMPY(K_V))
                for i in range(nset):
                    K_V[i] = K_V[i] + K_V[i].T
                K = K_V - K
            # K = _1e_operator_gamma2k(mydf.cell.nao_nr(), mydf.kmesh, K)
            K = ToTENSOR(
                ToNUMPY(
                    [
                        ToNUMPY(
                            _1e_operator_gamma2k(mydf.cell.nao_nr(), mydf.kmesh, K[i])
                        )
                        for i in range(nset)
                    ]
                )
            )
            K = CAST_TO_COMPLEX(K)
        comm.barrier()
        K = bcast(K, root=0)
        K = ToTENSOR(K)
    else:
        K = [pack_JK(K[i], mydf.kmesh, nao_prim) for i in range(nset)]
        K = ToTENSOR(ToNUMPY(K))
        for i in range(nset):
            K[i] = (K[i] + K[i].T) / 2.0  # make it symmetric
        if with_robust_fitting:
            K_V = [pack_JK(K_V[i], mydf.kmesh, nao_prim) for i in range(nset)]
            K_V = ToTENSOR(ToNUMPY(K_V))
            for i in range(nset):
                K_V[i] = K_V[i] + K_V[i].T
            K = K_V - K
        # K = _1e_operator_gamma2k(mydf.cell.nao_nr(), mydf.kmesh, K)
        K = ToTENSOR(
            ToNUMPY(
                [
                    ToNUMPY(_1e_operator_gamma2k(mydf.cell.nao_nr(), mydf.kmesh, K[i]))
                    for i in range(nset)
                ]
            )
        )
        K = CAST_TO_COMPLEX(K)

    t2 = (logger.process_clock(), logger.perf_counter())
    misc._benchmark_time(mydf, mydf.rank, "_get_k_dm_k_local", t1, t2)

    return K * mydf.ngrids / mydf.cell.vol


def get_jk_dm_k_local(
    self,
    _dm,
    hermi=1,
    kpts=None,
    kpts_band=None,
    with_j=True,
    with_k=True,
    omega=None,
    exxdiv=None,
):

    from copy import deepcopy

    dm = deepcopy(_dm)

    if self.use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, bcast, comm

        dm = bcast(dm, root=0)

    if omega is not None:  # J/K for RSH functionals
        raise NotImplementedError
        # with self.range_coulomb(omega) as rsh_df:
        #     return rsh_df.get_jk(dm, hermi, kpts, kpts_band, with_j, with_k,
        #                          omega=None, exxdiv=exxdiv)

    from pyscf.pbc.df.aft import _check_kpts

    kpts, is_single_kpt = _check_kpts(self, kpts)

    if is_single_kpt:
        assert np.allclose(kpts[0], np.zeros(3))

    ### preprocess dm ###

    if dm.ndim == 3:
        dm = dm.reshape(1, *dm.shape)
    nset = dm.shape[0]

    vj = None
    if with_j:
        vj = np.zeros(dm.shape, dtype=np.complex128)
        for iset in range(nset):
            vj[iset] = ToNUMPY(_get_j_dm_k_local(self, dm[iset].copy(), self.use_mpi))

    vk = None
    if with_k:
        vk = ToNUMPY(_get_k_dm_k_local(self, dm.copy(), use_mpi=self.use_mpi))
        # vk = np.zeros_like(vj)

    ### post process J and K ###

    if not self.use_mpi or (self.use_mpi and rank == 0):

        kpts = np.asarray(kpts)
        dm_kpts = lib.asarray(dm, order="C")
        assert dm_kpts.ndim == 4
        assert dm_kpts.shape[1] == len(kpts)
        assert dm_kpts.shape[2] == dm_kpts.shape[3]
        dms = _format_dms(dm_kpts, kpts)
        nset, nkpts, nao = dms.shape[:3]
        assert nset <= 4

        kpts_band, input_band = _format_kpts_band(kpts_band, kpts), kpts_band
        nband = len(kpts_band)
        assert nband == nkpts

        cell = self.prim_cell

        if vk is not None:
            vk_kpts = vk.reshape(nset, nband, nao, nao)

            if exxdiv == "ewald":
                _ewald_exxdiv_for_G0(cell, kpts, dms, vk_kpts, kpts_band=kpts_band)

            vk = _format_jks(vk_kpts, dm_kpts, input_band, kpts)

        if vj is not None:
            vj_kpts = vj.reshape(nset, nband, nao, nao)
            vj = _format_jks(vj_kpts, dm_kpts, input_band, kpts)

        if nset == 1:

            vj = vj[0]
            vk = vk[0]

    if is_single_kpt:
        imag_j = IMAG(vj)
        vj = REAL(vj)
        imag_k = IMAG(vk)
        vk = REAL(vk)
        if MAX(ABS(imag_j)) > 1e-9:
            logger.warn(self, "J has imaginary part of size %s", MAX(ABS(imag_j)))
        if MAX(ABS(imag_k)) > 1e-9:
            logger.warn(self, "K has imaginary part of size %s", MAX(ABS(imag_k)))

    if self.use_mpi:

        vj = bcast(vj, root=0)
        vk = bcast(vk, root=0)

        comm.Barrier()

    # vk = np.zeros_like(vj)

    return vj, vk
