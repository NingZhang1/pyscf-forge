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
from pyscf import ao2mo
from pyscf.lib import logger
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact

############ isdf backends ############

import pyscf.isdf.BackEnd.isdf_backend as BACKEND
from pyscf.isdf.BackEnd.isdf_memory_allocator import SimpleMemoryAllocator
from pyscf.isdf.BackEnd.isdf_fft_cache import DynamicCached3DRFFT

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
INDEX_ADD = BACKEND._index_add

############ isdf utils ############

import pyscf.isdf.misc as misc
from pyscf.isdf.isdf_ao2mo import _get_moR
from pyscf.isdf.isdf_tools_local import (
    _range_partition,
    _pack_aoR_holder,
)
from pyscf.isdf._isdf_local_K_kernel import (
    _build_V_local_bas_kernel,
    _build_W_local_bas_kernel,
    _get_V_W,
    _build_moPairR,
    _build_V_product_moPairR,
)

from pyscf.isdf.isdf_ao2mo import AOPAIR_BLKSIZE


def isdf_local_eri(
    mydf,
    mo_coeff=None,
    with_robust_fitting=None,
    use_mpi=None,
    AOPAIR_BLKSIZE=AOPAIR_BLKSIZE,
):
    """
    Perform AO2MO transformation from ISDF with robust fitting with s4 symmetry
    Locality is explored
    """

    t1 = (logger.process_clock(), logger.perf_counter())

    if with_robust_fitting is None:
        with_robust_fitting = mydf.with_robust_fitting
    if use_mpi is None:
        use_mpi = mydf.use_mpi
    direct = mydf.direct

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast
        from pyscf.isdf.isdf_tools_mpi import reduce as mpi_reduce
    else:
        rank = 0
        comm = None
        comm_size = 1
        mpi_reduce = None

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
    if with_robust_fitting:
        res_V = ZEROS((nnmo, nnmo), cpu=not USE_GPU)
    else:
        res_V = None
    res_ddot_buf = None

    # allocate buffer #

    size_aoRg_packed = mydf.nao * mydf.max_group_naux_possible(group=mydf.group)
    # print("group naux possible = ", mydf.max_group_naux_possible(group=mydf.group))
    if size_aoRg_packed * 8 > AOPAIR_BLKSIZE:
        raise ValueError(
            "The AOPAIR_BLKSIZE is too small for the current system, "
            "please increase the AOPAIR_BLKSIZE"
        )
    size0 = nnmo  # store moPairRgBra
    # size1 = nmo**2  # construct moPairRgBra
    size2 = 0  # build V and W
    if direct:
        # if with_robust_fitting:
        size2 += mydf.ngrids
        size2 += mydf.naux
    size3 = nmo * (nmo + 1) // 2  # store moPairRVKet
    size4 = nmo * (nmo + 1) // 2  # store moPairRgKet
    size5 = nnmo + nmo**2  # build moPairRgKet
    size = size0 + size2 + size3 + size4 + size5
    # print("size = ", size)

    GRID_BUNCHIZE = int((AOPAIR_BLKSIZE - size_aoRg_packed * 8) / (size * 8))

    buffer = SimpleMemoryAllocator(
        size * GRID_BUNCHIZE + size_aoRg_packed, FLOAT64, gpu=USE_GPU
    )
    if direct:
        buffer_fft = DynamicCached3DRFFT((GRID_BUNCHIZE, *mydf.mesh))
    else:
        buffer_fft = None

    # perform the contraction group by group bunch by bunch #

    group = mydf.group
    group_begin, group_end = _range_partition(len(group), rank, comm_size, use_mpi)
    assert group_begin == mydf.group_begin
    assert group_end == mydf.group_end

    coul_G = mydf.coul_G
    coul_G = ToTENSOR(coul_G, cpu=True).reshape(*mydf.mesh)
    coul_G = ToTENSOR(ToNUMPY(coul_G[:, :, : mydf.mesh[2] // 2 + 1].reshape(-1)).copy())

    # moPairInd #

    if with_robust_fitting:
        moPairRInd = []
        for _moR_ in moR:
            row_indices, col_indices = np.tril_indices(_moR_.nao_involved)
            row_indices = ToTENSOR(np.array(row_indices, dtype=np.int64))
            col_indices = ToTENSOR(np.array(col_indices, dtype=np.int64))
            indices = (
                _moR_.ao_involved[row_indices]
                * (_moR_.ao_involved[row_indices] + 1)
                // 2
                + _moR_.ao_involved[col_indices]
            )
            moPairRInd.append(indices)
    else:
        moPairRInd = None
    moPairRgInd = []
    for _moRg_ in moRg:
        row_indices, col_indices = np.tril_indices(_moRg_.nao_involved)
        row_indices = ToTENSOR(np.array(row_indices, dtype=np.int64))
        col_indices = ToTENSOR(np.array(col_indices, dtype=np.int64))
        indices = (
            _moRg_.ao_involved[row_indices] * (_moRg_.ao_involved[row_indices] + 1) // 2
            + _moRg_.ao_involved[col_indices]
        )
        moPairRgInd.append(indices)

    # cache indices take #

    indices_take_cached = {}

    def _find_indices_take(nmo):
        if nmo not in indices_take_cached:
            row_indices, col_indices = np.tril_indices(nmo)
            indices_take = ToTENSOR(row_indices * nmo + col_indices)
            indices_take_cached[nmo] = indices_take
        return indices_take_cached[nmo]

    # perform calculation #

    IP_begin_loc = 0
    for group_id in range(group_begin, group_end):
        # pack moRg #

        moRg_unpacked = [moRg[atm_id] for atm_id in group[group_id]]
        nIP_involved = sum(
            [moRg.aoR.shape[1] for moRg in moRg_unpacked if moRg is not None]
        )
        assert nIP_involved == mydf.IP_segment[group_id + 1] - mydf.IP_segment[group_id]
        nIP_i = nIP_involved
        # print("nIP_i = ", nIP_i)

        packed_buf_moRg = buffer.malloc(
            (nmo, nIP_involved), dtype=FLOAT64, name="packed_buf_moRg"
        )
        CLEAN(packed_buf_moRg)
        moRg_packed = _pack_aoR_holder(moRg_unpacked, mydf.nao, out_buf=packed_buf_moRg)

        # get the indices #

        nmo_tmp = moRg_packed.nao_involved
        row_indices, col_indices = np.tril_indices(nmo_tmp)
        indices_take = _find_indices_take(nmo_tmp)
        indcies_add = ToTENSOR(
            moRg_packed.ao_involved[row_indices]
            * (moRg_packed.ao_involved[row_indices] + 1)
            // 2
            + moRg_packed.ao_involved[col_indices]
        )

        for p0, p1 in lib.prange(0, nIP_i, GRID_BUNCHIZE):
            # print("p0, p1 = ", p0, p1)
            # build moPairRg #
            moPairRgBra = _build_moPairR(
                moRg_packed, p0, p1, indices_take, buffer, name="moPairRgBra"
            )
            # build V and W #
            V_tmp, W_tmp = _get_V_W(
                # mydf.aux_basis,
                mydf,
                group_id,
                p0,
                p1,
                coul_G,
                IP_begin_loc,
                direct,
                with_robust_fitting,
                buffer,
                buffer_fft,
            )
            # do the calculation #
            ## V term
            if with_robust_fitting:
                moPairRVKet = _build_V_product_moPairR(
                    moR,
                    p0,
                    p1,
                    nmo,
                    V_tmp,
                    indices_take_cached,
                    moPairRInd,
                    GRID_BUNCHIZE,
                    buffer,
                    name="moPairRVKet",
                )
                # do the final dot #
                if nmo_tmp == nmo:
                    DOT(moPairRgBra, moPairRVKet.T, c=res_V, beta=1)
                else:
                    if res_ddot_buf is None:
                        res_ddot_buf = MALLOC((nnmo, nnmo), dtype=FLOAT64, gpu=USE_GPU)
                    ddot_res = MALLOC(
                        (nmo_tmp * (nmo_tmp + 1) // 2, nnmo),
                        dtype=FLOAT64,
                        buf=res_ddot_buf,
                    )
                    DOT(moPairRgBra, moPairRVKet.T, c=ddot_res)
                    INDEX_ADD(res_V, 0, indcies_add, ddot_res)
                buffer.free(count=1)
            ## W term

            moPairRgWKet = _build_V_product_moPairR(
                moRg,
                p0,
                p1,
                nmo,
                W_tmp,
                indices_take_cached,
                moPairRgInd,
                GRID_BUNCHIZE,
                buffer,
                name="moPairRgWKet",
            )
            # do the final dot #
            if nmo_tmp == nmo:
                DOT(moPairRgBra, moPairRgWKet.T, c=res, beta=1)
            else:
                if res_ddot_buf is None:
                    res_ddot_buf = MALLOC((nnmo, nnmo), dtype=FLOAT64, gpu=USE_GPU)
                ddot_res = MALLOC(
                    (nmo_tmp * (nmo_tmp + 1) // 2, nnmo),
                    dtype=FLOAT64,
                    buf=res_ddot_buf,
                )
                DOT(moPairRgBra, moPairRgWKet.T, c=ddot_res)
                INDEX_ADD(res, 0, indcies_add, ddot_res)
            buffer.free(count=2)
            if direct:
                buffer.free(count=2)
        IP_begin_loc += nIP_i

        buffer.free(count=1)

    if use_mpi:
        res = ToTENSOR(mpi_reduce(res, root=0))
        if with_robust_fitting:
            res_V = ToTENSOR(mpi_reduce(res_V, root=0))

    if with_robust_fitting:
        if rank == 0:
            ADD_TRANSPOSE_(res_V)
            res = res_V - res

    if use_mpi:
        res = ToTENSOR(bcast(res, root=0))

    t2 = (logger.process_clock(), logger.perf_counter())
    misc._benchmark_time(t1, t2, "isdf_local_eri", mydf, mydf.rank)

    del res_ddot_buf

    return ToNUMPY(res) * mydf.ngrids / mydf.cell.vol


def isdf_local_eri_ovov(
    mydf,
    mo_coeff_o: np.ndarray = None,
    mo_coeff_v: np.ndarray = None,
    with_robust_fitting=None,
    use_mpi=None,
    AOPAIR_BLKSIZE=AOPAIR_BLKSIZE,
):
    """
    Perform AO2MO transformation from ISDF with robust fitting with s4 symmetry
    Locality is explored
    """

    if with_robust_fitting is None:
        with_robust_fitting = mydf.with_robust_fitting
    if use_mpi is None:
        use_mpi = mydf.use_mpi
    direct = mydf.direct

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast
        from pyscf.isdf.isdf_tools_mpi import reduce as mpi_reduce
    else:
        rank = 0
        comm = None
        comm_size = 1
        mpi_reduce = None

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
    if with_robust_fitting:
        res_V = ZEROS((nmo_ov, nmo_ov), cpu=not USE_GPU)
    else:
        res_V = None
    res_ddot_buf = None

    # allocate buffer #

    size_aoRg_packed = mydf.nao * mydf.max_group_naux_possible(group=mydf.group)
    if size_aoRg_packed * 8 > AOPAIR_BLKSIZE:
        raise ValueError(
            "The AOPAIR_BLKSIZE is too small for the current system, "
            "please increase the AOPAIR_BLKSIZE"
        )
    size0 = nmo_ov  # store moPairRgBra
    # size1 = 0  # construct moPairRgBra
    size2 = 0  # build V and W
    if direct:
        if with_robust_fitting:
            size2 += mydf.ngrids
        size2 += mydf.naux
    size3 = nmo_ov  # store moPairRVKet
    size4 = nmo_ov  # store moPairRgKet
    size5 = nmo_ov  # build moPairRgKet
    size = size0 + size2 + size3 + size4 + size5
    # print("size = ", size)

    GRID_BUNCHIZE = int((AOPAIR_BLKSIZE - size_aoRg_packed * 8) / (size * 8))

    buffer = SimpleMemoryAllocator(
        size * GRID_BUNCHIZE + size_aoRg_packed, FLOAT64, gpu=USE_GPU
    )
    if direct:
        buffer_fft = DynamicCached3DRFFT((GRID_BUNCHIZE, *mydf.mesh))
    else:
        buffer_fft = None

    # perform the contraction group by group bunch by bunch #

    group = mydf.group
    group_begin, group_end = _range_partition(len(group), rank, comm_size, use_mpi)
    assert group_begin == mydf.group_begin
    assert group_end == mydf.group_end

    coul_G = mydf.coul_G
    coul_G = ToTENSOR(coul_G, cpu=True).reshape(*mydf.mesh)
    coul_G = ToTENSOR(ToNUMPY(coul_G[:, :, : mydf.mesh[2] // 2 + 1].reshape(-1)).copy())

    # perform calculation #

    IP_begin_loc = 0
    for group_id in range(group_begin, group_end):
        # pack moRg_o #

        moRg_o_unpacked = [moRg_o[atm_id] for atm_id in group[group_id]]
        nIP_involved = sum(
            [moRg.aoR.shape[1] for moRg in moRg_o_unpacked if moRg is not None]
        )
        assert nIP_involved == mydf.IP_segment[group_id + 1] - mydf.IP_segment[group_id]
        nIP_i = nIP_involved

        packed_buf_moRg = buffer.malloc(
            (nmo_o, nIP_involved), dtype=FLOAT64, name="packed_buf_moRg_o"
        )
        CLEAN(packed_buf_moRg)
        moRg_o_packed = _pack_aoR_holder(
            moRg_o_unpacked, nmo_o, out_buf=packed_buf_moRg
        )

        # pack moRg_v #

        moRg_v_unpacked = [moRg_v[atm_id] for atm_id in group[group_id]]
        nIP_involved = sum(
            [moRg.aoR.shape[1] for moRg in moRg_v_unpacked if moRg is not None]
        )

        packed_buf_moRg = buffer.malloc(
            (nmo_v, nIP_involved), dtype=FLOAT64, name="packed_buf_moRg_v"
        )
        CLEAN(packed_buf_moRg)
        moRg_v_packed = _pack_aoR_holder(
            moRg_v_unpacked, nmo_v, out_buf=packed_buf_moRg
        )

        for p0, p1 in lib.prange(0, nIP_i, GRID_BUNCHIZE):
            # build moPairRg #
            moPairOVRgBra = buffer.malloc(
                (nmo_o, nmo_v, p1 - p0),
                dtype=FLOAT64,
                name="moPairOVRgBra",
            )
            EINSUM_IK_JK_IJK(
                moRg_o_packed.aoR[:, p0:p1],
                moRg_v_packed.aoR[:, p0:p1],
                out=moPairOVRgBra,
            )
            moPairOVRgBra = moPairOVRgBra.reshape(nmo_o * nmo_v, p1 - p0)
            # build V and W #
            if direct:
                V_tmp = _build_V_local_bas_kernel(
                    mydf.aux_basis,
                    group_id,
                    p0,
                    p1,
                    buffer,
                    buffer_fft,
                    mydf.partition_group_2_gridID,
                    mydf.gridID_ordering,
                    mydf.mesh,
                    coul_G,
                )  # V_tmp is stored in buffer_fft
                W_tmp = buffer.malloc((p1 - p0, mydf.naux), dtype=FLOAT64, name="W_tmp")
                W_tmp = _build_W_local_bas_kernel(V_tmp, mydf.aux_basis, W_tmp)
            else:
                if with_robust_fitting:
                    V_tmp = mydf.V[IP_begin_loc + p0 : IP_begin_loc + p1, :]
                else:
                    V_tmp = None
                W_tmp = mydf.W[IP_begin_loc + p0 : IP_begin_loc + p1, :]
            # do the calculation #
            ## V term
            if with_robust_fitting:
                moPairRVOVKet = buffer.malloc(
                    (nmo_o * nmo_v, p1 - p0),
                    dtype=FLOAT64,
                    name="moPairRVOVKet",
                )
                CLEAN(moPairRVOVKet)
                # build moPairRgKet #
                for atmid, (_moR_o_, _moR_v_) in enumerate(zip(moR_o, moR_v)):
                    ngrids_tmp = _moR_o_.aoR.shape[1]
                    V_grid_begin_ID = _moR_o_.global_gridID_begin
                    for q0, q1 in lib.prange(0, ngrids_tmp, GRID_BUNCHIZE):
                        moPairROVKet1 = buffer.malloc(
                            (nmo_o, nmo_v, q1 - q0),
                            dtype=FLOAT64,
                            name="moPairROVKet1",
                        )
                        EINSUM_IK_JK_IJK(
                            _moR_o_.aoR[:, q0:q1],
                            _moR_v_.aoR[:, q0:q1],
                            out=moPairROVKet1,
                        )
                        moPairROVKet1 = moPairROVKet1.reshape(nmo_o * nmo_v, q1 - q0)
                        DOT(
                            moPairROVKet1,
                            V_tmp[:, V_grid_begin_ID + q0 : V_grid_begin_ID + q1].T,
                            c=moPairRVOVKet,
                            beta=1,
                        )
                        buffer.free(count=1)
                # do the final dot #
                DOT(moPairOVRgBra, moPairRVOVKet.T, c=res_V, beta=1)
                buffer.free(count=1)
            ## W term
            moPairRgWKet = buffer.malloc(
                (nmo_o * nmo_v, p1 - p0),
                dtype=FLOAT64,
                name="moPairRgWKet",
            )
            CLEAN(moPairRgWKet)
            # build moPairRgWKet #
            # for atmid, _moRg_ in enumerate(moRg):
            for atmid, (_moRg_o_, _moRg_v_) in enumerate(zip(moRg_o, moRg_v)):
                ngrids_tmp = _moRg_o_.aoR.shape[1]
                W_grid_begin_ID = _moRg_o_.global_gridID_begin
                for q0, q1 in lib.prange(0, ngrids_tmp, GRID_BUNCHIZE):
                    moPairRgOVKet1 = buffer.malloc(
                        (nmo_o, nmo_v, q1 - q0),
                        dtype=FLOAT64,
                        name="moPairRgOVKet1",
                    )
                    EINSUM_IK_JK_IJK(
                        _moRg_o_.aoR[:, q0:q1],
                        _moRg_v_.aoR[:, q0:q1],
                        out=moPairRgOVKet1,
                    )
                    moPairRgOVKet1 = moPairRgOVKet1.reshape(nmo_o * nmo_v, q1 - q0)
                    DOT(
                        moPairRgOVKet1,
                        W_tmp[:, W_grid_begin_ID + q0 : W_grid_begin_ID + q1].T,
                        c=moPairRgWKet,
                        beta=1,
                    )
                    buffer.free(count=1)
            # do the final dot #
            DOT(moPairOVRgBra, moPairRgWKet.T, c=res, beta=1)
            buffer.free(count=2)
            if direct:
                buffer.free(count=1)
        IP_begin_loc += nIP_i

        buffer.free(count=2)

    if use_mpi:
        res = ToTENSOR(mpi_reduce(res, root=0))
        if with_robust_fitting:
            res_V = ToTENSOR(mpi_reduce(res_V, root=0))

    if with_robust_fitting:
        if rank == 0:
            ADD_TRANSPOSE_(res_V)
            res = res_V - res

    if use_mpi:
        res = ToTENSOR(bcast(res, root=0))

    t2 = (logger.process_clock(), logger.perf_counter())
    misc._benchmark_time(t1, t2, "isdf_local_eri", mydf, mydf.rank)

    del res_ddot_buf

    return (
        ToNUMPY(res).reshape(nmo_o, nmo_v, nmo_o, nmo_v) * mydf.ngrids / mydf.cell.vol
    )


def get_eri(
    mydf,
    kpts=None,
    compact=getattr(__config__, "pbc_df_ao2mo_get_eri_compact", True),
    with_robust_fitting=None,
    use_mpi=None,
    AOPAIR_BLKSIZE=AOPAIR_BLKSIZE,
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
        eri = isdf_local_eri(
            mydf,
            with_robust_fitting=with_robust_fitting,
            use_mpi=use_mpi,
            AOPAIR_BLKSIZE=AOPAIR_BLKSIZE,
        )

        if compact:
            return eri
        else:
            return ao2mo.restore(1, eri, nao)
    else:
        raise NotImplementedError("k-point dependent ISDF AO2MO is not implemented yet")


############# the most general ao2mo, with only s1 symmetry #############


def _general(
    mydf,
    mo_coeff=None,
    with_robust_fitting=None,
    AOPAIR_BLKSIZE=AOPAIR_BLKSIZE,
    debug=False,
):
    assert mo_coeff is not None and len(mo_coeff) == 4

    # log = logger.Logger(mydf.stdout, mydf.verbose)
    # time1 = (logger.process_clock(), logger.perf_counter())

    if with_robust_fitting is None:
        with_robust_fitting = mydf.with_robust_fitting
    if with_robust_fitting:
        logger.warn(
            mydf,
            "Robust fitting used in _general, since only W matrix is used, larger error may be introduced!",
        )

    from pyscf.isdf.isdf_tools_local import _pack_aoR_holder

    moRgP = _get_moR(mydf.aoRg, mo_coeff[0])
    moRgP = _pack_aoR_holder(moRgP, mo_coeff[0].shape[1]).aoR
    moRgQ = _get_moR(mydf.aoRg, mo_coeff[1])
    moRgQ = _pack_aoR_holder(moRgQ, mo_coeff[1].shape[1]).aoR
    moRgR = _get_moR(mydf.aoRg, mo_coeff[2])
    moRgR = _pack_aoR_holder(moRgR, mo_coeff[2].shape[1]).aoR
    moRgS = _get_moR(mydf.aoRg, mo_coeff[3])
    moRgS = _pack_aoR_holder(moRgS, mo_coeff[3].shape[1]).aoR

    sizeP = mo_coeff[0].shape[1]
    sizeQ = mo_coeff[1].shape[1]
    sizeR = mo_coeff[2].shape[1]
    sizeS = mo_coeff[3].shape[1]

    if debug:
        moRgP = ToNUMPY(moRgP)
        moRgQ = ToNUMPY(moRgQ)
        moRgR = ToNUMPY(moRgR)
        moRgS = ToNUMPY(moRgS)
        out = numpy.einsum(
            "pP,qP,rQ,sQ,PQ->pqrs", moRgP, moRgQ, moRgR, moRgS, mydf.W, optimize=True
        )
        return out * mydf.ngrids / mydf.cell.vol

    # memory requires is size_rs * (GRID_BUNCHSIZE) + size_rs * 2 * (GRID_BUNCHSIZE)
    # memory requires is size_qrs * (LOOP_SIZE) + GRID_BUNCHSIZE * sizeQ * (GRID_BUNCHSIZE)

    GRID_BUNCHIZE = int(AOPAIR_BLKSIZE // (sizeR * sizeS + sizeP * sizeQ) // 5 * 8)
    GRID_BUNCHIZE = min(GRID_BUNCHIZE, mydf.naux)
    LOOP_SIZE = int(AOPAIR_BLKSIZE // (sizeQ * sizeR * sizeS))
    LOOP_SIZE = min(LOOP_SIZE, sizeP)

    size_max = sizeR * sizeS * GRID_BUNCHIZE + sizeR * sizeS * 2 * GRID_BUNCHIZE
    size_max2 = (
        sizeR * sizeS * GRID_BUNCHIZE
        + LOOP_SIZE * sizeQ * GRID_BUNCHIZE
        + LOOP_SIZE * sizeQ * sizeR * sizeS
    )
    size_max = max(size_max, size_max2)

    buffer = SimpleMemoryAllocator(size_max, FLOAT64)

    # loop over blocks #

    W = mydf.W
    sizeRS = sizeR * sizeS
    out = ZEROS((sizeP, sizeQ, sizeR, sizeS), cpu=not USE_GPU, dtype=FLOAT64)

    for p0, p1 in lib.prange(0, mydf.naux, GRID_BUNCHIZE):
        # construct W * ket #
        moPairRgWKet = buffer.malloc((sizeRS, p1 - p0), name="moPairRgWKet")
        CLEAN(moPairRgWKet)
        for q0, q1 in lib.prange(0, mydf.naux, GRID_BUNCHIZE):
            moPairRg2 = buffer.malloc((sizeR, sizeS, q1 - q0), name="moPairRg2")
            EINSUM_IK_JK_IJK(moRgR[:, q0:q1], moRgS[:, q0:q1], out=moPairRg2)
            moPairRgKet = moPairRg2.reshape((sizeRS, q1 - q0))
            DOT(moPairRgKet, W[p0:p1, q0:q1].T, c=moPairRgWKet, beta=1)
            buffer.free(count=1)
        # construct bra * W * ket #
        for pp0, pp1 in lib.prange(0, sizeP, LOOP_SIZE):
            moPairRgBra = buffer.malloc((pp1 - pp0, sizeQ, p1 - p0), name="moPairRgBra")
            EINSUM_IK_JK_IJK(moRgP[pp0:pp1, p0:p1], moRgQ[:, p0:p1], out=moPairRgBra)
            moPairRgBra = moPairRgBra.reshape(((pp1 - pp0) * sizeQ, p1 - p0))
            out_buffer = buffer.malloc((pp1 - pp0, sizeQ, sizeRS), name="out_buffer")
            DOT(moPairRgBra, moPairRgWKet.T, c=out_buffer)
            out[pp0:pp1] += out_buffer.reshape((pp1 - pp0, sizeQ, sizeR, sizeS))
            buffer.free(count=2)
        buffer.free(count=1)

    return ToNUMPY(out) * mydf.ngrids / mydf.cell.vol


def general(
    mydf,
    mo_coeffs,
    kpts=None,
    compact=getattr(__config__, "pbc_df_ao2mo_general_compact", True),
    with_robust_fitting=None,
    AOPAIR_BLKSIZE=AOPAIR_BLKSIZE,
):
    from pyscf.pbc.df.df_ao2mo import warn_pbc2d_eri

    warn_pbc2d_eri(mydf)
    cell = mydf.cell
    # nao = cell.nao_nr()
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

            eri = isdf_local_eri(
                mydf,
                ToNUMPY(mo_coeffs[0]).copy(),
                with_robust_fitting=with_robust_fitting,
                AOPAIR_BLKSIZE=AOPAIR_BLKSIZE,
            )

            if compact:
                return eri
            else:
                return ao2mo.restore(1, eri, mo_coeffs[0].shape[1])
        else:
            #### ovov MO-ERI ####

            if iden_coeffs(mo_coeffs[0], mo_coeffs[2]) and iden_coeffs(
                mo_coeffs[1], mo_coeffs[3]
            ):
                eri_ovov = isdf_local_eri_ovov(
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
                if compact:
                    raise NotImplementedError(
                        "compact mode is not supported in general with different mo_coeffs"
                    )
                return _general(
                    mydf,
                    mo_coeffs,
                    with_robust_fitting=with_robust_fitting,
                    AOPAIR_BLKSIZE=AOPAIR_BLKSIZE,
                )
    else:
        raise NotImplementedError("k-point dependent ISDF AO2MO is not implemented yet")
