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
from itertools import product

############ pyscf module ############

from pyscf import lib
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
from pyscf.isdf.isdf_tools_local import _range_partition, aoR_Holder, _pack_aoR_holder
from pyscf.isdf._isdf_local_K_kernel import (
    _build_V_local_bas_kernel,
    _build_W_local_bas_k_kernel,
)

from pyscf.isdf.isdf_ao2mo import AOPAIR_BLKSIZE
from pyscf.isdf.isdf_local_k import ISDF_Local_K

# utils #


def eri_restore(eri, symmetry, nemb):
    """
    Restore eri with given permutation symmetry.
    """

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    spin_pair = eri.shape[0]
    if spin_pair == 1:
        eri_res = ao2mo.restore(symmetry, eri[0].real, nemb)
    else:
        if symmetry == 4:
            nemb_pair = nemb * (nemb + 1) // 2
            if eri.size == spin_pair * nemb_pair * nemb_pair:
                eri_res = eri.real.reshape(spin_pair, nemb_pair, nemb_pair)
            else:
                eri_res = np.empty((spin_pair, nemb_pair, nemb_pair))
                for s in range(spin_pair):
                    eri_res[s] = ao2mo.restore(symmetry, eri[s].real, nemb)
        elif symmetry == 1:
            if eri.size == spin_pair * nemb**4:
                eri_res = eri.real.reshape(spin_pair, nemb, nemb, nemb, nemb)
            else:
                eri_res = np.empty((spin_pair, nemb, nemb, nemb, nemb))
                for s in range(spin_pair):
                    eri_res[s] = ao2mo.restore(symmetry, eri[s].real, nemb)
        else:
            raise ValueError("Spin unrestricted ERI does not support 8-fold symmetry.")
    eri_res = np.asarray(eri_res, order="C")

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

    global cputime_restore, walltime_restore
    cputime_restore += t2[0] - t1[0]
    walltime_restore += t2[1] - t1[1]

    return eri_res


def _get_embR_box(
    mydf: ISDF_Local_K,
    C_ao_emb,
    ix,
    iy,
    iz,
    kmesh,
    build_embRg=False,
    aoR_cutoff=1e-8,
    buffer=None,
):
    """
    get embeded orbital values on given box,
    return list[aoR_holder] with size nspin with aoR_cutoff
    """

    assert buffer is not None

    nspin, nao_full, nemb = C_ao_emb.shape
    # C_ao_emb_T = C_ao_emb.transpose(0, 2, 1).copy()

    # ILOC = ix * kmesh[1] * kmesh[2] + iy * kmesh[2] + iz

    if build_embRg:
        # ngrid = mydf.nauxPrim
        permutation = mydf._get_permutation_column_aoRg(ix, iy, iz)
    else:
        # ngrid = mydf.ngridPrim
        permutation = mydf._get_permutation_column_aoR(ix, iy, iz)

    ###### calculate ######

    res = []
    for i in range(nspin):
        res.append([])

    for i in range(mydf.natmPrim):

        _permutation_ = permutation[i]

        if build_embRg:
            aoR_i = mydf.aoRg[i]
        else:
            aoR_i = mydf.aoR[i]

        ngrid_i = aoR_i.aoR.shape[1]
        nao_involved_i = aoR_i.aoR.shape[0]

        for j in nspin:
            mocoeff_packed = buffer.malloc(
                (nao_involved_i, nemb), dtype=FLOAT64, name="mocoeff_packed"
            )

            TAKE(C_ao_emb[j], _permutation_, axis=0, out=mocoeff_packed)
            ddot_res = buffer.malloc((nemb, ngrid_i), dtype=FLOAT64, name="ddot_res")
            DOT(mocoeff_packed.T, aoR_i.aoR, c=ddot_res)
            embR = ToNUMPY(ddot_res)
            # cutoff #
            embR_max = np.max(np.abs(embR), axis=1)
            embR_row_max_id = np.where(embR_max > aoR_cutoff)[0]
            embR_cutoff = ToTENSOR(embR[embR_row_max_id].copy())
            embR_rowid = ToTENSOR(np.array(embR_row_max_id, dtype=np.int64))
            res[j].append(
                aoR_Holder(
                    embR_cutoff,
                    embR_rowid,
                    aoR_i.global_gridID_begin,  # NOTE :should not add ILOC * ngrid
                    aoR_i.global_gridID_end,
                )
            )
            buffer.free(count=2)

    return res


def get_emb_eri_isdf(
    mydf,
    C_ao_emb: np.ndarray = None,
    symmetry=4,
    aoR_cutoff=1e-8,
    use_mpi=None,
    AOPAIR_BLKSIZE=AOPAIR_BLKSIZE,
):
    #### preprocess ####

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast
        from pyscf.isdf.isdf_tools_mpi import reduce as mpi_reduce
    else:
        rank = 0
        comm = None
        comm_size = 1
        mpi_reduce = None

    assert isinstance(mydf, ISDF_Local_K)

    log = lib.logger.Logger(mydf.stdout, mydf.verbose)
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    C_ao_emb = ToTENSOR(C_ao_emb)
    if C_ao_emb.ndim == 2:
        C_ao_emb = C_ao_emb.reshape(1, *C_ao_emb.shape)
    assert C_ao_emb.ndim == 3
    assert C_ao_emb.dtype == FLOAT64  ## supercell basis

    nspin, nao_full, nemb = C_ao_emb.shape
    supercell = mydf.cell
    assert supercell.nao == nao_full

    with_robust_fitting = mydf.with_robust_fitting
    direct = mydf.direct

    if direct:
        coulG = mydf.coulG
        coulG = (
            coulG.reshape(*mydf.mesh)[:, :, : mydf.mesh[2] // 2 + 1].reshape(-1).copy()
        )
    else:
        coulG = None

    ## allocate buffer ##

    nnmo = nemb * (nemb + 1) // 2
    nnspin = nspin * (nspin + 1) // 2
    natmPrim = mydf.natmPrim

    res = ZEROS((nnspin, nnmo, nnmo), cpu=not USE_GPU)
    if with_robust_fitting:
        res_V = ZEROS((nnspin, nnmo, nnmo), cpu=not USE_GPU)
    else:
        res_V = None
    res_ddot_buf = None

    size_aoRg_packed = (
        nemb
        * mydf.max_group_naux_possible(group=mydf.group)
        * np.prod(mydf.kmesh)
        * nspin
    )
    if (size_aoRg_packed + np.prod(C_ao_emb.shape)) * 8 > AOPAIR_BLKSIZE:
        raise ValueError(
            "The AOPAIR_BLKSIZE is too small for the current system, "
            "please increase the AOPAIR_BLKSIZE"
        )
    size0 = nnmo  # store moPairRgBra
    # size1 = nemb**2  # construct moPairRgBra
    size2 = 0  # build V and W
    if direct:
        if with_robust_fitting:
            size2 += mydf.ngrids
        size2 += mydf.naux
    size3 = nnmo  # store moPairRVKet
    size4 = nnmo  # store moPairRgKet
    size5 = nnmo + nemb**2  # build moPairRgKet
    size = size0 + size2 + size3 + size4 + size5

    GRID_BUNCHIZE = int((AOPAIR_BLKSIZE - size_aoRg_packed * 8) / (size * 8))

    buffer = SimpleMemoryAllocator(
        size * GRID_BUNCHIZE + size_aoRg_packed, FLOAT64, gpu=USE_GPU
    )
    if direct:
        buffer_fft = DynamicCached3DRFFT((GRID_BUNCHIZE, *mydf.mesh), mydf.mesh)
    else:
        buffer_fft = None

    ###############################

    if with_robust_fitting:
        emb_R = []
    else:
        emb_R = None
    emb_Rg = []
    for i in range(nspin):
        if with_robust_fitting:
            emb_R.append([])
        emb_Rg.append([])

    for kx in range(mydf.kmesh[0]):
        for ky in range(mydf.kmesh[1]):
            for kz in range(mydf.kmesh[2]):
                if with_robust_fitting:
                    emb_R_k = _get_embR_box(
                        mydf,
                        C_ao_emb,
                        kx,
                        ky,
                        kz,
                        mydf.kmesh,
                        build_embRg=False,
                        aoR_cutoff=aoR_cutoff,
                        buffer=buffer,
                    )
                    for i in range(nspin):
                        emb_R[i].extend(emb_R_k[i])
                        for x in emb_R_k[i]:
                            log.info(
                                "emb values on grids box %2d %2d %2d for spin %1d, nao : %4d/%4d"
                                % (kx, ky, kz, i, x.aoR.shape[0], nemb)
                            )
                emb_Rg_k = _get_embR_box(
                    mydf,
                    C_ao_emb,
                    kx,
                    ky,
                    kz,
                    mydf.kmesh,
                    build_embRg=True,
                    aoR_cutoff=aoR_cutoff,
                    buffer=buffer,
                )
                for i in range(nspin):
                    emb_Rg[i].extend(emb_Rg_k[i])
                    for x in emb_Rg_k[i]:
                        log.info(
                            "emb values on grids box %2d %2d %2d for spin %1d, nao : %4d/%4d"
                            % (kx, ky, kz, i, x.aoR.shape[0], nemb)
                        )

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    misc._benchmark_time(t1, t2, "get emb_R", mydf, mydf.rank)

    ###############################

    # perform the contraction group by group bunch by bunch #

    group = mydf.group
    group_begin, group_end = _range_partition(len(group), rank, comm_size, use_mpi)
    assert group_begin == mydf.group_begin
    assert group_end == mydf.group_end

    # permutation of VW when get ERI #

    permutation_VW = []
    kmesh = mydf.kmesh

    for brax, bray, braz in product(range(kmesh[0]), range(kmesh[1]), range(kmesh[2])):
        permutation_VW.append([])
        for ketx, kety, ketz in product(
            range(kmesh[0]), range(kmesh[1]), range(kmesh[2])
        ):
            kx = (ketx - brax + kmesh[0]) % kmesh[0]
            ky = (kety - bray + kmesh[1]) % kmesh[1]
            kz = (ketz - braz + kmesh[2]) % kmesh[2]
            permutation_VW[-1].append(kx * kmesh[1] * kmesh[2] + ky * kmesh[2] + kz)

    # moPairInd #

    if with_robust_fitting:
        moPairRInd = []
        for ispin in range(nspin):
            moPairRInd.append([])
            for _moR_ in emb_R[ispin]:
                row_indices, col_indices = np.tril_indices(_moR_.nao_involved)
                row_indices = ToTENSOR(np.array(row_indices, dtype=np.int64))
                col_indices = ToTENSOR(np.array(col_indices, dtype=np.int64))
                indices = (
                    _moR_.ao_involved[row_indices]
                    * (_moR_.ao_involved[row_indices] + 1)
                    // 2
                    + _moR_.ao_involved[col_indices]
                )
                moPairRInd[ispin].append(indices)
    else:
        moPairRInd = None
    moPairRgInd = []
    for ispin in range(nspin):
        for _moRg_ in emb_Rg[ispin]:
            row_indices, col_indices = np.tril_indices(_moRg_.nao_involved)
            row_indices = ToTENSOR(np.array(row_indices, dtype=np.int64))
            col_indices = ToTENSOR(np.array(col_indices, dtype=np.int64))
            indices = (
                _moRg_.ao_involved[row_indices]
                * (_moRg_.ao_involved[row_indices] + 1)
                // 2
                + _moRg_.ao_involved[col_indices]
            )
            moPairRgInd[ispin].append(indices)

    # cache indices take #

    indices_take_cached = {}

    def _find_indices_take(nmo):
        if nmo not in indices_take_cached:
            row_indices, col_indices = np.tril_indices(nmo)
            indices_take = ToTENSOR(row_indices * nmo + col_indices)
            indices_take_cached[nmo] = indices_take
        return indices_take_cached[nmo]

    # perform calculation #

    assert nspin <= 2
    if nspin == 1:
        spin_lst = [[0, 0]]
    else:
        spin_lst = [[0, 0], [1, 1], [0, 1]]
    IP_begin_loc = 0

    for group_id in range(group_begin, group_end):

        # pack moRg #

        moRg_packed = []
        indices_take = []
        indices_add = []
        nIP_i = None

        for ispin in range(nspin):
            moRg_packed.append([])
            indices_take.append([])
            indices_add.append([])
            IBOX = 0
            for ibox_x, ibox_y, ibox_z in product(
                range(mydf.kmesh[0]), range(mydf.kmesh[1]), range(mydf.kmesh[2])
            ):

                moRg_unpacked = [
                    emb_Rg[ispin][atm_id + IBOX * natmPrim]
                    for atm_id in group[group_id]
                ]
                nIP_involved = sum(
                    [moRg.aoR.shape[1] for moRg in moRg_unpacked if moRg is not None]
                )
                assert (
                    nIP_involved
                    == mydf.IP_segment[group_id + 1] - mydf.IP_segment[group_id]
                )
                nIP_i = nIP_involved

                packed_buf_moRg = buffer.malloc(
                    (nemb, nIP_involved), dtype=FLOAT64, name="packed_buf_moRg"
                )
                CLEAN(packed_buf_moRg)
                moRg_packed_tmp = _pack_aoR_holder(
                    moRg_unpacked, nemb, out_buf=packed_buf_moRg
                )

                moRg_packed[ispin].append(moRg_packed_tmp)

                # get the indices #

                nmo_tmp = moRg_packed_tmp.nao_involved
                row_indices, col_indices = np.tril_indices(nmo_tmp)
                indices_take[ispin].append(_find_indices_take(nmo_tmp))
                indices_add[ispin].append(
                    ToTENSOR(
                        moRg_packed.ao_involved[row_indices] * nemb
                        + moRg_packed.ao_involved[col_indices]
                    )
                )

                IBOX += 1

        for p0, p1 in lib.prange(0, nIP_i, GRID_BUNCHIZE):
            # (1) build V and W #
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
                    coulG,
                )  # V_tmp is stored in buffer_fft
                W_tmp = buffer.malloc((p1 - p0, mydf.naux), dtype=FLOAT64, name="W_tmp")
                W_tmp = _build_W_local_bas_k_kernel(
                    V_tmp, mydf.aux_basis, mydf.kmesh, W_tmp
                )
            else:
                if with_robust_fitting:
                    V_tmp = mydf.V[IP_begin_loc + p0 : IP_begin_loc + p1, :]
                else:
                    V_tmp = None
                W_tmp = mydf.W[IP_begin_loc + p0 : IP_begin_loc + p1, :]
            ## first loop over all boxes ##
            for bra_x, bra_y, bra_z in product(
                range(mydf.kmesh[0]), range(mydf.kmesh[1]), range(mydf.kmesh[2])
            ):
                bra_box_id = (
                    bra_x * mydf.kmesh[1] * mydf.kmesh[2]
                    + bra_y * mydf.kmesh[2]
                    + bra_z
                )
                pert_VW = permutation_VW[bra_box_id]
                ## second loop over all possible combination of spin ##
                for iERI, (spin_bra, spin_ket) in enumerate(spin_lst):
                    # build moPairRg #
                    bra_nemb = moRg_packed[spin_bra][bra_box_id].nao_involved
                    indices_take_bra = indices_take[spin_bra][bra_box_id]
                    indices_add_bra = indices_add[spin_bra][bra_box_id]
                    moPairRgBra = buffer.malloc(
                        (bra_nemb * (bra_nemb + 1) // 2, p1 - p0),
                        dtype=FLOAT64,
                        name="moPairRgBra",
                    )
                    moPairRgBra2 = buffer.malloc(
                        (bra_nemb * bra_nemb, p1 - p0),
                        dtype=FLOAT64,
                        name="moPairRgBra2",
                    )
                    EINSUM_IK_JK_IJK(
                        moRg_packed[spin_bra][bra_box_id].aoR[:, p0:p1],
                        moRg_packed[spin_bra][bra_box_id].aoR[:, p0:p1],
                        out=moPairRgBra2,
                    )
                    moPairRgBra2 = moPairRgBra2.reshape(nmo_tmp * nmo_tmp, p1 - p0)
                    TAKE(moPairRgBra2, indices_take_bra, 0, out=moPairRgBra)
                    buffer.free(count=1)
                    # do the calculation #
                    ## V W must be permutated ##
                    ## V term
                    if with_robust_fitting:
                        pass
                    ## W term
                    moPairRgWKet = buffer.malloc(
                        (nemb * (nemb + 1) // 2, p1 - p0),
                        dtype=FLOAT64,
                        name="moPairRgWKet",
                    )
                    CLEAN(moPairRgWKet)
                    for ket_x, ket_y, ket_z in product(
                        range(mydf.kmesh[0]), range(mydf.kmesh[1]), range(mydf.kmesh[2])
                    ):
                        ket_box_id = (
                            ket_x * mydf.kmesh[1] * mydf.kmesh[2]
                            + ket_y * mydf.kmesh[2]
                            + ket_z
                        )
                        pert_W = pert_VW[ket_box_id]
                        for atmid, _moRg_ in enumerate(
                            emb_Rg[spin_ket][
                                ket_box_id * natmPrim : ket_box_id * (natmPrim + 1)
                            ]
                        ):
                            ket_nemb = _moRg_.nao_involved
                            ngrids_ket = _moRg_.aoR.shape[1]
                            W_grid_begin_ID = _moRg_.global_gridID_begin
                            indices_take_ket = _find_indices_take(ket_nemb)
                            indices_add_ket = moPairRgInd[spin_ket][
                                ket_box_id * natmPrim + atmid
                            ]
                            moPairRgKet = buffer.malloc(
                                (ket_nemb * (ket_nemb + 1) // 2, p1 - p0),
                                dtype=FLOAT64,
                                name="moPairRgKet",
                            )
                            CLEAN(moPairRgKet)
                            # do the contraction #
                            # add it back #
                            if ket_nemb == nemb:
                                moPairRgWKet += moPairRgKet
                            else:
                                INDEX_ADD(moPairRgWKet, 0, indices_add_ket, moPairRgKet)
                            buffer.free(count=1)
                    ## do the final dot ##
                    if bra_nemb == nemb:
                        DOT(moPairRgBra, moPairRgWKet.T, c=res[iERI], beta=1)
                    else:
                        if res_ddot_buf is None:
                            res_ddot_buf = MALLOC(
                                (nnmo, nnmo), dtype=FLOAT64, gpu=not USE_GPU
                            )
                        ddot_res = buffer.malloc(
                            (bra_nemb * (bra_nemb + 1) // 2, nnmo),
                            dtype=FLOAT64,
                            buf=res_ddot_buf,
                        )
                        DOT(moPairRgBra, moPairRgWKet.T, c=ddot_res)
                        INDEX_ADD(res[iERI], ddot_res, indices_add_bra, ddot_res)
                    buffer.free(count=1)

            if direct:
                buffer.free(count=2)

        buffer.free(count=np.prod(kmesh) * len(group[group_id]) * nspin)
        IP_begin_loc += nIP_i

    ###############################

    ###############################

    ## post process ##

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

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    misc._benchmark_time(t1, t2, "get_emb_eri_isdf", mydf, mydf.rank)

    del res_ddot_buf

    return ToNUMPY(res) * mydf.ngrids / mydf.cell.vol
