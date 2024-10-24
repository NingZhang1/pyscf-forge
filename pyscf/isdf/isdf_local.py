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

import tempfile
import copy
import numpy as np
import scipy, numpy
import h5py, pickle
import ctypes, sys, os
import random
import string

############ pyscf module ############

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools

try:
    from pyscf.lib import generate_pickle_methods
except ImportError:
    print("pyscf.lib.generate_pickle_methods is not available, ISDF will not support pickle")

############ isdf backends ############

import pyscf.isdf.BackEnd.isdf_backend as BACKEND
from pyscf.isdf.BackEnd.isdf_memory_allocator import SimpleMemoryAllocator

USE_GPU = BACKEND.USE_GPU
NUM_THREADS = BACKEND.NUM_THREADS

if USE_GPU == 1:
    from pyscf.isdf.BackEnd.isdf_fft_cache import (
        DynamicCached3DRFFT_GPU as DynamicCached3DRFFT,
    )
else:
    from pyscf.isdf.BackEnd.isdf_fft_cache import DynamicCached3DRFFT

assert (USE_GPU == 0) or (not USE_GPU), "ISDF_Local does not support GPU backend!"

FLOAT64 = BACKEND.FLOAT64
ToNUMPY = BACKEND._toNumpy
ToTENSOR = BACKEND._toTensor
ZEROS = BACKEND._zeros
MAX = BACKEND._maximum
ABS = BACKEND._absolute
DOT = BACKEND._dot
TAKE = BACKEND._take
CLEAN = BACKEND._clean
MALLOC = BACKEND._malloc

############ isdf utils ############

import pyscf.isdf.isdf as isdf
import pyscf.isdf.misc as misc
from pyscf.isdf.isdf_tools_local import (
    aoR_Holder,
    build_cutoff_info,
    get_partition,
    get_aoR,
    _pack_aoR_holder,
    _range_partition,
    _sync_list,
    _get_grid_ordering,
    _get_aoR_holders_memory,
)
import pyscf.isdf.isdf_local_jk as isdf_local_jk


def generate_string():
    chars = string.ascii_letters + string.digits
    return "tmp_" + "".join(random.choice(chars) for _ in range(12 - len("tmp_")))


############ subroutines --- select IP ############


def select_IP_local_step1(
    mydf,
    c: int,
    m: int,
    first_natm=None,
    rela_cutoff=0.0,
    no_retriction_on_nIP=False,
    use_mpi=False,
):
    # NOTE: this subroutine assume that all the calculations are performed on CPU #

    # preprocess #

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size
    else:
        rank = 0
        comm_size = 1

    assert isinstance(mydf.aoR, list)
    assert isinstance(mydf.partition, list)

    if first_natm is None:
        first_natm = mydf.first_natm

    # buffer #

    buffer = mydf.buffer_cpu
    buffer.free_all()

    # funcs #

    QR = BACKEND._qr
    QR_PIVOT = BACKEND._qr_col_pivoting
    EINSUM_IK_JK_IJK = BACKEND._einsum_ik_jk_ijk

    ### loop over atm ###

    # split tasks if use_mpi #

    atm_begin, atm_end = _range_partition(first_natm, rank, comm_size, use_mpi)

    results = [None] * first_natm

    misc._debug4(
        mydf,
        rank,
        " ---------- In pyscf.isdf.isdf_local.select_IP_local_atm ----------",
    )
    misc._debug4(
        mydf,
        rank,
        "select_IP_local_atm: atm_begin = %4d, atm_end = %4d",
        atm_begin,
        atm_end,
    )

    for atm_id in range(atm_begin, atm_end):
        aoR = mydf.aoR[atm_id]
        if aoR is None:
            continue

        grid_ID = ToNUMPY(mydf.partition[atm_id])
        aoR_atm = aoR.aoR
        nao_tmp = aoR_atm.shape[0]

        # random projection #

        nao_atm = mydf.atmID2nao[atm_id]
        naux_now = int(np.sqrt(nao_atm * c) + m)
        naux_now = min(naux_now, nao_tmp)
        naux2_now = naux_now**2

        G1 = ToTENSOR(np.random.randn(nao_tmp, naux_now), cpu=True)
        G1, _ = QR(G1, mode="economic")
        G1 = G1.T
        G2 = ToTENSOR(np.random.randn(nao_tmp, naux_now), cpu=True)
        G2, _ = QR(G2, mode="economic")
        G2 = G2.T

        aoR_atm1 = buffer.malloc(
            (naux_now, grid_ID.shape[0]), dtype=FLOAT64, name="aoR_atm1"
        )
        aoR_atm2 = buffer.malloc(
            (naux_now, grid_ID.shape[0]), dtype=FLOAT64, name="aoR_atm2"
        )
        aoPairR = buffer.malloc(
            (naux_now, naux_now, grid_ID.shape[0]), dtype=FLOAT64, name="aoPairR"
        )

        DOT(G1, aoR_atm, c=aoR_atm1)
        DOT(G2, aoR_atm, c=aoR_atm2)
        EINSUM_IK_JK_IJK(aoR_atm1, aoR_atm2, out=aoPairR)
        aoPairR = aoPairR.reshape(naux2_now, grid_ID.shape[0])

        # qr pivot #

        _, R, pivot = QR_PIVOT(aoPairR, mode="r")

        # determine the pnts to select #

        if no_retriction_on_nIP:
            max_rank = min(naux2_now, grid_ID.shape[0])
        else:
            max_rank = min(naux2_now, grid_ID.shape[0], nao_atm * c + m)
        R = ToNUMPY(R)
        threshold = abs(R[0, 0]) * rela_cutoff
        indices = np.where(np.abs(np.diag(R)) > threshold)[0]
        npt_found = min(len(indices), max_rank)
        pivot = ToNUMPY(pivot[:npt_found])
        pivot.sort()
        results[atm_id] = ToNUMPY(grid_ID[pivot]).copy()
        results[atm_id].sort()

        misc._debug4(mydf, rank, "select_IP_local_atm: npt_found = %4d", npt_found)

        buffer.free_all()

    if use_mpi:
        # from pyscf.isdf.isdf_tools_local import _sync_list

        results = _sync_list(results, first_natm)

    assert len(results) == first_natm

    results = [ToTENSOR(x, cpu=True) if x is not None else None for x in results]

    return results


def _select_IP_local_group(
    mydf,
    aoRg_possible,
    c: int,
    m: int,
    rela_cutoff=0.0,
    group=None,
    atm_2_IP_possible=None,
):
    ### WARNING: this subroutine has bugs ###

    assert isinstance(aoRg_possible, list)
    assert isinstance(group, list) or isinstance(group, np.ndarray)
    assert isinstance(atm_2_IP_possible, list)
    assert len(aoRg_possible) == len(atm_2_IP_possible)

    if group is None:
        raise ValueError("group is not set!")

    # buffer #

    buffer = mydf.buffer_cpu
    buffer.free_all()

    # funcs #

    QR = BACKEND._qr
    QR_PIVOT = BACKEND._qr_col_pivoting
    EINSUM_IK_JK_IJK = BACKEND._einsum_ik_jk_ijk

    ##### random projection #####

    nao_group = sum(mydf.atmID2nao[group])
    misc._debug4(mydf, mydf.rank, "select_IP_local_group: nao_group = %4d", nao_group)
    nao = mydf.nao

    max_naux_group = mydf.max_group_naux_possible(c, m, [group])
    packed_buf = buffer.malloc((nao, max_naux_group), dtype=FLOAT64, name="packed_buf")

    # pack aoRg #

    aoRg_unpacked = []
    for atm_id in group:
        aoRg_unpacked.append(aoRg_possible[atm_id])
    if len(aoRg_unpacked) == 1:
        aoRg_packed = aoRg_unpacked[0].aoR
    else:
        CLEAN(packed_buf)
        aoRg_packed = _pack_aoR_holder(aoRg_unpacked, nao, out_buf=packed_buf).aoR
    nao = aoRg_packed.shape[0]
    ngrids = aoRg_packed.shape[1]

    # build IP_possible #

    IP_possible = []
    for atm_id in group:
        if atm_2_IP_possible[atm_id] is None:
            continue
        IP_possible.extend(list(atm_2_IP_possible[atm_id]))
    IP_possible = np.array(IP_possible, dtype=np.int32)

    # random projection #

    naux_now = int(np.sqrt(c * nao_group)) + m
    naux_now = min(naux_now, nao)
    G1 = ToTENSOR(np.random.randn(nao, naux_now), cpu=True)
    G1, _ = QR(G1, mode="economic")
    G1 = G1.T
    G2 = ToTENSOR(np.random.randn(nao, naux_now), cpu=True)
    G2, _ = QR(G2, mode="economic")
    G2 = G2.T

    aoR_atm1 = buffer.malloc((naux_now, ngrids), dtype=FLOAT64, name="aoR_atm1")
    aoR_atm2 = buffer.malloc((naux_now, ngrids), dtype=FLOAT64, name="aoR_atm2")
    aoPairR = buffer.malloc((naux_now, naux_now, ngrids), dtype=FLOAT64, name="aoPairR")

    DOT(G1, aoRg_packed, c=aoR_atm1)
    DOT(G2, aoRg_packed, c=aoR_atm2)
    EINSUM_IK_JK_IJK(aoR_atm1, aoR_atm2, out=aoPairR)
    aoPairR = aoPairR.reshape(naux_now**2, ngrids)

    if mydf.no_retriction_on_nIP:
        max_rank = min(naux_now**2, len(IP_possible))
    else:
        max_rank = min(naux_now**2, len(IP_possible), nao_group * c)

    misc._debug4(mydf, mydf.rank, "select_IP_local_group: naux_now  = %4d", naux_now)
    misc._debug4(mydf, mydf.rank, "select_IP_local_group: max_rank  = %4d", max_rank)

    # qr pivot #

    _, R, pivot = QR_PIVOT(aoPairR, mode="r")

    R = ToNUMPY(R)
    threshold = abs(R[0, 0]) * rela_cutoff
    indices = np.where(np.abs(np.diag(R)) > threshold)[0]
    npt_found = min(len(indices), max_rank)
    pivot = ToNUMPY(pivot[:npt_found])
    pivot.sort()
    results = list(IP_possible[pivot])
    results.sort()
    results = np.array(results, dtype=np.int32)

    misc._debug4(mydf, mydf.rank, "select_IP_local_group: npt_found = %4d", npt_found)

    del R
    del G1
    del G2

    buffer.free_all()

    return ToTENSOR(results)


def select_IP_local_step2(
    mydf,
    c,
    m,
    rela_cutoff,
    group,
    IP_possible_atm,
    aoRg_possible,
    global_IP_selection=True,
    use_mpi=False,
):
    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size
    else:
        rank = 0
        comm_size = 1

    # IP_group = [None] * len(group)
    IP_group = [None for _ in range(len(group))]

    assert len(IP_possible_atm) == mydf.first_natm

    # find IP for each group #

    if len(group) < mydf.first_natm:
        if global_IP_selection:
            # split tasks #

            group_begin, group_end = _range_partition(
                len(group), rank, comm_size, use_mpi
            )

            for i in range(group_begin, group_end):
                IP_group[i] = _select_IP_local_group(
                    mydf,
                    aoRg_possible=aoRg_possible,
                    c=c,
                    m=m,
                    rela_cutoff=rela_cutoff,
                    group=group[i],
                    atm_2_IP_possible=IP_possible_atm,
                )

            if use_mpi:
                comm.barrier()
                IP_group = _sync_list(IP_group, len(group))
        else:
            # print("no global selection")
            # print("group = ", group)
            # group_begin, group_end = 0, len(group)
            group_begin, group_end = _range_partition(
                len(group), rank, comm_size, use_mpi
            )
            for i in range(len(group)):
                IP_group[i] = []
                for atm_id in group[i]:
                    # print(IP_possible_atm[atm_id])
                    IP_group[i].extend(list(IP_possible_atm[atm_id]))
                IP_group[i] = np.array(IP_group[i], dtype=np.int32)
                IP_group[i].sort()
                # print(IP_group[i])
                IP_group[i] = ToTENSOR(IP_group[i])

    else:
        # group_begin, group_end = 0, len(group)
        group_begin, group_end = _range_partition(len(group), rank, comm_size, use_mpi)
        IP_group = IP_possible_atm

    mydf.group_begin = group_begin
    mydf.group_end = group_end
    mydf.IP_group = [ToTENSOR(x, cpu=True) if x is not None else None for x in IP_group]

    return IP_group


def select_IP_local_step3(mydf, group, use_mpi=False):
    from itertools import accumulate

    #### build indices info related to IP_group ####

    # Flatten IP_group and create IP_segment

    IP_group_NUMPY = [ToNUMPY(x) if x is not None else None for x in mydf.IP_group]
    mydf.IP_flat = ToTENSOR(np.concatenate(IP_group_NUMPY).astype(np.int32))
    mydf.IP_segment = list(accumulate([0] + [len(x) for x in IP_group_NUMPY]))
    mydf.naux = len(mydf.IP_flat)
    mydf.IP_segment.append(mydf.naux)
    mydf.IP_segment = ToTENSOR(mydf.IP_segment, cpu=True)

    # Create partition_IP

    partition_IP = [[] for _ in range(mydf.first_natm)]
    for ip_id in mydf.IP_flat:
        # assert mydf.gridID_2_atmID[ip_id] < mydf.first_natm
        partition_IP[mydf.gridID_2_atmID[ip_id]].append(ip_id)

    # Convert to sorted numpy arrays

    partition_IP = [np.sort(np.array(p, dtype=np.int32)) for p in partition_IP]
    mydf.partition_IP = [ToTENSOR(p, cpu=True) for p in partition_IP]


############ subroutines --- build aux_basis ############


def _find_common_elements_positions(arr1, arr2):
    arr1 = ToNUMPY(arr1)
    arr2 = ToNUMPY(arr2)

    # Create a dictionary to store elements and their positions in arr1
    elem_positions = {elem: i for i, elem in enumerate(arr1)}

    # Find common elements and their positions
    common_positions = [
        (elem_positions[elem], i)
        for i, elem in enumerate(arr2)
        if elem in elem_positions
    ]

    # If common elements exist, return two arrays of positions; otherwise, return empty arrays
    if common_positions:
        common_positions = np.array(common_positions).T
        return common_positions[0], common_positions[1]
    else:
        return np.array([], dtype=int), np.array([], dtype=int)


def build_aux_basis_local(mydf, group, IP_group, use_mpi=False):
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size
    else:
        rank = 0
        comm_size = 1

    # buffer #

    buffer = mydf.buffer_cpu
    buffer.free_all()

    # funcs #

    TAKE = BACKEND._take
    SQUARE_ = BACKEND._square_
    CHO_SOLVE = BACKEND._solve_cholesky

    # split task #

    group_begin, group_end = _range_partition(len(group), rank, comm_size, use_mpi)

    assert group_begin == mydf.group_begin
    assert group_end == mydf.group_end

    # build aux basis #

    mydf.aux_basis = [None] * len(group)

    for i in range(group_begin, group_end):
        aoRg_unpacked = [mydf.aoRg[atm_id] for atm_id in group[i]]
        aoR_unpacked = [mydf.aoR[atm_id] for atm_id in group[i]]

        max_group_ngrids_possible = mydf.max_group_ngrids_possbile(group)
        max_group_naux_possible = mydf.max_group_naux_possible(group=group)

        packed_buf_aoR = buffer.malloc(
            (mydf.nao, max_group_ngrids_possible), dtype=FLOAT64, name="packed_buf_aoR"
        )
        packed_buf_aoRg = buffer.malloc(
            (mydf.nao, max_group_naux_possible), dtype=FLOAT64, name="packed_buf_aoRg"
        )
        CLEAN(packed_buf_aoR)
        CLEAN(packed_buf_aoRg)

        aoR1 = _pack_aoR_holder(aoR_unpacked, mydf.nao, out_buf=packed_buf_aoR)
        aoRg1 = _pack_aoR_holder(aoRg_unpacked, mydf.nao, out_buf=packed_buf_aoRg)

        ### check whether the number of invovled nao is the same ###

        if aoRg1.aoR.shape[0] == aoR1.aoR.shape[0]:
            aoRg1 = aoRg1.aoR
            aoR1 = aoR1.aoR
        else:
            pos1, pos2 = _find_common_elements_positions(
                aoRg1.ao_involved, aoR1.ao_involved
            )
            # print(pos1, pos2)
            assert len(pos1) == aoRg1.aoR.shape[0]
            aoRg1 = aoRg1.aoR
            # aoR1 = aoR1.aoR[pos2, :]
            aoR1 = TAKE(aoR1, ToTENSOR(pos2, cpu=True), axis=0, out=aoR1)

        naux_now = aoRg1.shape[1]
        ngrids_now = aoR1.shape[1]

        A = buffer.malloc((naux_now, naux_now), dtype=FLOAT64, name="A")
        B = buffer.malloc((naux_now, ngrids_now), dtype=FLOAT64, name="B")
        DOT(aoRg1.T, aoRg1, c=A)
        SQUARE_(A)
        DOT(aoRg1.T, aoR1, c=B)
        SQUARE_(B)

        # build aux basis, AX=B #

        mydf.aux_basis[i] = CHO_SOLVE(A, B, overwrite_b=False)

        buffer.free_all()

    if use_mpi:
        comm.barrier()
        mydf.aux_basis = _sync_list(mydf.aux_basis, len(group))

    mydf.aux_basis = [ToTENSOR(x, cpu=True) for x in mydf.aux_basis]

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

    misc._benchmark_time(t1, t2, "build_aux_basis_local", mydf, mydf.rank)

    return mydf.aux_basis


############ subroutines --- build V W ############


def build_V_W_local(mydf, use_mpi=False):
    assert not mydf.direct

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size
    else:
        rank = 0
        # comm_size = 1

    misc._debug4(
        mydf, rank, " ---------- In pyscf.isdf.isdf_local.build_V_W_local ----------"
    )

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    # buffer #

    buffer = mydf.buffer_cpu
    buffer_fft = mydf.buffer_fft

    # task info #

    group_begin_id, group_end_id = mydf.group_begin, mydf.group_end
    naux_involved = mydf.IP_segment[group_end_id] - mydf.IP_segment[group_begin_id]
    naux_tot = mydf.naux  ## NOTE: the meaning of mydf.naux
    ngrids = mydf.ngrids
    bunchsize = mydf._build_V_K_bunchsize

    mydf.W = ZEROS((naux_involved, naux_tot), dtype=FLOAT64, cpu=True)
    if mydf.with_robust_fitting:
        mydf.V = ZEROS((naux_involved, ngrids), dtype=FLOAT64, cpu=True)
    else:
        mydf.V = None

    ## print memory info ##

    memory_V = mydf.V.nbytes if mydf.V is not None else 0
    memory_W = mydf.W.nbytes

    misc._info(mydf, rank, "Memory usage for V: %10.3f MB", memory_V / 1024 / 1024)
    misc._info(mydf, rank, "Memory usage for W: %10.3f MB", memory_W / 1024 / 1024)

    ## do the work ##

    from pyscf.isdf._isdf_local_K_kernel import (
        _build_V_local_bas_kernel,
        _build_W_local_bas_kernel,
    )

    # coul_G = tools.get_coulG(mydf.cell, mesh=mydf.mesh)
    coul_G = mydf.coul_G
    coul_G = ToTENSOR(coul_G, cpu=True).reshape(*mydf.mesh)
    coul_G = ToTENSOR(ToNUMPY(coul_G[:, :, : mydf.mesh[2] // 2 + 1].reshape(-1)).copy())

    V_loc = 0
    for group_id in range(group_begin_id, group_end_id):
        naux_tmp = mydf.IP_segment[group_id + 1] - mydf.IP_segment[group_id]

        for p0, p1 in lib.prange(0, naux_tmp, bunchsize):
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
            )

            if mydf.with_robust_fitting:
                mydf.V[V_loc : V_loc + V_tmp.shape[0], :] = V_tmp

            # loop over all aux basis to construct W #

            _build_W_local_bas_kernel(
                V_tmp, mydf.aux_basis, mydf.W[V_loc : V_loc + V_tmp.shape[0]]
            )

            V_loc += V_tmp.shape[0]

            buffer.free(count=1)

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

    misc._benchmark_time(t1, t2, "build_V_W_local", mydf, rank)

    if use_mpi:
        comm.barrier()

    return mydf.V, mydf.W


def _create_h5file(filename, dataname, mydf=None):
    filename = filename.name if hasattr(filename, "name") else filename
    if isinstance(getattr(filename, "name", None), str):
        filename = filename.name

    if h5py.is_hdf5(filename):
        feri = lib.H5FileWrap(filename, "a")
        misc._debug4(mydf, mydf.rank, "file already exists: %s" % filename)

        if dataname in feri:
            misc._debug4(mydf, mydf.rank, "delete %s" % dataname)
            del feri[dataname]
    else:
        misc._debug4(mydf, mydf.rank, "create new file: %s" % filename)
        feri = lib.H5FileWrap(filename, "w")

    return feri


def build_V_W_local_outcore(mydf, use_mpi=False):
    assert not (mydf.direct is False)

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size
    else:
        rank = 0
        # comm_size = 1

    misc._debug4(
        mydf, rank, " ---------- In pyscf.isdf.isdf_local.build_V_W_local_outcore ----------"
    )

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    # buffer #

    buffer = mydf.buffer_cpu
    buffer_fft = mydf.buffer_fft

    # task info #

    group_begin_id, group_end_id = mydf.group_begin, mydf.group_end
    naux_involved = mydf.IP_segment[group_end_id] - mydf.IP_segment[group_begin_id]
    naux_tot = mydf.naux  ## NOTE: the meaning of mydf.naux
    # ngrids = mydf.ngrids
    bunchsize = mydf._build_V_K_bunchsize

    # use the file specified by mydf._isdf_to_save
    swapfile = mydf._isdf_to_save + ".w.h5"
    mydf._swapfile = swapfile  # one must create sth to hold swapfile, so that it will not be wrongly deleted
    mydf.W = _create_h5file(swapfile, "W")
    h5d_eri = mydf.W.create_dataset("W", (naux_involved, naux_tot), "f8")
    if mydf.with_robust_fitting:
        raise NotImplementedError("outcore mode does not support robust fitting!")
    else:
        mydf.V = None

    ## print memory info ##

    # memory_V = mydf.V.nbytes if mydf.V is not None else 0
    # memory_W = mydf.W.nbytes
    # misc._info(mydf, rank, "Memory usage for V: %10.3f MB", memory_V / 1024 / 1024)
    # misc._info(mydf, rank, "Memory usage for W: %10.3f MB", memory_W / 1024 / 1024)

    ## do the work ##

    from pyscf.isdf._isdf_local_K_kernel import (
        _build_V_local_bas_kernel,
        _build_W_local_bas_kernel,
    )

    # coul_G = tools.get_coulG(mydf.cell, mesh=mydf.mesh)
    coul_G = mydf.coul_G
    coul_G = ToTENSOR(coul_G, cpu=True).reshape(*mydf.mesh)
    coul_G = ToTENSOR(ToNUMPY(coul_G[:, :, : mydf.mesh[2] // 2 + 1].reshape(-1)).copy())

    buf1 = buffer.malloc((bunchsize, naux_tot), dtype=FLOAT64, name="buf1")
    buf2 = buffer.malloc((bunchsize, naux_tot), dtype=FLOAT64, name="buf2")

    log = logger.Logger(mydf.stdout, mydf.verbose)
    time1 = (logger.process_clock(), logger.perf_counter())

    W_loc = 0
    ISTEP = 0
    for group_id in range(group_begin_id, group_end_id):
        naux_tmp = mydf.IP_segment[group_id + 1] - mydf.IP_segment[group_id]

        def process(bunch_range):
            nonlocal buf1, buf2
            buf2, buf1 = buf1, buf2
            istep, (p0, p1) = bunch_range
            W_tmp = MALLOC((p1 - p0, naux_tot), dtype=FLOAT64, buf=buf1, offset=0)
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
            )
            _build_W_local_bas_kernel(V_tmp, mydf.aux_basis, W_tmp)
            buffer.free(count=1)
            return W_tmp

        bunch_range = list(enumerate(lib.prange(0, naux_tmp, bunchsize)))

        for istep, W_tmp in enumerate(lib.map_with_prefetch(process, bunch_range)):
            p0, p1 = bunch_range[istep][1]
            # label = "%s/%d" % ("W", ISTEP)
            # mydf.W[label] = W_tmp
            h5d_eri[W_loc : W_loc + (p1 - p0), :] = W_tmp
            W_tmp = None
            W_loc += p1 - p0
            ISTEP += 1
            # log.debug(
            #     "gen W [%4d|[%4d:%4d] with naux_tmp = %4d", ISTEP, p0, p1, naux_tmp
            # )
            time1 = log.timer("gen W [%4d|[%4d:%4d]" % (ISTEP, p0, p1), *time1)

        mydf.W.flush()
    mydf.W.flush()

    filename = mydf.W.filename
    mydf.W.close()
    mydf.W = filename

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

    misc._benchmark_time(t1, t2, "build_V_W_local", mydf, rank)

    if use_mpi:
        comm.barrier()

    buffer.free(count=2)

    return mydf.V, mydf.W


class ISDF_Local(isdf.ISDF):
    """Interpolative separable density fitting (ISDF) for periodic systems.
    The locality is explored!

    Fitting aux basis is linear scaling!

    Quadratic scaling for constructing V and W matrix as well as build K matrix!

    Examples:
    """

    def __init__(
        self,
        cell: Cell,
        with_robust_fitting=True,
        kmesh=None,
        kpts=None,
        aoR_cutoff=1e-8,
        direct=False,  # should be one of [True, False, "outcore"]
        limited_memory=False,
        build_V_K_bunchsize=None,
        verbose=None,
        use_mpi=False,
    ):
        super().__init__(
            cell,
            with_robust_fitting,
            kmesh,
            kpts,
            verbose,
            use_mpi,
        )

        self.aoR_cutoff = aoR_cutoff
        if isinstance(direct, str):
            assert direct.lower() == "outcore"
            self.direct = True
            self.outcore = True

        else:
            assert isinstance(direct, bool)
            self.outcore = False
            self.direct = direct

            if direct and (not self.with_robust_fitting):
                warn = "Without robust fitting, direct fitting is not useful!"
                misc._warn(self, self.rank, warn)

        nkpts = np.prod(self.kmesh)
        if nkpts > 1:
            self.with_translation_symmetry = True
        else:
            self.with_translation_symmetry = False

        ### some new members ###
        self.shl_atm = None
        self.aoloc_atm = None
        self.partition_group_2_gridID = None
        self.gridID_2_atmID = None
        self.gridID_ordering = None
        self.no_retriction_on_nIP = True

        self.buffer_cpu = None
        self.buffer_gpu = None

        self._force_translation_sym = False
        self._T_mesh = None

        ### limited memory ###

        self._limited_memory = limited_memory
        self._build_V_K_bunchsize = build_V_K_bunchsize
        if build_V_K_bunchsize is None:
            from pyscf.isdf._isdf_local_K_kernel import K_DIRECT_NAUX_BUNCHSIZE
            self._build_V_K_bunchsize = K_DIRECT_NAUX_BUNCHSIZE

        # if _isdf_to_save is specified, useful information will be saved to this file
        self._isdf_to_save = tempfile.TemporaryFile(dir=lib.param.TMPDIR)

        # if _isdf is specified, will read from the file
        self._isdf = None

    # def __del__(self):
    #     if hasattr(self, "W"):
    #         # del self.W
    #         if isinstance(self.W, str):
    #             import os
    #             os.remove(self.W)  # known issue: cannot delete this file

    __getstate__, __setstate__ = generate_pickle_methods(
            excludes=(
                '_isdf_to_save', '_isdf', 'buffer_fft',
                'buffer_cpu', 'buffer_gpu', 'buffer'
            ), reset_state=True
            )

    # build
    def build(
        self, c=None, m=5, rela_cutoff=None, group=None, global_IP_selection=True
    ): 
        if self._isdf is not None:
            assert isinstance(self._isdf, str)
            assert os.path.exists(self._isdf)

            misc._debug4(self, self.rank, "try to read results from %s" % self._isdf)
            misc._debug4(self, self.rank, "_isdf_to_save will be ignored")
            self._isdf_to_save = None

            self = pickle.loads(self._isdf)

        # preprocess #
        rela_cutoff = abs(rela_cutoff)

        if group is None:
            group = [[i] for i in range(self.first_natm)]
        else:
            # sort each subgroup #
            for subgroup in group:
                subgroup.sort()

        if c is None:
            if rela_cutoff is not None and rela_cutoff > 1e-6:
                c = self._determine_c(rela_cutoff)
            else:
                c = 15
                rela_cutoff = 0.0

        # some info update #

        self.c = c
        self.m = m
        self.group = group

        self._build_cell_info()
        self._build_aoR(group)

        # some info update #
        self._build_buffer(c, m, group)
        self._build_IP(c, m, rela_cutoff, group, global_IP_selection)
        self._build_fft_buffer()
        self._build_aux_basis(group)
        self._build_V_W()

        if self._isdf_to_save is not None:
            pickle.dump(self, self._isdf_to_save)

    def _build_cell_info(self):
        self.distance_matrix, self.AtmConnectionInfo = build_cutoff_info(
            self.cell, self.aoR_cutoff, self.ngrids
        )
        super()._build_cell_info()

    def _build_aoR(self, group):
        self.group = group

        self.partition = None
        self.aoR = None
        self.partition_group_2_gridID = None
        self.gridID_2_atmID = None
        self.gridID_ordering = None

        from pyscf.isdf.isdf_eval_gto import ISDF_eval_gto

        ##### build partition ####
        lattice_x = self.cell.lattice_vectors()[0][0]
        lattice_y = self.cell.lattice_vectors()[1][1]
        lattice_z = self.cell.lattice_vectors()[2][2]

        Ls = [
            int(lattice_x) // 3 + 6,
            int(lattice_y) // 3 + 6,
            int(lattice_z) // 3 + 6,
        ]

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        if self.partition is None:
            self.partition = get_partition(
                self.cell,
                self.coords,
                self.AtmConnectionInfo,
                Ls,
                self.with_translation_symmetry,
                self.kmesh,
                self.use_mpi,
            )
        self.partition = [ToTENSOR(x) for x in self.partition]
        assert len(self.partition) == self.cell.natm
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

        misc._benchmark_time(t1, t2, "get_partition", self, self.rank)

        if self.rank == 0:
            for x in range(self.natm):
                misc._debug4(
                    self,
                    self.rank,
                    "len of partition[%4d] = %6d",
                    x,
                    len(self.partition[x]),
                )

        ## deal with translation symmetry and build aoR ##

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        first_natm = self.first_natm
        self.aoR = get_aoR(
            self.cell,
            self.AtmConnectionInfo,
            self.coords,
            self.partition,
            None,
            first_natm,
            self.group,
            self.use_mpi,
            self.use_mpi,
        )

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        misc._benchmark_time(t1, t2, "get_aoR", self, self.rank)

        # build some indices info #

        # Create partition_group_2_gridID
        self.partition_group_2_gridID = [
            np.concatenate([self.partition[atm_id] for atm_id in subgroup]).astype(
                np.int64
            )  # NOTE: to be compatible with torch backend
            for subgroup in group
        ]

        self.partition_group_2_gridID = [
            ToTENSOR(x, cpu=True) for x in self.partition_group_2_gridID
        ]

        # Create gridID_2_atmID
        ngrids = self.coords.shape[0]
        self.gridID_2_atmID = np.zeros(ngrids, dtype=np.int32)
        for atm_id, partition in enumerate(self.partition):
            self.gridID_2_atmID[partition] = atm_id
        self.gridID_2_atmID = ToTENSOR(self.gridID_2_atmID)

        # Get grid ordering
        if self.gridID_ordering is None:
            self.gridID_ordering = _get_grid_ordering(self.partition, group)

    def _build_buffer(self, c, m, group):
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        # 1. atm selection #
        naux_atm_sqrt = self.nauxMaxPerAtm_sqrt(c, m)
        naux_atm = naux_atm_sqrt**2
        ngrid_max_atm = self.max_ngrid_involved
        size1 = naux_atm_sqrt * ngrid_max_atm * 2
        size1 += naux_atm * ngrid_max_atm * 2
        # 2. group selection #
        nao = self.nao
        max_naux_group = self.max_group_naux_possible(c, m, group)
        size2 = nao * max_naux_group  # pack aoR
        max_naux_group_sqrt = self.max_group_naux_possible_sqrt(c, m, group)
        size2 += max_naux_group_sqrt * max_naux_group * 2
        size2 += max_naux_group_sqrt**2 * max_naux_group
        # 3. build aux basis #
        max_group_ngrids = self.max_group_ngrids_possbile(group)
        size3 = nao * max_group_ngrids  # pack aoR
        size3 += nao * max_naux_group  # pack aoRg
        size3 += max_naux_group**2  # A
        size3 += max_naux_group * max_group_ngrids  # B
        # 4. build V and W #
        self._build_V_K_bunchsize = min(max_naux_group, self._build_V_K_bunchsize)
        if self.direct:
            size4 = 0
            if self.outcore:
                size4 = self._build_V_K_bunchsize * (
                    np.prod(self.mesh) + self.naux_max(c, m)
                )
        else:
            _size_4_1 = self._build_V_K_bunchsize * np.prod(self.mesh)
            size4 = _size_4_1
        # 5. get J #
        from pyscf.isdf.isdf_local_jk import J_MAX_GRID_BUNCHSIZE

        nao_max_atm = self.max_nao_involved
        size51 = (
            nao_max_atm**2
            + ngrid_max_atm
            + nao_max_atm * min(ngrid_max_atm, J_MAX_GRID_BUNCHSIZE)
        )
        size51 += self.ngrids
        size52 = nao_max_atm**2 + nao_max_atm * min(ngrid_max_atm, J_MAX_GRID_BUNCHSIZE)
        size5 = max(size51, size52)
        # 6. get K #
        size6 = self.nao * max_naux_group + 4 * nao_max_atm * nao  # pack aoRg and dm
        size6 += max_naux_group * nao_max_atm  # used in _isdf_local_K_kernel
        if self.direct:
            size6 += self._build_V_K_bunchsize * self.ngrids
            size6 += self._build_V_K_bunchsize * self.naux_max(c, m)
            if self.outcore:
                size6 += self._build_V_K_bunchsize * self.naux_max(c, m) * 2
        size6 += self._build_V_K_bunchsize * self.nao
        if self.with_robust_fitting:
            size6 += self._build_V_K_bunchsize * self.ngrids
        else:
            size6 += self._build_V_K_bunchsize * self.naux_max(c, m)
        size6 += self._build_V_K_bunchsize * self.nao * 2
        size6 += nao_max_atm * self.nao
        # build buf #
        size = max(size1, size2, size3, size4, size5, size6)
        misc._info(
            self,
            self.rank,
            "In _build_buffer: ISDF Local size of buffer = %.3f MB",
            size * 8 / 1024 / 1024,
        )
        self.buffer_cpu = SimpleMemoryAllocator(total_size=size, gpu=False)
        self.buffer = self.buffer_cpu

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        misc._benchmark_time(t1, t2, "_build_buffer", self, self.rank)

    def _build_fft_buffer(self):
        # if USE_GPU == 0:
        self.buffer_fft = DynamicCached3DRFFT((self._build_V_K_bunchsize, *self.mesh))
        # else:
        #     self.buffer_fft = DynamicCached3DRFFT_GPU(
        #         (self._build_V_K_bunchsize, *self.mesh)
        #     )

    def _build_IP(self, c, m, rela_cutoff, group, global_IP_selection):
        self.IP_group = None

        if self.IP_group is None:
            t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
            IP_Atm = select_IP_local_step1(
                self, c if len(group) < self.first_natm else c + 1, 
                m, rela_cutoff=rela_cutoff,
                no_retriction_on_nIP=abs(rela_cutoff) > 1e-6,
                use_mpi=self.use_mpi,
            )
            t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

            misc._benchmark_time(t1, t2, "select_IP_local_step1", self, self.rank)

            aoRg_possible = self._build_aoRg(IP_Atm, None)

            t3 = (lib.logger.process_clock(), lib.logger.perf_counter())

            misc._benchmark_time(t2, t3, "build_aoRg_possible", self, self.rank)

            # step2 selection for each group #

            select_IP_local_step2(
                self,
                c=c,
                m=m,
                rela_cutoff=rela_cutoff,
                group=group,
                IP_possible_atm=IP_Atm,
                aoRg_possible=aoRg_possible,
                global_IP_selection=global_IP_selection,
                use_mpi=self.use_mpi,
            )

            t4 = (lib.logger.process_clock(), lib.logger.perf_counter())

            misc._benchmark_time(t3, t4, "select_IP_local_step2", self, self.rank)

            # step3 build indices info related to IP_group #
            select_IP_local_step3(self, group, self.use_mpi)

            if len(group) < self.first_natm:
                del aoRg_possible
                self.aoRg = self._build_aoRg(self.partition_IP, group)
            else:
                self.aoRg = aoRg_possible

            t5 = (lib.logger.process_clock(), lib.logger.perf_counter())

            misc._benchmark_time(t4, t5, "select_IP_local_step3", self, self.rank)

        # print memory #

        memory_aoR = _get_aoR_holders_memory(self.aoR)
        memory_aoRg = _get_aoR_holders_memory(self.aoRg)
        misc._info(
            self,
            self.rank,
            "Memory usage for aoR  : %10.3f MB",
            memory_aoR / 1024 / 1024,
        )
        misc._info(
            self,
            self.rank,
            "Memory usage for aoRg : %10.3f MB",
            memory_aoRg / 1024 / 1024,
        )
        misc._benchmark_time(t1, t5, "build_IP", self, self.rank)
        return self.IP_group

    def _build_aoRg(self, IP_group, group=None):
        # assert group is not None
        if group is None:
            group = [[i] for i in range(self.first_natm)]
        assert len(IP_group) == self.first_natm

        t0 = (lib.logger.process_clock(), lib.logger.perf_counter())

        aoRg_holders = [None] * self.first_natm
        atm_ordering = [atm for subgroup in group for atm in subgroup]
        IP_ID_now = 0

        for atm_id in atm_ordering:
            aoR_holder = self.aoR[atm_id]
            _IP_group = IP_group[atm_id]
            if aoR_holder is None or _IP_group is None:
                IP_ID_now += len(_IP_group) if _IP_group is not None else 0
                continue
            nIP = len(_IP_group)

            # NOTE: debugging code #
            # assert all(
            #     [
            #         IP_ID in self.partition[atm_id]
            #         for IP_ID in ToNUMPY(_IP_group)
            #     ]
            # )

            idx = np.searchsorted(ToNUMPY(self.partition[atm_id]), ToNUMPY(_IP_group))
            aoRg_holders[atm_id] = aoR_Holder(
                TAKE(aoR_holder.aoR, ToTENSOR(idx, cpu=True), axis=1),
                ToTENSOR(ToNUMPY(aoR_holder.ao_involved).copy()),
                IP_ID_now,
                IP_ID_now + nIP,
            )
            IP_ID_now += nIP
        # self.aoRg = aoRg_holders
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        misc._benchmark_time(t0, t1, "_build_aoRg", self, self.rank)

        return aoRg_holders

    def _build_aux_basis(self, group):
        self.aux_basis = build_aux_basis_local(self, group, self.IP_group, self.use_mpi)

    def _build_V_W(self):
        self.coul_G = tools.get_coulG(self.cell, mesh=self.mesh)
        if self.direct:
            return
        
        if self.outcore:
            self.V, self.W = build_V_W_local_outcore(self, self.use_mpi)
        else:
            self.V, self.W = build_V_W_local(self, self.use_mpi)

    def rebuild(self, direct=True, with_robust_fitting=True):
        self.direct = direct
        self.with_robust_fitting = with_robust_fitting
        self._rebuild_buffer()
        self._rebuild_VW()

    def _rebuild_buffer(self):
        del self.buffer_cpu
        del self.buffer_gpu
        self._build_buffer(self.c, self.m, self.group)

    def _rebuild_VW(self):
        del self.V
        del self.W
        if self.direct:
            return
        else:
            self._build_V_W()

    def force_translation_symmetry(self, T_mesh=None):
        assert T_mesh is not None
        self._force_translation_sym = True
        self._T_mesh = T_mesh

    ### properties ###

    @property
    def first_natm(self):
        if self.kmesh is not None:
            return self.cell.natm // np.prod(self.kmesh)
        else:
            return self.cell.natm

    @property
    def max_nao_involved(self):
        return np.max(
            [
                aoR_holder.aoR.shape[0]
                for aoR_holder in self.aoR
                if aoR_holder is not None
            ]
        )

    @property
    def max_ngrid_involved(self):
        return np.max(
            [
                aoR_holder.aoR.shape[1]
                for aoR_holder in self.aoR
                if aoR_holder is not None
            ]
        )

    @property
    def max_nIP_involved(self):
        return np.max(
            [
                aoR_holder.aoR.shape[1]
                for aoR_holder in self.aoRg
                if aoR_holder is not None
            ]
        )

    @property
    def maxsize_group_naux(self):
        maxsize_group_naux = 0
        for group_id, atm_ids in enumerate(self.group):
            naux_tmp = 0
            for atm_id in atm_ids:
                naux_tmp += self.aoRg[atm_id].aoR.shape[1]
            maxsize_group_naux = max(maxsize_group_naux, naux_tmp)
        return maxsize_group_naux

    # used to infer memory usage #

    def max_group_ngrids_possbile(self, group):
        max_group_ngrids_possbile = 0
        for group_id, atm_ids in enumerate(group):
            ngrids_tmp = 0
            for atm_id in atm_ids:
                ngrids_tmp += self.aoR[atm_id].aoR.shape[1]
            max_group_ngrids_possbile = max(max_group_ngrids_possbile, ngrids_tmp)
        return max_group_ngrids_possbile

    def max_group_naux_possible(self, c=None, m=None, group=None):
        assert group is not None
        if c is None:
            c = self.c
        if m is None:
            m = self.m
        max_group_naux_possible = 0
        for group_id, atm_ids in enumerate(group):
            naux_tmp = 0
            for atm_id in atm_ids:
                naux_tmp += int(np.sqrt(self.atmID2nao[atm_id] * (c + 1)) + m) ** 2
            max_group_naux_possible = max(max_group_naux_possible, naux_tmp)
        return max_group_naux_possible

    def max_group_naux_possible_sqrt(self, c, m, group):
        max_group_naux_possible_sqrt = 0
        for group_id, atm_ids in enumerate(group):
            nao_group = sum(self.atmID2nao[atm_ids])
            naux_tmp = int(np.sqrt(c * nao_group)) + m
            max_group_naux_possible_sqrt = max(max_group_naux_possible_sqrt, naux_tmp)
        return max_group_naux_possible_sqrt

    # #

    def _determine_c(self, rela_cutoff):
        """
        called in _build_IP when c is not set
        empirical rule to determine c
        """

        DEFAULT = 15
        SEGMENT = [1e-2, 1e-3, 1e-4, 1e-5]
        C = [10, 20, 30, 35, 40]

        if rela_cutoff is None:
            return DEFAULT
        else:
            if rela_cutoff > SEGMENT[0]:
                return C[0]
            else:
                for i in range(1, len(SEGMENT)):
                    if rela_cutoff > SEGMENT[i]:
                        return C[i]
                return C[-1]

    def get_pp(self, kpts=None):
        res = super().get_pp(kpts)
        if self._force_translation_sym:
            from pyscf.isdf.isdf_tools_Tsym import symmetrize_mat

            res = ToNUMPY(symmetrize_mat(ToTENSOR(res), self._T_mesh))
        return res

    def aoRg_full(self):
        group_begin, group_end = 0, len(self.group)
        aoRg_to_pack = []
        for i in range(group_begin, group_end):
            aoRg_to_pack.extend([self.aoRg[atm_id] for atm_id in self.group[i]])
        return _pack_aoR_holder(aoRg_to_pack, self.nao)

    ########## other funcs ##########

    get_jk = isdf_local_jk.get_jk_dm_local

    from pyscf.isdf.isdf_local_ao2mo import general, get_eri

    get_eri = get_ao_eri = get_eri
    ao2mo = get_mo_eri = general
