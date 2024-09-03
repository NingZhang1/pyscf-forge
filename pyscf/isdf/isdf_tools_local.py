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

########## pyscf module ##########

import copy
import numpy as np
import numpy
from functools import partial
import pyscf
import sys
from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
import pyscf.pbc.df.ft_ao as ft_ao
from pyscf.pbc.df import aft, rsdf_builder, aft_jk

########## isdf  backend ##########

import pyscf.isdf.BackEnd.isdf_backend as BACKEND

NUM_THREADS = BACKEND.NUM_THREADS
FLOAT64 = BACKEND.FLOAT64
INT32 = BACKEND.INT32
ToNUMPY = BACKEND._toNumpy
ToTENSOR = BACKEND._toTensor
MALLOC = BACKEND._malloc
ZEROS = BACKEND._zeros
MAX = BACKEND._maximum
MIN = BACKEND._minimum
ABS = BACKEND._absolute
DOT = BACKEND._dot
TAKE = BACKEND._take

########## isdf  module  ##########

from pyscf.isdf.misc import _benchmark_time
from pyscf.isdf.isdf_eval_gto import ISDF_eval_gto

########## global parameter ##########

DISTANCE_CUTOFF = 25  # suitable for cuprates !


############ build atm connection graph ############


class AtmConnectionInfo:
    def __init__(
        self, cell: Cell, atmID, distance_matrix, precision, rcut, rcut_max, atm_to_bas
    ):
        """
        rcut: the cutoff radius of each bas
        """

        self.precision = precision
        self.atmID = atmID
        self.atmID_connection = ToTENSOR(
            np.where(ToNUMPY(distance_matrix)[atmID] < rcut_max)[0]
        )
        self.distance = ToTENSOR(distance_matrix[atmID][self.atmID_connection])
        self.atm_connected_info = list(
            zip(ToNUMPY(self.atmID_connection), ToNUMPY(self.distance))
        )
        # sort by distance
        self.atm_connected_info.sort(key=lambda x: x[1])
        self.bas_range = ToTENSOR(np.arange(atm_to_bas[atmID][0], atm_to_bas[atmID][1]))
        self.bas_cut = ToTENSOR(rcut[atm_to_bas[atmID][0] : atm_to_bas[atmID][1]])

    def __repr__(self):
        return "atmID = %d, atm_connected_info = %s, bas_range = %s, bas_cut = %s" % (
            self.atmID,
            self.atm_connected_info,
            self.bas_range,
            self.bas_cut,
        )


class aoR_Holder:
    def __init__(
        self,
        aoR,
        ao_involved,
        global_gridID_begin,
        global_gridID_end,
    ):
        """
        currently local_gridID_begin, local_gridID_end is not useful
        """

        assert aoR.shape[0] == len(ao_involved)
        assert aoR.shape[1] <= (global_gridID_end - global_gridID_begin)
        self.ngrid_tot = global_gridID_end - global_gridID_begin
        self.ngrid_kept = aoR.shape[1]

        aoR = ToNUMPY(aoR)
        ao_involved = np.array(ao_involved, dtype=np.int64)
        sort_indices = np.argsort(ao_involved)
        ao_involved = ao_involved[sort_indices].copy()
        aoR = aoR[sort_indices].copy()

        self.aoR = ToTENSOR(aoR)
        self.ao_involved = ToTENSOR(ao_involved)  # NOTE: to be consistent with torch
        self.nao_involved = len(ao_involved)
        self.global_gridID_begin = global_gridID_begin
        self.global_gridID_end = global_gridID_end

    def size(self):
        return self.aoR.nbytes + self.ao_involved.nbytes

    def todense(self, nao):
        # aoR = np.zeros((nao, self.aoR.shape[1]))
        # aoR[self.ao_involved] = self.aoR
        # return aoR
        aoR = ZEROS((nao, self.ngrid_tot), dtype=FLOAT64)
        BACKEND._index_copy(aoR, dim=0, index=self.ao_involved, source=self.aoR)
        return aoR

    def toFull(self, nao):
        aoR = ZEROS((nao, self.ngrid_tot), dtype=FLOAT64)
        BACKEND._index_copy(aoR, dim=0, index=self.ao_involved, source=self.aoR)
        ao_involved = ToTENSOR(np.array(list(range(nao)), dtype=np.int64))
        return aoR_Holder(
            aoR, ao_involved, self.global_gridID_begin, self.global_gridID_end
        )


# utils for aoR Holder #


def _get_aoR_holders_memory(aoR_holders: list[aoR_Holder]):
    return sum(
        [aoR_holder.size() for aoR_holder in aoR_holders if aoR_Holder is not None]
    )


def _pack_aoR_holder(aoR_holders: list[aoR_Holder], nao, out_buf=None):
    # NOTE: everything is should be put on CPU #
    # NOTE: the grid ordering is determined by the input of aoR_holders! #
    # determine basic info #
    has_involved = np.asarray([False] * nao)
    for aoR_holder in aoR_holders:
        if aoR_holder is not None:
            has_involved[ToNUMPY(aoR_holder.ao_involved)] = True
    ngrids = sum(
        _aoR_holder.ngrid_tot for _aoR_holder in aoR_holders if _aoR_holder is not None
    )
    ao2loc = ZEROS(nao, dtype=INT32)
    loc_now = 0
    for ao_id, involved in enumerate(has_involved):
        if involved:
            ao2loc[ao_id] = loc_now
            loc_now += 1
        else:
            ao2loc[ao_id] = -1
    nao_involved = loc_now

    # pack aoR #

    if out_buf is None:
        aoR_packed = ZEROS((nao_involved, ngrids), dtype=FLOAT64)
    else:
        aoR_packed = MALLOC((nao_involved, ngrids), dtype=FLOAT64, buf=out_buf)

    grid_begin_id = 0
    for _aoR_holder in aoR_holders:
        if _aoR_holder is None:
            continue
        grid_end_id = grid_begin_id + _aoR_holder.ngrid_tot
        loc_packed = TAKE(ao2loc, _aoR_holder.ao_involved, axis=0)
        aoR_packed[loc_packed, grid_begin_id:grid_end_id] = _aoR_holder.aoR
        grid_begin_id = grid_end_id
    aoR_packed_involved = np.array(
        [i for i in range(nao) if has_involved[i]], dtype=np.int32
    )
    aoR_packed_involved = ToTENSOR(aoR_packed_involved)

    assert ngrids == grid_begin_id
    global_gridID_begin = 0
    global_gridID_end = ngrids

    return aoR_Holder(
        aoR_packed,
        aoR_packed_involved,
        global_gridID_begin,
        global_gridID_end,
    )


# get the rcut #


def _atm_to_bas(cell: Cell):
    shl_atm = []

    natm = cell.natm

    for i in range(natm):
        shl_atm.append([None, None])

    for i in range(cell.nbas):
        atm_id = cell.bas_atom(i)
        if shl_atm[atm_id][0] is None:
            shl_atm[atm_id][0] = i
        shl_atm[atm_id][1] = i + 1

    return shl_atm


def _estimate_rcut(cell, ngrids, precision):
    """
    Cutoff raidus, above which each shell decays to a value less than the
    required precsion
    """

    weight = numpy.sqrt(cell.vol / ngrids)  # note the weight !
    log_prec = numpy.log(precision / weight)
    rcut = []
    for ib in range(cell.nbas):
        l = cell.bas_angular(ib)
        es = cell.bas_exp(ib)
        cs = abs(cell.bas_ctr_coeff(ib)).max(axis=1)
        r = 5.0
        r = (((l + 2) * numpy.log(r) + numpy.log(cs) - log_prec) / es) ** 0.5
        r[r < 1.0] = 1.0
        r = (((l + 2) * numpy.log(r) + numpy.log(cs) - log_prec) / es) ** 0.5
        rcut.append(r.max())
    return ToTENSOR(numpy.array(rcut))


# the distance graph #


def _distance_translation(pa: np.ndarray, pb: np.ndarray, a: np.ndarray):
    """
    Calculate the distance between pa and pb, taking periodic boundary conditions into account.

    :param pa: Coordinates of point a
    :param pb: Coordinates of point b
    :param a: Lattice vectors (assumed to be a 3x3 array)
    :return: Minimum distance considering periodic boundary conditions
    """

    dx = pa[0] - pb[0]
    dx = MIN(ABS(ToTENSOR([dx, dx - a[0][0], dx + a[0][0]])))
    dy = pa[1] - pb[1]
    dy = MIN(ABS(ToTENSOR([dy, dy - a[1][1], dy + a[1][1]])))
    dz = pa[2] - pb[2]
    dz = MIN(ABS(ToTENSOR([dz, dz - a[2][2], dz + a[2][2]])))

    return np.sqrt(dx**2 + dy**2 + dz**2)


def get_cell_distance_matrix(cell: Cell):
    """
    get the distance matrix of the cell
    """
    a = ToTENSOR(cell.lattice_vectors())
    n = cell.natm
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            distance_matrix[i][j] = _distance_translation(
                ToTENSOR(cell.atom_coord(i)), ToTENSOR(cell.atom_coord(j)), a
            )
            distance_matrix[j][i] = distance_matrix[i][j]
    return ToTENSOR(distance_matrix)


# build cutoff info #


def build_cutoff_info(cell: Cell, precision, ngrids):
    distance_matrix = get_cell_distance_matrix(cell)
    rcut = _estimate_rcut(cell, ngrids, precision)
    rcut_max = np.max(ToNUMPY(rcut))
    atm2bas = _atm_to_bas(cell)
    AtmConnectInfo = []
    for atm_id in range(cell.natm):
        AtmConnectInfo.append(
            AtmConnectionInfo(
                cell,
                atm_id,
                distance_matrix,
                precision,
                rcut,
                rcut_max,
                atm2bas,
            )
        )
    return distance_matrix, AtmConnectInfo


############ routines to split tasks ############


def _range_partition(ngroup, rank, comm_size, use_mpi=False):
    """
    Given ngroup tasks, split them into comm_size parts, and return the range of tasks for the rank-th process.
    """
    if not use_mpi:
        return 0, ngroup

    # Calculate the base size and the number of larger partitions
    base_size, larger_count = divmod(ngroup, comm_size)

    if rank < larger_count:
        start = rank * (base_size + 1)
        end = start + base_size + 1
    else:
        start = larger_count * (base_size + 1) + (rank - larger_count) * base_size
        end = start + base_size

    return start, end


def _range_partition_array(ngroup, comm_size, use_mpi=False):
    if not use_mpi:
        return ToTENSOR(np.array([[0, ngroup]], dtype=np.int32))

    # Calculate the base size and the number of larger partitions
    base_size, larger_count = divmod(ngroup, comm_size)
    sizes = np.full(comm_size, base_size)
    sizes[:larger_count] += 1

    # Calculate cumulative sums to get start and end points
    endpoints = np.cumsum(sizes)
    startpoints = np.roll(endpoints, 1)
    startpoints[0] = 0

    # Combine start and end points
    res = np.column_stack((startpoints, endpoints)).astype(np.int32)

    return ToTENSOR(res)


def _get_grid_ordering(atmid_to_gridID, group):
    """given the grid points associated to each atom, return the reordering of grid points according to the ID of atms."""

    grid_ordering = []
    for i in range(len(group)):
        for atmid in group[i]:
            grid_ordering.extend(atmid_to_gridID[atmid])

    return ToTENSOR(np.array(grid_ordering, dtype=np.int32))


def _get_grid_partition(atmid_to_gridID, group, use_mpi=False):
    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import comm_size

    ngrid = np.sum([len(x) for x in atmid_to_gridID])

    if not use_mpi:
        return ToTENSOR(np.array([0, ngrid], dtype=np.int32))
    else:
        group_partition_array = _range_partition_array(len(group), comm_size, use_mpi)

        grid_partition = [0]
        for i in range(comm_size):
            group_begin = group_partition_array[i][0]
            group_end = group_partition_array[i][1]

            ngrid_local = 0
            for j in range(group_begin, group_end):
                for atmid in group[j]:
                    ngrid_local += len(atmid_to_gridID[atmid])

            grid_partition.append(grid_partition[-1] + ngrid_local)

        return ToTENSOR(np.array(grid_partition, dtype=np.int32))


def _get_atm_2_grid_segment(atmid_to_gridID, group):
    natm = len(atmid_to_gridID)
    assert (
        sum([len(x) for x in group]) == natm
        or (natm % sum([len(x) for x in group])) == 0
    )

    res = []
    for _ in range(natm):
        res.append([None, None])

    grid_loc_now = 0
    for j in range(len(group)):
        for atmid in group[j]:
            res[atmid][0] = grid_loc_now
            res[atmid][1] = grid_loc_now + len(atmid_to_gridID[atmid])
            grid_loc_now += len(atmid_to_gridID[atmid])

    return res


############ algorithm based on the distance graph and AtmConnectionInfo ############

# utils to build aoR #


def _sync_list(list_data, ngroup):
    from pyscf.isdf.isdf_tools_mpi import rank, comm_size, bcast

    if len(list_data) != ngroup:
        raise ValueError("The length of list_data is not equal to ngroup")

    # Partial function for range partition
    range_partition = partial(
        _range_partition, ngroup=ngroup, comm_size=comm_size, use_mpi=True
    )

    # Check data integrity
    group_begin, group_end = range_partition(rank=rank)
    assert all(x is None for x in list_data[:group_begin])
    assert all(x is not None for x in list_data[group_begin:group_end])
    assert all(x is None for x in list_data[group_end:])

    # Generate groupid_2_root
    range_partition_array = [range_partition(rank=i) for i in range(comm_size)]
    groupid_2_root = [
        i
        for i, (start, end) in enumerate(range_partition_array)
        for _ in range(start, end)
    ]

    # Sync data
    for i in range(ngroup):
        list_data[i] = bcast(list_data[i], root=groupid_2_root[i])

    return list_data


def _sync_aoR(aoR_holders, natm):
    """Synchronize aoR_holders in MPI environment"""
    assert len(aoR_holders) == natm

    # Extract data from aoR_holders

    aoR, bas_id, grid_ID_begin = zip(
        *[
            (
                (ToNUMPY(holder.aoR) if holder is not None else None),
                (ToNUMPY(holder.ao_involved) if holder is not None else None),
                (
                    np.array([holder.global_gridID_begin], dtype=np.int32)
                    if holder is not None
                    else None
                ),
            )
            for holder in aoR_holders
        ]
    )

    # Synchronize data
    aoR = _sync_list(list(aoR), natm)
    bas_id = _sync_list(list(bas_id), natm)
    grid_ID_begin = _sync_list(list(grid_ID_begin), natm)

    # Reconstruct aoR_holders
    return [
        aoR_Holder(
            aoR[i],
            bas_id[i],
            grid_ID_begin[i][0],
            grid_ID_begin[i][0] + aoR[i].shape[1],
        )
        for i in range(natm)
    ]


def _build_submol(cell: Cell, atm_invovled):
    # import pyscf.pbc.gto as pbcgto

    subcell = pbcgto.Cell()
    subcell.a = cell.a

    atm = []
    for atm_id in atm_invovled:
        atm.append(cell.atom[atm_id])

    subcell.atom = atm
    subcell.basis = cell.basis
    subcell.pseudo = cell.pseudo
    subcell.verbose = 0
    subcell.ke_cutoff = cell.ke_cutoff
    subcell.max_memory = cell.max_memory
    subcell.precision = cell.precision
    subcell.use_particle_mesh_ewald = cell.use_particle_mesh_ewald
    subcell.mesh = cell.mesh
    subcell.unit = cell.unit
    subcell.build(mesh=cell.mesh)

    return subcell


# get partition #


def get_partition(
    cell,
    coords,
    AtmConnectionInfoList,
    Ls=[3, 3, 3],
    with_translation_symmetry=False,
    kmesh=None,
    use_mpi=False,
):
    #### funcs ####

    from pyscf.isdf.BackEnd._isdf_numpy import distance_translation as DISTANCE_NUMPY

    DISTANCE = BACKEND._distance_translation

    #### preprocess ####

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, bcast

    if with_translation_symmetry and kmesh is None:
        raise ValueError("kmesh must be provided if with_translation_symmetry is True")

    log = lib.logger.Logger(cell.stdout, cell.verbose)
    if not use_mpi or (use_mpi and rank == 0):
        log.debug4("************* get_partition *************")

    mesh = cell.mesh
    lattice_vector = np.array(cell.lattice_vectors())

    meshPrim = None
    if with_translation_symmetry:
        meshPrim = np.array(mesh) // np.array(kmesh)

    #### partition the cubic into small cubic ####

    def get_box_info(mesh, Ls):
        mesh_box = np.array([(m // L + (1 if m % L else 0)) for m, L in zip(mesh, Ls)])
        nbox = np.array(
            [m // mb + (1 if m % mb else 0) for m, mb in zip(mesh, mesh_box)]
        )
        Ls_box = [lv / m * mb for lv, m, mb in zip(lattice_vector, mesh, mesh_box)]
        return mesh_box, nbox, Ls_box

    mesh_box, nbox, Ls_box = get_box_info(mesh, Ls)

    assert Ls_box[0][0] < 3.0
    assert Ls_box[1][1] < 3.0
    assert Ls_box[2][2] < 3.0

    ##### helper functions #####

    def get_box_id(coord):
        return tuple(int(c / Ls_box[i][i]) for i, c in enumerate(coord))

    def get_box_id_from_coord(coord):
        return get_box_id(coord)

    def get_mesh_id(ix, iy, iz):
        return ix * mesh[1] * mesh[2] + iy * mesh[2] + iz

    ### build infor between atm and box id ###

    atm_box_id = []
    box_2_atm = {}

    atm_coords = []

    for i in range(cell.natm):
        box_id = get_box_id_from_coord(cell.atom_coord(i))
        atm_box_id.append(box_id)
        if box_id not in box_2_atm:
            box_2_atm[box_id] = [i]
        else:
            box_2_atm[box_id].append(i)
        atm_coords.append(cell.atom_coord(i))

    atm_coords = np.array(atm_coords)

    # do the work #

    natm_tmp = cell.natm // np.prod(kmesh) if with_translation_symmetry else cell.natm
    partition = [[] for _ in range(natm_tmp)]

    if not use_mpi or (use_mpi and rank == 0):
        # get a rough partition #

        partition_rough = [[] for _ in range(natm_tmp)]
        grid_id_global = np.arange(np.prod(mesh), dtype=np.int32).reshape(mesh)

        for ix, iy, iz in np.ndindex(tuple(nbox)):
            box_id = (ix, iy, iz)
            slice_x = slice(ix * mesh_box[0], min((ix + 1) * mesh_box[0], mesh[0]))
            slice_y = slice(iy * mesh_box[1], min((iy + 1) * mesh_box[1], mesh[1]))
            slice_z = slice(iz * mesh_box[2], min((iz + 1) * mesh_box[2], mesh[2]))

            if with_translation_symmetry:
                if any(
                    s.start >= mp
                    for s, mp in zip([slice_x, slice_y, slice_z], meshPrim)
                ):
                    continue
                slice_x = slice(slice_x.start, min(slice_x.stop, meshPrim[0]))
                slice_y = slice(slice_y.start, min(slice_y.stop, meshPrim[1]))
                slice_z = slice(slice_z.start, min(slice_z.stop, meshPrim[2]))

            grid_ID = grid_id_global[slice_x, slice_y, slice_z].flatten()
            grid_ID.sort()
            grid_ID = np.array(grid_ID, dtype=np.int32)

            # atm_id = min(get_box_id(cell.atom_coord(i)) for i in range(cell.natm))
            # partition[atm_id % natm_tmp].extend(grid_ID)

            if box_id in box_2_atm:
                partition_rough[box_2_atm[box_id][0] % natm_tmp].extend(grid_ID)
                # print("box_id", box_id, "atm_id", box_2_atm[box_id][0] % natm_tmp)
            else:
                # random pickup one coord in the box #
                grid_ID_random_pick = grid_ID[np.random.randint(0, len(grid_ID))]
                grid_coord = coords[grid_ID_random_pick]
                grid_coord = np.array(grid_coord)
                distance = DISTANCE_NUMPY(
                    grid_coord.reshape(1, 3), atm_coords, lattice_vector
                ).reshape(-1)
                atm_id = np.argmin(distance)
                # print(atm_id)
                partition_rough[atm_id % natm_tmp].extend(grid_ID)
                # print("box_id", box_id, "atm_id", atm_id % natm_tmp)

        # check info

        ngrid_involved = 0
        for atm_id, x in enumerate(partition_rough):
            ngrid_involved += len(x)
        if with_translation_symmetry:
            assert ngrid_involved == np.prod(meshPrim)
        else:
            assert ngrid_involved == np.prod(mesh)

        # Refine partition based on AtmConnectionInfo

        from copy import deepcopy

        lattice_vector = deepcopy(cell.lattice_vectors())
        if with_translation_symmetry:
            lattice_vector = np.array(lattice_vector) / np.array(kmesh)
        lattice_vector = ToTENSOR(lattice_vector)

        for atm_id in range(natm_tmp):
            atm_involved = [
                i % natm_tmp
                for i, d in AtmConnectionInfoList[atm_id].atm_connected_info
                if d < DISTANCE_CUTOFF
            ][:25]
            atm_involved = sorted(set(atm_involved))
            atm_involved = np.array(atm_involved, dtype=np.int32)

            coords_now = ToTENSOR(ToNUMPY(coords[partition_rough[atm_id]]).copy())
            if coords_now.shape[0] == 0:
                continue
            atm_coords_involved = ToTENSOR(
                np.array([cell.atom_coord(i) for i in atm_involved])
            )
            # print(coords_now.shape)
            # print(atm_coords_involved.shape)
            distance = ToNUMPY(
                DISTANCE(coords_now, atm_coords_involved, lattice_vector)
            )
            argmin_distance = np.argmin(distance, axis=1)

            for grid_id, _atm_id_ in zip(partition_rough[atm_id], argmin_distance):
                partition[atm_involved[_atm_id_] % natm_tmp].append(grid_id)

    # sort #

    for i in range(natm_tmp):
        partition[i].sort()

    if use_mpi:
        partition = [bcast(np.array(x, dtype=np.int32)) for x in partition]

    if not use_mpi or (use_mpi and rank == 0):
        log.debug4("************* end get_partition *************")
    return partition


# get aoR #


def get_aoR(
    cell,
    AtmConnectionInfoList,
    coords,
    partition,
    first_npartition=None,
    first_natm=None,
    group=None,
    use_mpi=False,
    sync_res=False,
):
    # prerpocess #

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm_size, bcast
    else:
        rank, comm_size = 0, 1

    log = lib.logger.Logger(cell.stdout, cell.verbose)
    if not use_mpi or (use_mpi and rank == 0):
        log.debug4("************* get_aoR *************")

    if first_natm is None:
        first_natm = cell.natm
    if first_npartition is None:
        first_npartition = len(partition)
    if group is None:
        group = [[i] for i in range(cell.natm)]

    weight = np.sqrt(cell.vol / coords.shape[0])
    RcutMax = max(np.max(ToNUMPY(info.bas_cut)) for info in AtmConnectionInfoList)
    precision = AtmConnectionInfoList[0].precision
    ao_loc = cell.ao_loc_nr()

    # grid_partition = _get_grid_partition(partition, group, use_mpi)
    atm_2_grid_segment = _get_atm_2_grid_segment(partition, group)

    aoR_holder = [None] * first_npartition
    atm_begin, atm_end = _range_partition(first_npartition, rank, comm_size, use_mpi)

    for atm_id in range(atm_begin, atm_end):
        grid_ID = partition[atm_id]
        if len(grid_ID) == 0:
            # aoR_holder[atm_id] = aoR_Holder(
            #     ZEROS((0, 0), dtype=FLOAT64),
            #     ToTENSOR(np.array([], dtype=np.int64)),
            #     global_gridID_begin=global_gridID_begin,
            #     global_gridID_end=global_gridID_begin
            # )
            aoR_holder[atm_id] = None
            continue

        ##### find the involved atms within RcutMax #####

        atm_involved = [
            atm
            for atm, dist in AtmConnectionInfoList[atm_id].atm_connected_info
            if dist < RcutMax and atm < first_natm
        ]

        ##### get the involved ao #####

        bas_id = []
        for atm in atm_involved:
            shl_begin, shl_end = (
                AtmConnectionInfoList[atm].bas_range[0],
                AtmConnectionInfoList[atm].bas_range[-1] + 1,
            )
            bas_id.extend(range(ao_loc[shl_begin], ao_loc[shl_end]))

        if len(bas_id) == 0:
            aoR_holder[atm_id] = None
            continue

        subcell = _build_submol(cell, atm_involved)
        aoR = ISDF_eval_gto(subcell, coords=coords[grid_ID]) * weight

        assert aoR.shape[0] == len(bas_id)

        # Screening the aoR #

        max_row = ToNUMPY(MAX(ABS(aoR), axis=1))
        where = np.where(max_row > precision)[0]
        if len(where) < aoR.shape[0] * 0.9:
            where = ToTENSOR(where)
            aoR = TAKE(aoR, where, axis=0)
            bas_id = ToTENSOR(np.array(bas_id)[where])

        global_gridID_begin = atm_2_grid_segment[atm_id][0]
        aoR_holder[atm_id] = aoR_Holder(
            aoR,
            bas_id,
            global_gridID_begin=global_gridID_begin,
            global_gridID_end=global_gridID_begin + len(grid_ID),
        )

    if use_mpi and sync_res:
        aoR_holder = _sync_aoR(aoR_holder, first_npartition)

    if not use_mpi or (use_mpi and rank == 0):
        log.debug4("************* end get_aoR *************")
    return aoR_holder
