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

import copy
import numpy as np
import scipy, numpy
import ctypes, sys
import itertools
from functools import lru_cache

############ pyscf module ############

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools

############ isdf backends ############

import pyscf.isdf.BackEnd.isdf_backend as BACKEND
from pyscf.isdf.BackEnd.isdf_memory_allocator import SimpleMemoryAllocator
from pyscf.isdf.BackEnd.isdf_fft_cache import DynamicCached3DRFFT

USE_GPU = BACKEND.USE_GPU
NUM_THREADS = BACKEND.NUM_THREADS

if USE_GPU == 1:
    from pyscf.isdf.BackEnd.isdf_fft_cache import DynamicCached3DRFFT_GPU

assert (USE_GPU == 0) or (not USE_GPU), "ISDF_Local does not support GPU backend!"

FLOAT64 = BACKEND.FLOAT64
ToNUMPY = BACKEND._toNumpy
ToTENSOR = BACKEND._toTensor
ZEROS = BACKEND._zeros
MAX = BACKEND._maximum
ABS = BACKEND._absolute
DOT = BACKEND._dot
TAKE = BACKEND._take

############ isdf utils ############

from pyscf.isdf.isdf_local import ISDF_Local
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
from pyscf.isdf.isdf_local_k_jk import get_jk_dm_k_local

############ subroutines --- select IP ############


def _canonicalize_partition_prim(mydf, partition_prim):

    DISTANCE = BACKEND._distance_translation
    mesh, kmesh = mydf.mesh, mydf.kmesh
    meshPrim = np.array(mesh) // np.array(kmesh)

    shifts = np.array(
        [
            i * meshPrim[0] * mesh[1] * mesh[2]
            + j * meshPrim[1] * mesh[2]
            + k * meshPrim[2]
            for i, j, k in itertools.product(
                range(kmesh[0]), range(kmesh[1]), range(kmesh[2])
            )
        ]
    )

    cell = mydf.cell

    assert len(partition_prim) == mydf.natmPrim

    lattice_vector = copy.deepcopy(cell.lattice_vectors())
    lattice_vector = ToTENSOR(lattice_vector)

    for i in range(mydf.first_natm):

        partition_i = partition_prim[i]
        atm_coord = ToTENSOR(np.array([cell.atom_coord(i)]))

        for j in range(len(partition_i)):
            imags = [(partition_i[j] + shift) % (np.prod(mesh)) for shift in shifts]
            coord_imag = ToTENSOR(mydf.coords[imags])
            distance = ToNUMPY(
                DISTANCE(coord_imag, atm_coord, lattice_vector).reshape(-1)
            )
            argmin_distance = np.argmin(distance, axis=0)
            if argmin_distance != 0:
                partition_i[j] = imags[argmin_distance]

        partition_i = ToNUMPY(partition_i)
        partition_i = np.sort(partition_i)

        partition_prim[i] = ToTENSOR(partition_i)

    return partition_prim


def _expand_partition_prim(partition_prim, kmesh, mesh):
    meshPrim = np.array(mesh) // np.array(kmesh)
    shifts = np.array(
        [
            i * meshPrim[0] * mesh[1] * mesh[2]
            + j * meshPrim[1] * mesh[2]
            + k * meshPrim[2]
            for i, j, k in itertools.product(
                range(kmesh[0]), range(kmesh[1]), range(kmesh[2])
            )
        ]
    )

    res = [data + shift for shift in shifts for data in partition_prim]

    for i in range(len(res)):
        res[i] = [x % np.prod(mesh) for x in res[i]]
        res[i] = np.sort(res[i])
        res[i] = ToTENSOR(res[i])

    return res


def _expand_primlist_2_superlist(primlist, kmesh, mesh):
    meshPrim = np.array(mesh) // np.array(kmesh)

    shifts = (
        i * meshPrim[0] * mesh[1] * mesh[2]
        + j * meshPrim[1] * mesh[2]
        + k * meshPrim[2]
        for i, j, k in itertools.product(
            range(kmesh[0]), range(kmesh[1]), range(kmesh[2])
        )
    )

    return ToTENSOR(
        np.array([ToNUMPY(primlist) + shift for shift in shifts], dtype=np.int64)
    )


def _get_grid_ordering_k(_input, kmesh, mesh):
    if not isinstance(_input, list):
        raise TypeError("Input must be a list")
    prim_ordering = []
    for data in _input:
        prim_ordering.extend(data)
    return _expand_primlist_2_superlist(prim_ordering, kmesh, mesh)


############ subroutines --- build V W ############


def build_V_W_local_k(mydf, use_mpi=False):

    assert not mydf.direct

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size
    else:
        rank = 0
        # comm_size = 1

    misc._debug4(
        mydf, rank, " ---------- In pyscf.isdf.isdf_local_k.build_V_W_local ----------"
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
    bucnhsize = mydf._build_V_K_bunchsize

    # print("naux_involved = ", naux_involved)
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
        _build_W_local_bas_k_kernel,
    )

    # coul_G = tools.get_coulG(mydf.cell, mesh=mydf.mesh)
    coul_G = mydf.coul_G
    coul_G = ToTENSOR(coul_G, cpu=True).reshape(*mydf.mesh)
    coul_G = ToTENSOR(ToNUMPY(coul_G[:, :, : mydf.mesh[2] // 2 + 1].reshape(-1)).copy())

    V_loc = 0
    for group_id in range(group_begin_id, group_end_id):

        naux_tmp = mydf.IP_segment[group_id + 1] - mydf.IP_segment[group_id]

        for p0, p1 in lib.prange(0, naux_tmp, bucnhsize):

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

            _build_W_local_bas_k_kernel(
                V_tmp,
                mydf.aux_basis,
                mydf.kmesh,
                mydf.W[V_loc : V_loc + V_tmp.shape[0]],
            )

            V_loc += V_tmp.shape[0]

            buffer.free(count=1)

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

    misc._benchmark_time(t1, t2, "build_V_W_local", mydf, rank)

    if use_mpi:
        comm.barrier()

    return mydf.V, mydf.W


class ISDF_Local_K(ISDF_Local):

    def __init__(
        self,
        cell: Cell,
        with_robust_fitting=True,
        kmesh=None,
        kpts=None,
        aoR_cutoff=1e-8,
        direct=False,
        limited_memory=False,
        build_V_K_bunchsize=None,
        verbose=None,
        use_mpi=False,
    ):
        ### extract the info from the primitive cell ###

        atm = []

        #### TODO: remove the following restriction on the structure of lattice ####

        assert cell.a[0][1] == 0.0
        assert cell.a[0][2] == 0.0
        assert cell.a[1][0] == 0.0
        assert cell.a[1][2] == 0.0
        assert cell.a[2][0] == 0.0
        assert cell.a[2][1] == 0.0

        from pyscf.lib.parameters import BOHR

        for i in range(cell.natm):
            coords = cell.atom_coord(i)
            coords = np.array(coords) * BOHR
            atm.append([cell.atom_symbol(i), tuple(coords)])

        from pyscf.isdf.isdf_tools_Tsym import _make_kpts_kmesh

        kpts, kmesh = _make_kpts_kmesh(cell, kpts, kmesh)

        prim_mesh = cell.mesh
        mesh = np.array(prim_mesh) * np.array(kmesh)

        nelectron = np.sum(cell.nelectron)

        from pyscf.isdf.isdf_tools_cell import build_supercell

        supercell = build_supercell(
            atm,
            cell.a,
            spin=nelectron * np.prod(kmesh) % 2,
            mesh=mesh,
            Ls=kmesh,
            basis=cell.basis,
            pseudo=cell.pseudo,
            ke_cutoff=cell.ke_cutoff,
            max_memory=cell.max_memory,
            verbose=cell.verbose,
        )

        self.prim_cell = cell

        super().__init__(
            supercell,
            with_robust_fitting,
            None,
            None,
            aoR_cutoff,
            direct,
            limited_memory=limited_memory,
            build_V_K_bunchsize=build_V_K_bunchsize,
            verbose=verbose,
            use_mpi=use_mpi,
        )

        self.kpts = kpts.kpts
        self.kmesh = kmesh

        assert self.mesh[0] % kmesh[0] == 0
        assert self.mesh[1] % kmesh[1] == 0
        assert self.mesh[2] % kmesh[2] == 0

    ### build funcs ###

    def build(
        self, c=None, m=5, rela_cutoff=None, group=None, global_IP_selection=True
    ):
        self._build_cell_info()

        self._set_group(group)

        # preprocess #

        rela_cutoff = abs(rela_cutoff)
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

        self._build_aoR(group)

        # some info update #

        self._build_buffer(c, m, group)
        self._build_IP(c, m, rela_cutoff, group, global_IP_selection)
        self._build_fft_buffer()
        self._build_aux_basis(group)
        self._build_V_W()

    def _build_cell_info(self):

        #### information relating primitive cell and supercell

        self.meshPrim = np.array(self.mesh) // np.array(self.kmesh)
        self.ngridPrim = np.prod(self.meshPrim)
        self.natm = self.cell.natm
        self.natmPrim = self.cell.natm // np.prod(self.kmesh)
        self.with_translation_symmetry = True
        self.naoPrim = self.nao // np.prod(self.kmesh)
        assert self.naoPrim == self.prim_cell.nao_nr()

        super()._build_cell_info()

    def _build_aoR(self, group):

        from pyscf.isdf.isdf_eval_gto import ISDF_eval_gto

        ##### build partition #####

        lattice_x = self.cell.lattice_vectors()[0][0]
        lattice_y = self.cell.lattice_vectors()[1][1]
        lattice_z = self.cell.lattice_vectors()[2][2]

        Ls = [
            int(lattice_x) // 3 + 6,
            int(lattice_y) // 3 + 6,
            int(lattice_z) // 3 + 6,
        ]

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        self.partitionPrim = get_partition(
            self.cell,
            self.coords,
            self.AtmConnectionInfo,
            Ls,
            self.with_translation_symmetry,
            self.kmesh,
            self.use_mpi,
        )
        self.partitionPrim = [ToTENSOR(x) for x in self.partitionPrim]
        # self.partitionPrim = _canonicalize_partition_prim(self, self.partitionPrim)  # NOTE: one shold further polish the partition within the ref unit cell, TODO: do it
        assert len(self.partitionPrim) == self.natmPrim
        self.partition = _expand_partition_prim(
            self.partitionPrim, self.kmesh, self.mesh
        )
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
            first_natm,
            self.cell.natm,
            self.group_global,
            self.use_mpi,
            self.use_mpi,
        )  # COLs

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

        misc._benchmark_time(t1, t2, "get_aoR", self, self.rank)

        # build some indices info #

        # Create partition_group_2_gridID
        self.partition_group_2_gridID = [
            np.concatenate([self.partition[atm_id] for atm_id in subgroup]).astype(
                np.int64
            )  # NOTE: to be compatible with torch backend
            for subgroup in self.group_global
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
        self.gridID_ordering = _get_grid_ordering(self.partition, self.group_global)

        self.gridID_ordering_prim = ToTENSOR(
            ToNUMPY(self.gridID_ordering[: ngrids // np.prod(self.kmesh)]).copy()
        )

        def process_grid_id(grid_ID):
            ix = grid_ID // (self.mesh[1] * self.mesh[2])
            iy = (grid_ID % (self.mesh[1] * self.mesh[2])) // self.mesh[2]
            iz = grid_ID % self.mesh[2]

            ix = ix % self.meshPrim[0]
            iy = iy % self.meshPrim[1]
            iz = iz % self.meshPrim[2]

            return ix * self.meshPrim[1] * self.meshPrim[2] + iy * self.meshPrim[2] + iz

        self.gridID_ordering_prim = [
            process_grid_id(grid_ID) for grid_ID in self.gridID_ordering_prim
        ]

        # get grid segment #

        self.grid_segment = [0]
        for atm_id in self.atm_ordering:
            loc_now = self.grid_segment[-1] + len(self.partition[atm_id])
            self.grid_segment.append(loc_now)
        self.grid_segment = ToTENSOR(np.array(self.grid_segment, dtype=np.int64))

    def _set_group(self, group=None):
        first_natm = self.first_natm

        # Initialize group if None
        if group is None:
            self.group = [[i] for i in range(first_natm)]
        else:
            self.group = group

        # Check the group
        natm_involved = sum(len(data) for data in self.group)
        assert natm_involved == first_natm
        assert all(all(atm_id < first_natm for atm_id in data) for data in self.group)
        assert len(self.group) <= first_natm

        # Convert group elements to numpy arrays
        self.group = [np.array(data, dtype=np.int32) for data in self.group]

        # Generate group_global and atm_ordering
        shifts = (
            ix * self.natmPrim * self.kmesh[1] * self.kmesh[2]
            + iy * self.natmPrim * self.kmesh[2]
            + iz * self.natmPrim
            for ix, iy, iz in itertools.product(
                range(self.kmesh[0]), range(self.kmesh[1]), range(self.kmesh[2])
            )
        )

        shift = 0
        self.group_global = []
        self.atm_ordering = []
        for shift in shifts:
            for data in self.group:
                self.group_global.append(data + shift)
                self.atm_ordering.extend(data + shift)
            shift += self.natmPrim
        self.atm_ordering = ToTENSOR(np.array(self.atm_ordering, dtype=np.int64))

    def _build_IP(self, c, m, rela_cutoff, group, global_IP_selection):
        super()._build_IP(c, m, rela_cutoff, group, global_IP_selection)
        self.nauxPrim = self.naux
        self.naux *= np.prod(self.kmesh)

    def _build_V_W(self):
        self.coul_G = tools.get_coulG(self.cell, mesh=self.mesh)
        if not self.direct:
            self.V, self.W = build_V_W_local_k(self, self.use_mpi)

    @lru_cache(maxsize=None)
    def _aoR_calculate_permutation(self, box_x, box_y, box_z):
        assert all(
            0 <= box < kmesh for box, kmesh in zip((box_x, box_y, box_z), self.kmesh)
        )

        def calculate_new_nao_id(ao_id):
            box_id, nao_id = divmod(ao_id, self.naoPrim)
            box_x_, r = divmod(box_id, self.kmesh[1] * self.kmesh[2])
            box_y_, box_z_ = divmod(r, self.kmesh[2])

            new_box = [
                (box + off) % kmesh
                for box, off, kmesh in zip(
                    (box_x_, box_y_, box_z_), (box_x, box_y, box_z), self.kmesh
                )
            ]

            return (
                new_box[0] * self.kmesh[1] * self.kmesh[2]
                + new_box[1] * self.kmesh[2]
                + new_box[2]
            ) * self.naoPrim + nao_id

        return [
            ToTENSOR(
                np.array(
                    [calculate_new_nao_id(ao_id) for ao_id in aoR_holder.ao_involved],
                    dtype=np.int64,
                )
            )
            for aoR_holder in self.aoR
        ]

    @lru_cache(maxsize=None)
    def _aoRg_calculate_permutation(self, box_x, box_y, box_z):
        assert all(
            0 <= box < kmesh for box, kmesh in zip((box_x, box_y, box_z), self.kmesh)
        )

        def calculate_new_nao_id(ao_id):
            box_id, nao_id = divmod(ao_id, self.naoPrim)
            box_x_, r = divmod(box_id, self.kmesh[1] * self.kmesh[2])
            box_y_, box_z_ = divmod(r, self.kmesh[2])

            new_box = [
                (box + off) % kmesh
                for box, off, kmesh in zip(
                    (box_x_, box_y_, box_z_), (box_x, box_y, box_z), self.kmesh
                )
            ]

            return (
                new_box[0] * self.kmesh[1] * self.kmesh[2]
                + new_box[1] * self.kmesh[2]
                + new_box[2]
            ) * self.naoPrim + nao_id

        return [
            ToTENSOR(
                np.array(
                    [calculate_new_nao_id(ao_id) for ao_id in aoR_holder.ao_involved],
                    dtype=np.int64,
                )
            )
            for aoR_holder in self.aoRg
        ]

    def _get_permutation_column_aoR(self, box_x, box_y, box_z, loc_internal=None):
        permutation = self._aoR_calculate_permutation(box_x, box_y, box_z)

        if loc_internal is not None:
            return permutation[loc_internal]
        else:
            return permutation

    def _get_permutation_column_aoRg(self, box_x, box_y, box_z, loc_internal=None):
        permutation = self._aoRg_calculate_permutation(box_x, box_y, box_z)

        if loc_internal is not None:
            return permutation[loc_internal]
        else:
            return permutation

    @lru_cache(maxsize=None)
    def _aoR_ref_unit(self, iatm_prim, box_x, box_y, box_z):
        assert all(
            0 <= box < kmesh for box, kmesh in zip((box_x, box_y, box_z), self.kmesh)
        )

        # box_id = box_x * self.kmesh[1] * self.kmesh[2] + box_y * self.kmesh[2] + box_z
        target_box_x = (self.kmesh[0] - box_x) % self.kmesh[0]
        target_box_y = (self.kmesh[1] - box_y) % self.kmesh[1]
        target_box_z = (self.kmesh[2] - box_z) % self.kmesh[2]

        ao_shift = (
            target_box_x * self.kmesh[1] * self.kmesh[2]
            + target_box_y * self.kmesh[2]
            + target_box_z
        )
        grid_shift = ao_shift * self.ngridPrim
        ao_shift *= self.naoPrim

        ao_begin = ao_shift
        ao_end = ao_shift + self.naoPrim

        _aoR_Holder = self.aoR[iatm_prim]
        gridID_begin = _aoR_Holder.global_gridID_begin + grid_shift
        ngrids_involved = _aoR_Holder.aoR.shape[1]

        pos = [
            loc
            for loc, aoID in enumerate(_aoR_Holder.ao_involved)
            if aoID >= ao_begin and aoID < ao_end
        ]

        loc_begin = pos[0]
        loc_end = pos[-1] + 1
        assert loc_end - loc_begin == len(pos)

        ao_involved = ToTENSOR(
            np.array(
                [_aoR_Holder.ao_involved[loc] - ao_begin for loc in pos], dtype=np.int64
            )
        )

        return aoR_Holder(
            _aoR_Holder.aoR[loc_begin:loc_end],
            ao_involved,
            gridID_begin,
            gridID_begin + ngrids_involved,
        )

    get_jk = get_jk_dm_k_local
