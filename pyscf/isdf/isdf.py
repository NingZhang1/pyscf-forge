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
import numpy
import ctypes

############ pyscf module ############

from pyscf import lib
from pyscf.lib import logger
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf.gto.mole import ATOM_OF, NCTR_OF, ANG_OF
from pyscf.pbc.dft import multigrid
from pyscf.pbc import df

############ isdf utils ############

import pyscf.isdf.misc as misc

############ isdf backends ############

import pyscf.isdf.BackEnd.isdf_backend as BACKEND
from pyscf.isdf.BackEnd.isdf_memory_allocator import SimpleMemoryAllocator

USE_GPU = BACKEND.USE_GPU

NUM_THREADS = BACKEND.NUM_THREADS
ToNUMPY = BACKEND._toNumpy
ToTENSOR = BACKEND._toTensor
MAX = BACKEND._maximum
ABS = BACKEND._absolute
FLOAT64 = BACKEND.FLOAT64
DOT = BACKEND._dot

############ global variables ############

BASIS_CUTOFF = 1e-18  # too small may lead to numerical instability

############ subroutines --- select IP and build aux basis ############


def select_IP(
    mydf,
    c: int,
    m: int,
    rela_cutoff=0.0,
    first_natm=None,
    global_IP_selection=True,
    aoR_cutoff=None,
    no_retriction_on_nIP=False,
    use_mpi=False,
):
    r"""Select the interpolation points (IP) based on the given criteria.

    Args:
        mydf : object
            The interpolative separable density fitting (ISDF) object.

        c : int
            if rela_cutoff is None or 0.0, control the number of IPs selected with c * nao at most.


        rela_cutoff : float
            The relative cutoff value for IP selection.
            IPs with values smaller than rela_cutoff * max_QR_value will not be selected.
            Default is 0.0. (no control via QR values)


    Kwargs:
        first_natm : int
            The number of atoms to be considered for IP selection.
            If not given, all atoms will be considered.
            If set,       it *should* be used in ISDF with k-sampling class, first_natm is the number of atoms in the first cell.

        global_IP_selection : bool
            Whether to perform global IP selection.
            If True, IPs will be re-selected after the individual selection of each atom.
            Default is True.

        aoR_cutoff : float
            The cutoff value for AO values.
            Points with max AO values smaller than this cutoff will not be considered for IP selection.
            Default is None.

        no_retriction_on_nIP : bool
            Whether to remove the restriction on the number of IPs.
            If True, there will be no limit on the number of selected IPs.
            Default is False.

        m : int
            Control the number of

        Returns:
            selected_IP : list
                The list of selected interpolation points.

        Ref:

        (1) Sandeep2022 https://pubs.acs.org/doi/10.1021/acs.jctc.2c00720

    """

    # print info #

    misc._debug4(mydf, mydf.rank, "In pyscf.isdf.isdf.select_IP")
    misc._debug4(mydf, mydf.rank, "select_IP: num_threads = %d", NUM_THREADS)

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    # buffer #

    buffer = mydf.buffer
    buffer.free_all()

    # funcs #

    QR = BACKEND._qr
    QR_PIVOT = BACKEND._qr_col_pivoting
    EINSUM_IK_JK_IJK = BACKEND._einsum_ik_jk_ijk
    TAKE = BACKEND._take

    if first_natm is None:
        first_natm = mydf.natm
    else:
        assert first_natm == mydf.natm

    results = []

    # loop over atms #

    for atm_id in range(mydf.natm):

        # get the involved ao values #

        grid_ID = ToTENSOR(np.where(ToNUMPY(mydf.partition) == atm_id)[0], cpu=not USE_GPU)
        aoR_atm = buffer.malloc(
            (mydf.nao, grid_ID.shape[0]), dtype=FLOAT64, name="aoRatm"
        )
        TAKE(mydf.aoR, grid_ID, axis=1, out=aoR_atm)

        if aoR_cutoff is not None:
            misc._debug4(mydf, mydf.rank, "select_IP: aoR_cutoff = %12.6e", aoR_cutoff)
            max_row_id = MAX(ABS(aoR_atm), axis=1)
            where = ToTENSOR(np.where(ToNUMPY(max_row_id) > aoR_cutoff)[0])
            aoR_atm2 = buffer.malloc(
                (where.shape[0], grid_ID.shape[0]),
                dtype=FLOAT64,
                name="aoRatm2",
            )
            TAKE(aoR_atm, where, axis=0, out=aoR_atm2)
            aoR_atm = aoR_atm2

        nao_tmp = aoR_atm.shape[0]

        # random projection #

        nao_atm = mydf.atmID2nao[atm_id]
        naux_now = int(np.sqrt(nao_atm * c) + m)
        naux_now = min(naux_now, nao_tmp)
        naux2_now = naux_now**2

        G1 = ToTENSOR(np.random.randn(nao_tmp, naux_now), cpu=not USE_GPU)
        G1, _ = QR(G1, mode="economic")
        G1 = G1.T
        G2 = ToTENSOR(np.random.randn(nao_tmp, naux_now), cpu=not USE_GPU)
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

        if global_IP_selection:
            if no_retriction_on_nIP:
                max_rank = min(naux2_now, grid_ID.shape[0])
            else:
                max_rank = min(naux2_now, grid_ID.shape[0], nao_atm * c + m)
        else:
            if no_retriction_on_nIP:
                max_rank = min(naux2_now, grid_ID.shape[0])
            else:
                max_rank = min(naux2_now, grid_ID.shape[0], nao_atm * c)

        R = ToNUMPY(R)
        threshold = abs(R[0, 0]) * rela_cutoff
        indices = np.where(np.abs(np.diag(R)) > threshold)[0]
        npt_found = min(len(indices), max_rank)
        pivot = ToNUMPY(pivot[:npt_found])
        pivot.sort()
        results.extend(list(ToNUMPY(grid_ID[pivot])))

        misc._debug4(
            mydf,
            mydf.rank,
            "select_IP: ngrid = %6d, npt_find = %6d, cutoff = %12.6e",
            grid_ID.shape[0],
            npt_found,
            rela_cutoff,
        )

        # free all the buffer allocated #

        buffer.free_all()

    ## global selection ##

    results.sort()
    results = np.asarray(results, dtype=np.int32)
    results = ToTENSOR(results, cpu=not USE_GPU)

    if global_IP_selection:

        nao = mydf.nao

        aoRg = buffer.malloc((mydf.nao, results.shape[0]), dtype=FLOAT64, name="aoRg")
        TAKE(mydf.aoR, results, axis=1, out=aoRg)

        # random projection #

        naux_now = int(np.sqrt(nao * c) + m)
        naux_now = min(naux_now, nao)
        naux2_now = naux_now**2

        aoRg1 = buffer.malloc((naux_now, results.shape[0]), dtype=FLOAT64, name="aoRg1")
        aoRg2 = buffer.malloc((naux_now, results.shape[0]), dtype=FLOAT64, name="aoRg2")
        aoPairRg = buffer.malloc(
            (naux_now, naux_now, results.shape[0]), dtype=FLOAT64, name="aoPairRg"
        )

        G1 = ToTENSOR(np.random.randn(nao, naux_now), cpu=not USE_GPU)
        G1, _ = QR(G1, mode="economic")
        G1 = G1.T
        G2 = ToTENSOR(np.random.randn(nao, naux_now), cpu=not USE_GPU)
        G2, _ = QR(G2, mode="economic")
        G2 = G2.T

        DOT(G1, aoRg, c=aoRg1)
        DOT(G2, aoRg, c=aoRg2)
        EINSUM_IK_JK_IJK(aoRg1, aoRg2, out=aoPairRg)
        aoPairRg = aoPairRg.reshape(naux2_now, results.shape[0])

        # qr pivot #

        _, R, pivot = QR_PIVOT(aoPairRg, mode="r")

        if no_retriction_on_nIP:
            max_rank = min(naux2_now, results.shape[0])
        else:
            max_rank = min(naux2_now, results.shape[0], nao * c)

        R = ToNUMPY(R)
        threshold = abs(R[0, 0]) * rela_cutoff
        indices = np.where(np.abs(np.diag(R)) > threshold)[0]
        npt_found = min(len(indices), max_rank)
        pivot = ToNUMPY(pivot[:npt_found])
        pivot.sort()

        misc._debug4(
            mydf,
            mydf.rank,
            "select_IP: ngrid = %6d, npt_find = %6d, cutoff = %12.6e",
            results.shape[0],
            npt_found,
            rela_cutoff,
        )

        results = ToNUMPY(results)
        results = results[pivot]
        results = ToTENSOR(results)

    buffer.free_all()

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    misc._benchmark_time(t1, t2, "select_IP", mydf, mydf.rank)

    return results


def build_aux_basis(mydf, use_mpi=False):
    """build the auxiliary basis for ISDF given IP_ID and aoR."""
    assert not use_mpi

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    # allocate buffer #

    naux = mydf.IP_ID.shape[0]
    aoRg = mydf.aoRg
    buffer = mydf.buffer
    ngrids = mydf.ngrids

    # func #

    SQUARE_ = BACKEND._square_
    CHO_SOLVE = BACKEND._solve_cholesky

    # constracut A and B #

    A = buffer.malloc((naux, naux), dtype=FLOAT64, name="A")
    DOT(aoRg.T, aoRg, c=A)
    SQUARE_(A)

    B = buffer.malloc((naux, ngrids), dtype=FLOAT64, name="B")
    DOT(aoRg.T, mydf.aoR, c=B)
    SQUARE_(B)

    # build aux basis, AX=B #

    mydf.aux_basis = CHO_SOLVE(A, B, overwrite_b=False)

    # finish #

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    misc._benchmark_time(t1, t2, "build_aux_basis", mydf, mydf.rank)
    buffer.free_all()


def construct_V(mydf, use_mpi=False):

    assert not use_mpi

    # func #

    EINSUM_IJ_J_IJ = BACKEND._einsum_ij_j_ij
    RFFTN = BACKEND._rfftn
    IRFFTN = BACKEND._irfftn

    # reshape aux_basis #

    mesh = mydf.mesh
    mesh = tuple(mesh)
    mesh_complex = (mesh[0], mesh[1], mesh[2] // 2 + 1)
    aux_basis = mydf.aux_basis.reshape(-1, mesh[0], mesh[1], mesh[2])
    coul_G = (
        ToNUMPY(mydf.coul_G).reshape(*mesh)[:, :, : mesh[2] // 2 + 1].reshape(-1).copy()
    )
    coul_G = ToTENSOR(coul_G, cpu=not USE_GPU)

    # construct V #
    
    tmp1 = RFFTN(aux_basis, s=mesh, axes=(1, 2, 3), overwrite_input=False)
    tmp1 = tmp1.reshape(-1, np.prod(mesh_complex))
    EINSUM_IJ_J_IJ(tmp1, coul_G, out=tmp1)
    tmp1 = tmp1.reshape(-1, *mesh_complex)
    V = IRFFTN(tmp1, s=mesh, axes=(1, 2, 3), overwrite_input=False).reshape(
        -1, np.prod(mesh)
    )

    del tmp1

    return V


class ISDF(df.fft.FFTDF):
    """Interpolative separable density fitting (ISDF) for periodic systems.
    Not recommended as the locality is not explored!
    """

    def __init__(
        self,
        cell: Cell,
        # aoR: np.ndarray = None,  ## convention: aoR is scaled by np.sqrt(mol.vol / ngrids)
        with_robust_fitting=True,
        kmesh=None,
        kpts=None,
        get_partition=True,
        verbose=None,
        use_mpi=False,
    ):

        # deal with kmesh and kpts

        from pyscf.isdf.isdf_tools_Tsym import _make_kpts_kmesh

        kpts, kmesh = _make_kpts_kmesh(cell, kpts, kmesh)

        # call the parent class

        super().__init__(cell=cell, kpts=kpts)

        if verbose is not None:
            self.verbose = verbose

        # the following variables are used in build

        self.kmesh = kmesh
        self.with_robust_fitting = with_robust_fitting

        # cell

        self.cell = cell.copy()
        self.mesh = self.cell.mesh
        self.natm = self.cell.natm
        self.nao = self.cell.nao_nr()
        self.ke_cutoff = self.cell.ke_cutoff

        self.coords = np.asarray(self.grids.coords).reshape(-1, 3)
        self.ngrids = self.coords.shape[0]
        self.mesh = np.asarray(self.grids.mesh)

        # the following variables are to be built

        self.ao2atomID = None
        self.atmID2nao = None

        self.aoR = None  # TensorTy, cpu/gpu
        self.partition = None  # TensorTy, cpu

        self.IP_ID = None  # TensorTy, cpu
        self.naux = None
        self.aoRg = None  # TensorTy, cpu/gpu
        self.aux_basis = None  # TensorTy, cpu/gpu

        self.V = None  # TensorTy, cpu/gpu
        self.W = None  # TensorTy, cpu/gpu

        # buffer

        self.buffer = None
        self.fft_buffer = None  # must be singled out due to the alignment problem

        # other properties

        self._use_super_pp = False
        self._use_FFTDF_pp = self._use_super_pp
        self.ke_cutoff_pp = self.ke_cutoff
        self.use_mpi = use_mpi

        # init MPI

        if self.use_mpi:
            from pyscf.isdf.isdf_tools_mpi import rank

            self.rank = rank
        else:
            self.rank = 0

        misc._info(self, self.rank, "ISDF: mol.ke_cutoff = %f", self.cell.ke_cutoff)
        misc._info(self, self.rank, "ISDF: mol.natm      = %d", self.natm)
        misc._info(self, self.rank, "ISDF: mol.nao       = %d", self.nao)

    ### build ###

    def build(self, c=None, m=5, rela_cutoff=None, global_IP_selection=True):

        if c is None:
            c = 15

        self._build_cell_info()
        self._build_aoR()
        self._build_buffer(c, m)
        self._build_fft_buffer()
        self._build_IP(c, m, rela_cutoff, global_IP_selection)
        self._build_aux_basis()
        self._build_V_W()

    def _build_cell_info(self):

        ao2atomID = np.zeros(self.nao, dtype=np.int32)
        ao_loc = 0
        for i in range(self.cell._bas.shape[0]):
            atm_id = self.cell._bas[i, ATOM_OF]
            nctr = self.cell._bas[i, NCTR_OF]
            angl = self.cell._bas[i, ANG_OF]
            nao_now = nctr * (2 * angl + 1)  # NOTE: sph basis assumed!
            ao2atomID[ao_loc : ao_loc + nao_now] = atm_id
            ao_loc += nao_now

        atmID2nao = np.zeros(self.natm, dtype=np.int32)
        for i in range(self.natm):
            atmID2nao[i] = np.sum(ao2atomID == i)

        self.ao2atomID = ToTENSOR(ao2atomID)
        self.atmID2nao = ToTENSOR(atmID2nao)

    def _build_aoR(self):

        # NOTE: currently, do not consider k-sampling case #

        from pyscf.isdf.isdf_eval_gto import ISDF_eval_gto

        self.aoR = ISDF_eval_gto(self.cell, coords=self.coords) * np.sqrt(
            self.cell.vol / self.ngrids
        )

        max_id = np.argmax(ToNUMPY(ABS(self.aoR)), axis=0)
        self.partition = np.asarray([self.ao2atomID[x] for x in max_id])
        self.partition = ToTENSOR(self.partition)

        self.aoR = ToTENSOR(self.aoR, cpu=not USE_GPU)

    def _build_buffer(self, c, m):
        naux_max = self.naux_max(c, m)
        naux_max_atm_sqrt = self.nauxMaxPerAtm_sqrt(c, m)
        ngrid_max_atm = self.max_ngrid_atm()
        # used in select_IP #
        size1 = self.nao * ngrid_max_atm * 2
        size1 += naux_max_atm_sqrt * ngrid_max_atm * 2  # aoR_atm1, aoR_atm2
        size1 += (
            naux_max_atm_sqrt * naux_max_atm_sqrt * ngrid_max_atm
        )  # aoR_atm1, aoR_atm2
        naux_max_now = int(np.sqrt(self.nao * c) + m)
        size2 = self.nao * naux_max
        size2 += naux_max_now * naux_max * 2  # aoRg1, aoRg2
        size2 += naux_max_now * naux_max_now * naux_max  # aoPairRg
        # used in build_aux_basis #
        size3 = naux_max * naux_max + naux_max * self.ngrids
        # used in get J #
        # used in get K #
        # allocate buffer #
        size = max(size1, size2, size3)
        self.buffer = SimpleMemoryAllocator(total_size=size, gpu=USE_GPU)
        pass

    def _build_fft_buffer(self):
        # for this simple impl of ISDF there is no need to build fft_buffer #
        pass

    def _build_IP(self, c=5, m=5, rela_cutoff=None, global_IP_selection=True):
        self.IP_ID = select_IP(
            self,
            c=c,
            m=m,
            rela_cutoff=rela_cutoff,
            first_natm=None,
            global_IP_selection=global_IP_selection,
            aoR_cutoff=None,
            no_retriction_on_nIP=(rela_cutoff is not None),
            use_mpi=False,
        )
        self.IP_ID = ToTENSOR(self.IP_ID, cpu=not USE_GPU)
        self.aoRg = BACKEND._take(self.aoR, self.IP_ID, axis=1)
        assert self.aoRg.shape[1] == self.IP_ID.shape[0]
        assert self.aoRg.shape[0] == self.nao

    def _build_aux_basis(self):
        build_aux_basis(self)

    def _build_V_W(self):
        """build V and W matrix see eq(13) of Sandeep2022.

        Ref:
        (1) Sandeep2022 https://pubs.acs.org/doi/10.1021/acs.jctc.2c00720
        """

        coul_G = tools.get_coulG(self.cell, mesh=self.mesh)
        self.coul_G = ToTENSOR(coul_G, cpu=not USE_GPU)

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

        self.V = construct_V(self)

        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

        misc._benchmark_time(t1, t2, "construct_V", self, self.rank)

        self.W = DOT(self.aux_basis, self.V.T)

        t3 = (lib.logger.process_clock(), lib.logger.perf_counter())

        misc._benchmark_time(t2, t3, "construct_W", self, self.rank)

        if not self.with_robust_fitting:
            del self.V
            self.V = None

    ### properties ###

    @property
    def kpt(self):
        return np.zeros(3)

    @property
    def aoRgTensor(self):
        ToTensor = BACKEND._toTensor
        return ToTensor(self.aoRg), None

    def nauxMaxPerAtm_sqrt(self, c, m):
        res = 0
        for nao_atm in ToNUMPY(self.atmID2nao):
            res = max(res, int(np.sqrt(nao_atm * c) + m))
        # res = max(res, int(np.sqrt(self.nao * c) + m))
        return res

    def naux_max(self, c, m):
        res1 = 0
        for nao_atm in ToNUMPY(self.atmID2nao):
            res1 += int(np.sqrt(nao_atm * c) + m) ** 2
        res2 = int(np.sqrt(self.nao * c) + m) ** 2
        return max(res1, res2)

    def max_ngrid_atm(self):
        partition = ToNUMPY(self.partition)
        ngrid_atm = np.zeros(self.natm, dtype=np.int32)
        for i in range(self.ngrids):
            ngrid_atm[partition[i]] += 1
        return np.max(ngrid_atm)

    ### utils to infer the size of certain quantities ###

    ### get_pp ###

    def get_pp(self, kpts=None):
        if hasattr(self, "PP") and self.PP is not None:
            return self.PP
        else:

            use_super_pp = False

            if self._use_super_pp:
                use_super_pp = True
                t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
                if hasattr(self, "prim_cell"):
                    from pyscf.pbc.df.fft import FFTDF

                    df_tmp = FFTDF(self.prim_cell, kpts)
                    self.PP = df_tmp.get_pp(kpts=kpts)
                else:
                    self.PP = super().get_pp(kpts=np.zeros(3))
                t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

            if not use_super_pp:
                t0 = (lib.logger.process_clock(), lib.logger.perf_counter())
                cell = self.cell.copy()
                cell.omega = 0.0
                cell.ke_cutoff = self.ke_cutoff_pp
                cell.build()
                df_tmp = multigrid.MultiGridFFTDF2(cell)
                v_pp_loc2_nl = df_tmp.get_pp()
                v_pp_loc1_G = df_tmp.vpplocG_part1
                v_pp_loc1 = multigrid.multigrid_pair._get_j_pass2(df_tmp, v_pp_loc1_G)
                self.PP = (v_pp_loc1 + v_pp_loc2_nl)[0]
                t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

            misc._benchmark_time(t0, t1, "get_pp", self, self.rank)

            #### kpts ####

            if kpts is not None:

                nkpts = kpts.shape[0]
                kmesh = np.asarray(self.kmesh, dtype=np.int32)
                assert (
                    kpts.shape[0] == np.prod(self.kmesh, dtype=np.int32)
                    or kpts.shape[0] == 1
                    or kpts.ndim == 1
                )
                is_single_kpt = kpts.shape[0] == 1 or kpts.ndim == 1

                if is_single_kpt:
                    #### use the calculated one by default ####
                    if self.use_mpi:
                        from pyscf.isdf.isdf_tools_mpi import bcast

                        self.PP = bcast(self.PP, root=0)
                    return self.PP

                #### the following is used to test KRHF ####

                if self._use_super_pp:
                    if self.use_mpi:
                        from pyscf.isdf.isdf_tools_mpi import bcast

                        self.PP = bcast(self.PP, root=0)
                    return self.PP

                ### info used in super().get_pp() ###

                assert hasattr(self, "prim_cell")
                assert self.cell.nao_nr() % nkpts == 0

                from pyscf.isdf.isdf_tools_Tsym import (
                    pack_JK_in_FFT_space,
                    symmetrize_mat,
                    _1e_operator_gamma2k,
                )

                ##### NOTE: first symmetrization #####

                self.PP = symmetrize_mat(self.PP, kmesh)
                self.PP = _1e_operator_gamma2k(self.cell, kmesh, self.PP)

            if self.use_mpi:
                from pyscf.isdf.isdf_tools_mpi import bcast

                self.PP = bcast(self.PP, root=0)

            return self.PP

    ##### functions defined in isdf_ao2mo.py #####
    ##### functions defined in isdf_jk.py    #####
