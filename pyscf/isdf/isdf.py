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

NUM_THREADS = BACKEND.NUM_THREADS

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

    misc._debug4(mydf.verbose, None, "In pyscf.isdf.isdf.select_IP")
    misc._debug4(mydf.verbose, None, "select_IP: num_threads = %d", NUM_THREADS)


def build_aux_basis(mydf, debug=True):
    """build the auxiliary basis for ISDF given IP_ID and aoR."""
    pass


def constrcuct_V(aux_basis: np.ndarray, mesh, coul_G, buf):
    pass


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

        self.cell = cell.copy()
        self.mesh = self.cell.mesh
        self.natm = self.cell.natm
        self.nao = self.cell.nao_nr()
        self.ke_cutoff = self.cell.ke_cutoff

        misc._info(self, None, "ISDF: mol.ke_cutoff = %f", self.cell.ke_cutoff)

        ao2atomID = np.zeros(self.nao, dtype=np.int32)
        self.ao2atomID = ao2atomID

        self.coords = np.asarray(self.grids.coords).reshape(-1, 3)
        self.ngrids = self.coords.shape[0]

        # the following variables are to be built

        # buffer

        self.buffer = None

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

    ### build ###

    def build(self):
        pass

    def _build_aoR(self):
        pass

    def _build_IP(self, c=5, m=5, global_IP_selection=True, build_global_basis=True):
        pass

    ### properties ###

    @property
    def kpt(self):
        return np.zeros(3)

    @property
    def aoRgTensor(self):
        ToTensor = BACKEND._toTensor
        return ToTensor(self.aoRg), None

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

                nao_prim = self.cell.nao_nr() // nkpts
                assert self.cell.nao_nr() % nkpts == 0

                from pyscf.isdf.isdf_tools_kSampling import _RowCol_FFT_bench
                from pyscf.isdf.isdf_tools_Tsym import (
                    pack_JK_in_FFT_space,
                    symmetrize_mat,
                )

                ##### NOTE: first symmetrization #####

                self.PP = symmetrize_mat(self.PP, kmesh)

                PP_complex = _RowCol_FFT_bench(
                    self.PP[:nao_prim, :],
                    kmesh,
                    inv=False,
                    TransBra=False,
                    TransKet=True,
                )
                PP_complex = PP_complex.conj().copy()
                self.PP = []
                for i in range(nkpts):
                    self.PP.append(
                        PP_complex[:, i * nao_prim : (i + 1) * nao_prim].copy()
                    )

            if self.use_mpi:
                from pyscf.isdf.isdf_tools_mpi import bcast

                self.PP = bcast(self.PP, root=0)

            return self.PP
