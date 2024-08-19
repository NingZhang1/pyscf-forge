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

############ pyscf module ############

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point
from pyscf.pbc.df.df_jk import _ewald_exxdiv_for_G0
from pyscf.pbc.df.df_jk import _format_dms, _format_kpts_band, _format_jks

libisdf = lib.load_library("libisdf")

############ isdf utils ############

from pyscf.isdf.isdf_jk import _benchmark_time
from pyscf.isdf._isdf_local_K_direct import _isdf_get_K_direct_kernel_1
import pyscf.isdf.isdf_tools_linearop as lib_isdf
from pyscf.isdf.isdf_fast import EXTRA_ALLOC

############ GLOBAL PARAMETER ############

J_MAX_GRID_BUNCHSIZE = 8192

##################################################
#
# only Gamma Point
#
##################################################

### ls = linear scaling


def _half_J(mydf, dm, use_mpi=False, first_pass=None, short_range=False):

    if use_mpi:
        if not mydf.direct:
            raise ValueError("MPI is only supported for direct method.")
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast, reduce

    ######### prepare the parameter #########

    assert first_pass in [None, "only_dd", "only_cc", "exclude_cc", "all"]

    if first_pass is None:
        first_pass = "all"

    first_pass_all = first_pass == "all"
    first_pass_has_dd = first_pass in ["all", "only_dd", "exclude_cc"]
    first_pass_has_cc = first_pass in ["all", "only_cc"]
    first_pass_has_cd = first_pass in ["all", "exclude_cc"]

    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    nao = dm.shape[0]
    cell = mydf.cell
    assert cell.nao == nao
    mesh = np.array(cell.mesh, dtype=np.int32)
    ngrid = np.prod(mesh)

    aoR = mydf.aoR
    assert isinstance(aoR, list)

    #### step 0. allocate buffer

    max_nao_involved = np.max(
        [aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None]
    )
    max_ngrid_involved = np.max(
        [aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None]
    )
    ngrids_local = np.sum(
        [aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None]
    )

    density_R = np.zeros((ngrid,), dtype=np.float64)

    dm_buf = np.zeros((max_nao_involved, max_nao_involved), dtype=np.float64)
    max_col_buf = min(max_ngrid_involved, J_MAX_GRID_BUNCHSIZE)
    ddot_buf = np.zeros((max_nao_involved, max_col_buf), dtype=np.float64)

    fn_multiplysum = getattr(libisdf, "_fn_J_dmultiplysum", None)
    assert fn_multiplysum is not None

    ##### get the involved C function #####

    fn_extract_dm = getattr(libisdf, "_extract_dm_involved_ao", None)
    assert fn_extract_dm is not None

    fn_extract_dm2 = getattr(libisdf, "_extract_dm_involved_ao_RS", None)
    assert fn_extract_dm is not None

    fn_packadd_dm = getattr(libisdf, "_packadd_local_dm", None)
    assert fn_packadd_dm is not None

    #### step 1. get density value on real space grid and IPs

    density_R_tmp = None
    density_R_tmp_buf = np.zeros((max_ngrid_involved,), dtype=np.float64)

    def _get_rhoR(
        bra_aoR,
        bra_ao_involved,
        ket_aoR,
        ket_ao_involved,
        bra_type,
        ket_type,
        dm_buf=dm_buf,
        ddot_buf=ddot_buf,
        density_R_tmp_buf=density_R_tmp_buf,
    ):

        nbra_ao = bra_aoR.shape[0]
        nket_ao = ket_aoR.shape[0]
        if bra_type == ket_type:
            dm_now = np.ndarray((nbra_ao, nbra_ao), buffer=dm_buf)
            fn_extract_dm(
                dm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao),
                dm_now.ctypes.data_as(ctypes.c_void_p),
                bra_ao_involved.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nbra_ao),
            )

            # _density_R_tmp = np.zeros((ket_aoR.shape[1],), dtype=np.float64)
            _density_R_tmp = np.ndarray((ket_aoR.shape[1],), buffer=density_R_tmp_buf)

            for p0, p1 in lib.prange(0, ket_aoR.shape[1], J_MAX_GRID_BUNCHSIZE):
                ddot_res = np.ndarray((nbra_ao, p1 - p0), buffer=ddot_buf)
                lib.ddot(dm_now, ket_aoR[:, p0:p1], c=ddot_res)
                _res_tmp = np.ndarray(
                    (p1 - p0,),
                    dtype=_density_R_tmp.dtype,
                    buffer=_density_R_tmp,
                    offset=p0 * _density_R_tmp.dtype.itemsize,
                )
                fn_multiplysum(
                    _res_tmp.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbra_ao),
                    ctypes.c_int(p1 - p0),
                    bra_aoR.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(bra_aoR.shape[0]),
                    ctypes.c_int(bra_aoR.shape[1]),
                    ctypes.c_int(0),
                    ctypes.c_int(p0),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbra_ao),
                    ctypes.c_int(p1 - p0),
                    ctypes.c_int(0),
                    ctypes.c_int(0),
                )
            return _density_R_tmp
        else:
            dm_now = np.ndarray((nbra_ao, nket_ao), buffer=dm_buf)
            fn_extract_dm2(
                dm.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao),
                dm_now.ctypes.data_as(ctypes.c_void_p),
                bra_ao_involved.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(bra_ao_involved.shape[0]),
                ket_ao_involved.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ket_ao_involved.shape[0]),
            )
            # _density_R_tmp = np.zeros((ket_aoR.shape[1],), dtype=np.float64)
            _density_R_tmp = np.ndarray((ket_aoR.shape[1],), buffer=density_R_tmp_buf)

            for p0, p1 in lib.prange(0, ket_aoR.shape[1], J_MAX_GRID_BUNCHSIZE):
                ddot_res = np.ndarray((nbra_ao, p1 - p0), buffer=ddot_buf)
                lib.ddot(dm_now, ket_aoR[:, p0:p1], c=ddot_res)
                _res_tmp = np.ndarray(
                    (p1 - p0,),
                    dtype=_density_R_tmp.dtype,
                    buffer=_density_R_tmp,
                    offset=p0 * _density_R_tmp.dtype.itemsize,
                )
                fn_multiplysum(
                    _res_tmp.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbra_ao),
                    ctypes.c_int(p1 - p0),
                    bra_aoR.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(bra_aoR.shape[0]),
                    ctypes.c_int(bra_aoR.shape[1]),
                    ctypes.c_int(0),
                    ctypes.c_int(p0),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nbra_ao),
                    ctypes.c_int(p1 - p0),
                    ctypes.c_int(0),
                    ctypes.c_int(0),
                )

            return _density_R_tmp * 2.0

    for atm_id, aoR_holder in enumerate(aoR):

        if aoR_holder is None:
            continue

        if use_mpi:
            if atm_id % comm_size != rank:
                continue

        ngrids_now = aoR_holder.aoR.shape[1]
        global_gridID_begin = aoR_holder.global_gridID_begin
        nCompact = aoR_holder.nCompact

        if first_pass_all:
            density_R_tmp = _get_rhoR(
                aoR_holder.aoR,
                aoR_holder.ao_involved,
                aoR_holder.aoR,
                aoR_holder.ao_involved,
                "all",
                "all",
            )

            density_R[global_gridID_begin : global_gridID_begin + ngrids_now] = (
                density_R_tmp
            )
        else:

            if first_pass_has_cc:
                density_R_tmp = _get_rhoR(
                    aoR_holder.aoR[:nCompact, :],
                    aoR_holder.ao_involved[:nCompact],
                    aoR_holder.aoR[:nCompact, :],
                    aoR_holder.ao_involved[:nCompact],
                    "compact",
                    "compact",
                )

                density_R[
                    global_gridID_begin : global_gridID_begin + ngrids_now
                ] += density_R_tmp

            if first_pass_has_dd:
                density_R_tmp = _get_rhoR(
                    aoR_holder.aoR[nCompact:, :],
                    aoR_holder.ao_involved[nCompact:],
                    aoR_holder.aoR[nCompact:, :],
                    aoR_holder.ao_involved[nCompact:],
                    "diffuse",
                    "diffuse",
                )

                density_R[
                    global_gridID_begin : global_gridID_begin + ngrids_now
                ] += density_R_tmp

            if first_pass_has_cd:
                density_R_tmp = _get_rhoR(
                    aoR_holder.aoR[:nCompact, :],
                    aoR_holder.ao_involved[:nCompact],
                    aoR_holder.aoR[nCompact:, :],
                    aoR_holder.ao_involved[nCompact:],
                    "compact",
                    "diffuse",
                )
                density_R[
                    global_gridID_begin : global_gridID_begin + ngrids_now
                ] += density_R_tmp

    # assert local_grid_loc == ngrids_local

    if use_mpi:
        density_R = reduce(density_R, root=0)
    else:
        assert ngrids_local == np.prod(mesh)

    grid_ID_ordered = mydf.grid_ID_ordered

    if (use_mpi and rank == 0) or (not use_mpi):
        density_R_original = np.zeros_like(density_R)
        density_R_original[grid_ID_ordered] = density_R
        density_R = density_R_original.copy()

    J = None

    if (use_mpi and rank == 0) or (not use_mpi):

        fn_J = getattr(libisdf, "_construct_J", None)
        assert fn_J is not None

        J = np.zeros_like(density_R)

        if short_range:
            coulG = mydf.coulG_SR
        else:
            coulG = mydf.coulG

        fn_J(
            mesh.ctypes.data_as(ctypes.c_void_p),
            density_R.ctypes.data_as(ctypes.c_void_p),
            coulG.ctypes.data_as(ctypes.c_void_p),
            J.ctypes.data_as(ctypes.c_void_p),
        )

        J_ordered = np.zeros_like(J)
        J_ordered = J[grid_ID_ordered]
        J = J_ordered.copy()

    if use_mpi:
        J = bcast(J, root=0)

    t2 = (logger.process_clock(), logger.perf_counter())

    del dm_buf, ddot_buf, density_R
    del density_R_tmp

    _benchmark_time(t1, t2, "half_J", mydf)

    return J


def _contract_j_dm_ls(
    mydf, dm, use_mpi=False, first_pass=None, second_pass=None, short_range=False
):

    if use_mpi:
        # assert mydf.direct == True
        if not mydf.direct:
            raise ValueError("MPI is only supported for direct method.")
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast, reduce

    ###### Prepocess parameter for RS ######

    assert first_pass in [None, "only_dd", "only_cc", "exclude_cc", "all"]
    assert second_pass in [None, "only_dd", "only_cc", "exclude_cc", "all"]

    if short_range:
        assert first_pass == "only_dd"
        assert second_pass == "only_dd"

    if first_pass is None:
        first_pass = "all"
    if second_pass is None:
        second_pass = "all"

    second_pass_all = second_pass == "all"
    second_pass_has_dd = second_pass in ["all", "only_dd", "exclude_cc"]
    second_pass_has_cc = second_pass in ["all", "only_cc"]
    second_pass_has_cd = second_pass in ["all", "exclude_cc"]

    ####### judge whether to call the original one #######

    if isinstance(mydf.aoRg, np.ndarray):
        if hasattr(mydf, "aoR") and mydf.aoR is not None:
            assert isinstance(mydf.aoR, np.ndarray)
        else:
            raise NotImplementedError
        ### call the original get_j ###
        from isdf_jk import _contract_j_dm_fast

        return _contract_j_dm_fast(mydf, dm, use_mpi=use_mpi)

    ####### Start the calculation ########

    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    nao = dm.shape[0]
    cell = mydf.cell
    assert cell.nao == nao
    vol = cell.vol
    mesh = np.array(cell.mesh, dtype=np.int32)
    ngrid = np.prod(mesh)

    aoR = mydf.aoR
    assert isinstance(aoR, list)

    #### step 0. allocate buffer

    max_nao_involved = np.max(
        [aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None]
    )
    max_ngrid_involved = np.max(
        [aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None]
    )

    # max_dim_buf = max(max_ngrid_involved, max_nao_involved)
    max_dim_buf = max_nao_involved
    ddot_buf = np.zeros((max_dim_buf, max_dim_buf), dtype=np.float64)
    aoR_buf1 = np.zeros((max_nao_involved, max_ngrid_involved), dtype=np.float64)

    ##### get the involved C function #####

    fn_packadd_dm = getattr(libisdf, "_packadd_local_dm", None)
    assert fn_packadd_dm is not None

    fn_packadd_dm2 = getattr(libisdf, "_packadd_local_RS", None)
    assert fn_packadd_dm2 is not None

    #### step 1 2. get density value on real space grid and IPs

    J = _half_J(mydf, dm, use_mpi, first_pass, short_range)

    #### step 3. get J

    J_Res = np.zeros((nao, nao), dtype=np.float64)

    ordered_ao_ind = np.arange(nao)

    def _get_j_pass2_ls(
        _aoR_bra,
        _ao_involved_bra,
        _aoR_ket,
        _ao_involved_ket,
        _bra_type,
        _ket_type,
        _potential,
        _Res,
        aoR_buf1=aoR_buf1,
        ddot_buf=ddot_buf,
    ):

        nao_bra = _aoR_bra.shape[0]
        nao_ket = _aoR_ket.shape[0]

        if _bra_type == _ket_type:

            aoR_J_res = np.ndarray(_aoR_ket.shape, buffer=aoR_buf1)
            lib_isdf.d_ij_j_ij(_aoR_ket, _potential, out=aoR_J_res)
            ddot_res = np.ndarray((nao_ket, nao_ket), buffer=ddot_buf)
            lib.ddot(_aoR_ket, aoR_J_res.T, c=ddot_res)

            if nao_ket == nao and np.allclose(_ao_involved_ket, ordered_ao_ind):
                _Res += ddot_res
            else:
                fn_packadd_dm(
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_ket),
                    _ao_involved_ket.ctypes.data_as(ctypes.c_void_p),
                    _Res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(_Res.shape[0]),
                )
        else:

            ### J_Res = ddot_res + ddot_res.T

            aoR_J_res = np.ndarray(_aoR_ket.shape, buffer=aoR_buf1)
            lib_isdf.d_ij_j_ij(_aoR_ket, _potential, out=aoR_J_res)
            ddot_res = np.ndarray((nao_bra, nao_ket), buffer=ddot_buf)
            lib.ddot(_aoR_bra, aoR_J_res.T, c=ddot_res)

            fn_packadd_dm2(
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_bra),
                _ao_involved_bra.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nao_ket),
                _ao_involved_ket.ctypes.data_as(ctypes.c_void_p),
                _Res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(_Res.shape[0]),
            )

    for atm_id, aoR_holder in enumerate(aoR):

        if aoR_holder is None:
            continue

        if use_mpi:
            if atm_id % comm_size != rank:
                continue

        ngrids_now = aoR_holder.aoR.shape[1]
        nao_compact = aoR_holder.nCompact

        global_gridID_begin = aoR_holder.global_gridID_begin

        J_tmp = J[global_gridID_begin : global_gridID_begin + ngrids_now]

        if second_pass_all:  ### with RS case ###

            _get_j_pass2_ls(
                aoR_holder.aoR,
                aoR_holder.ao_involved,
                aoR_holder.aoR,
                aoR_holder.ao_involved,
                "all",
                "all",
                J_tmp,
                J_Res,
            )

        else:

            if second_pass_has_cc:
                _get_j_pass2_ls(
                    aoR_holder.aoR[:nao_compact, :],
                    aoR_holder.ao_involved[:nao_compact],
                    aoR_holder.aoR[:nao_compact, :],
                    aoR_holder.ao_involved[:nao_compact],
                    "compact",
                    "compact",
                    J_tmp,
                    J_Res,
                )

            if second_pass_has_dd:
                _get_j_pass2_ls(
                    aoR_holder.aoR[nao_compact:, :],
                    aoR_holder.ao_involved[nao_compact:],
                    aoR_holder.aoR[nao_compact:, :],
                    aoR_holder.ao_involved[nao_compact:],
                    "diffuse",
                    "diffuse",
                    J_tmp,
                    J_Res,
                )

            if second_pass_has_cd:
                _get_j_pass2_ls(
                    aoR_holder.aoR[:nao_compact, :],
                    aoR_holder.ao_involved[:nao_compact],
                    aoR_holder.aoR[nao_compact:, :],
                    aoR_holder.ao_involved[nao_compact:],
                    "compact",
                    "diffuse",
                    J_tmp,
                    J_Res,
                )

    J = J_Res

    if use_mpi:
        J = reduce(J, root=0)

    t2 = (logger.process_clock(), logger.perf_counter())

    _benchmark_time(t1, t2, "_contract_j_dm_fast", mydf)

    ######### delete the buffer #########

    del ddot_buf
    del aoR_buf1

    return J * ngrid / vol


############# quadratic scaling (not cubic!) #############


def __get_DensityMatrixonRgAO_qradratic(
    mydf, dm, bra_aoR_holder, bra_type=None, _res: np.ndarray = None, verbose=1
):

    assert bra_type in [None, "all", "compact", "diffuse"]

    # t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] <= 4
        # dm = dm[0]
    else:
        dm = dm.reshape(1, *dm.shape)

    assert dm.shape[1] == dm.shape[2]
    nset, nao = dm.shape[0], dm.shape[1]

    ngrid_bra = np.sum(
        [
            aoR_holder.aoR.shape[1]
            for aoR_holder in bra_aoR_holder
            if aoR_holder is not None
        ]
    )

    max_ngrid_bra = np.max(
        [
            aoR_holder.aoR.shape[1]
            for aoR_holder in bra_aoR_holder
            if aoR_holder is not None
        ]
    )
    # max_ao_involved = np.max(
    #     [
    #         aoR_holder.aoR.shape[0]
    #         for aoR_holder in bra_aoR_holder
    #         if aoR_holder is not None
    #     ]
    # )

    if _res is None:
        res = np.zeros((nset, ngrid_bra, nao), dtype=np.float64)
    else:
        res = np.ndarray((nset, ngrid_bra, nao), buffer=_res, dtype=np.float64)

    ### allocate buf ###

    offset = 0
    ddot_buf = np.ndarray((max_ngrid_bra, nao), buffer=mydf.build_k_buf, offset=offset)
    offset += ddot_buf.size * ddot_buf.dtype.itemsize
    dm_pack_buf = np.ndarray(
        (dm.shape[1], dm.shape[2]), buffer=mydf.build_k_buf, offset=offset
    )

    ### get pack fn ###

    fn_packrow = getattr(libisdf, "_buildK_packrow", None)
    assert fn_packrow is not None
    fn_packcol = getattr(libisdf, "_buildK_packcol", None)
    assert fn_packcol is not None

    ### perform aoR_bra.T * dm

    ordered_ao_ind = np.arange(nao)
    grid_shift = None
    # ngrid_loc = 0

    for aoR_holder in bra_aoR_holder:

        if aoR_holder is None:
            continue

        ngrid_now = aoR_holder.aoR.shape[1]
        nao_involved = aoR_holder.aoR.shape[0]
        nao_compact = aoR_holder.nCompact

        ao_begin_indx = 0
        ao_end_indx = nao_involved
        if bra_type == "compact":
            ao_end_indx = nao_compact
        elif bra_type == "diffuse":
            ao_begin_indx = nao_compact

        nao_at_work = ao_end_indx - ao_begin_indx

        for iset in range(nset):
            if (nao_at_work) == nao and np.allclose(
                aoR_holder.ao_involved, ordered_ao_ind
            ):
                dm_packed = dm[iset]
            else:
                dm_packed = np.ndarray((nao_at_work, nao), buffer=dm_pack_buf)
                fn_packrow(
                    dm_packed.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao_at_work),
                    ctypes.c_int(nao),
                    dm[iset].ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(nao),
                    ctypes.c_int(nao),
                    aoR_holder.ao_involved[ao_begin_indx:ao_end_indx].ctypes.data_as(
                        ctypes.c_void_p
                    ),
                )

            ddot_res = np.ndarray((ngrid_now, nao), buffer=ddot_buf)
            lib.ddot(
                aoR_holder.aoR[ao_begin_indx:ao_end_indx, :].T, dm_packed, c=ddot_res
            )
            grid_loc_begin = aoR_holder.global_gridID_begin

            if grid_shift is None:
                grid_shift = grid_loc_begin
            else:
                assert grid_loc_begin >= grid_shift

            res[
                iset,
                grid_loc_begin - grid_shift : grid_loc_begin - grid_shift + ngrid_now,
                :,
            ] = ddot_res

    # t2 = (logger.process_clock(), logger.perf_counter())
    return res


def _contract_k_dm_quadratic(mydf, dm, with_robust_fitting=True, use_mpi=False):

    if use_mpi:
        raise NotImplementedError("MPI is not supported yet.")

    ####### judge whether to call the original one #######

    if isinstance(mydf.aoRg, np.ndarray):
        from isdf_jk import _contract_k_dm, _contract_k_dm_wo_robust_fitting

        if mydf.aoR is None:
            return _contract_k_dm_wo_robust_fitting(mydf, dm, False, use_mpi=use_mpi)
        else:
            return _contract_k_dm(mydf, dm, with_robust_fitting, use_mpi=use_mpi)

    ######## start the calculation ########

    t1 = (logger.process_clock(), logger.perf_counter())

    if len(dm.shape) == 3:
        assert dm.shape[0] == 1
        dm = dm[0]

    nao = dm.shape[0]
    cell = mydf.cell
    assert cell.nao == nao
    vol = cell.vol
    mesh = np.array(cell.mesh, dtype=np.int32)
    ngrid = np.prod(mesh)

    aoRg = mydf.aoRg
    assert isinstance(aoRg, list)
    aoR = mydf.aoR
    assert isinstance(aoR, list)

    naux = mydf.naux
    nao = cell.nao

    #### step 0. allocate buffer

    max_nao_involved = np.max(
        [aoR_holder.aoR.shape[0] for aoR_holder in aoR if aoR_holder is not None]
    )
    max_ngrid_involved = np.max(
        [aoR_holder.aoR.shape[1] for aoR_holder in aoR if aoR_holder is not None]
    )
    max_nIP_involved = np.max(
        [aoR_holder.aoR.shape[1] for aoR_holder in aoRg if aoR_holder is not None]
    )

    mydf.allocate_k_buffer()

    # ddot_res_buf = np.zeros((naux, max_nao_involved), dtype=np.float64)
    ddot_res_buf = mydf.build_k_buf

    ##### get the involved C function #####

    fn_packadd_row = getattr(libisdf, "_buildK_packaddrow", None)
    assert fn_packadd_row is not None
    fn_packadd_col = getattr(libisdf, "_buildK_packaddcol", None)
    assert fn_packadd_col is not None

    fn_packcol1 = getattr(libisdf, "_buildK_packcol", None)
    fn_packcol2 = getattr(libisdf, "_buildK_packcol2", None)
    assert fn_packcol1 is not None
    assert fn_packcol2 is not None

    #### step 1. get density matrix value on real space grid and IPs

    Density_RgAO = __get_DensityMatrixonRgAO_qradratic(
        mydf, dm, aoRg, "all", mydf.Density_RgAO_buf
    )
    Density_RgAO = Density_RgAO[0]

    #### step 2. get K, those part which W is involved

    W = mydf.W
    assert W is not None
    assert isinstance(W, np.ndarray)

    K1 = np.zeros((naux, nao), dtype=np.float64)

    ####### buf for the first loop #######

    offset = 0
    ddot_buf1 = np.ndarray(
        (naux, max_nIP_involved), buffer=ddot_res_buf, offset=offset, dtype=np.float64
    )
    offset = ddot_buf1.size * ddot_res_buf.dtype.itemsize
    pack_buf = np.ndarray(
        (naux, max_nao_involved), buffer=ddot_res_buf, offset=offset, dtype=np.float64
    )
    offset += pack_buf.size * pack_buf.dtype.itemsize
    ddot_buf2 = np.ndarray(
        (naux, max(max_nIP_involved, max_nao_involved)),
        buffer=ddot_res_buf,
        offset=offset,
        dtype=np.float64,
    )

    ordered_ao_ind = np.arange(nao)

    ### TODO: consider MPI

    nIP_loc = 0
    for aoRg_holder in aoRg:

        if aoRg_holder is None:
            continue

        nIP_now = aoRg_holder.aoR.shape[1]
        nao_involved = aoRg_holder.aoR.shape[0]

        #### pack the density matrix ####

        if nao_involved == nao and np.allclose(aoRg_holder.ao_involved, ordered_ao_ind):
            Density_RgAO_packed = Density_RgAO
        else:
            # Density_RgAO_packed = Density_RgAO[:, aoRg_holder.ao_involved]
            Density_RgAO_packed = np.ndarray((naux, nao_involved), buffer=pack_buf)
            fn_packcol1(
                Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(nao_involved),
                Density_RgAO.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(nao),
                aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p),
            )

        # W_tmp = Density_RgRg[:, nIP_loc:nIP_loc+nIP_now] * W[:, nIP_loc:nIP_loc+nIP_now]

        ddot_res1 = np.ndarray((naux, nIP_now), buffer=ddot_buf1)
        lib.ddot(Density_RgAO_packed, aoRg_holder.aoR, c=ddot_res1)
        Density_RgRg = ddot_res1
        W_packed = np.ndarray((naux, nIP_now), buffer=ddot_buf2)
        fn_packcol2(
            W_packed.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(naux),
            ctypes.c_int(nIP_now),
            W.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(naux),
            ctypes.c_int(naux),
            ctypes.c_int(nIP_loc),
            ctypes.c_int(nIP_loc + nIP_now),
        )
        lib_isdf.cwise_mul(W_packed, Density_RgRg, out=Density_RgRg)
        W_tmp = Density_RgRg

        # ddot

        ddot_res = np.ndarray((naux, nao_involved), buffer=ddot_buf2)
        lib.ddot(W_tmp, aoRg_holder.aoR.T, c=ddot_res)

        if nao_involved == nao and np.allclose(aoRg_holder.ao_involved, ordered_ao_ind):
            K1 += ddot_res
        else:
            # K1[: , aoRg_holder.ao_involved] += ddot_res
            fn_packadd_col(
                K1.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(K1.shape[0]),
                ctypes.c_int(K1.shape[1]),
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ddot_res.shape[0]),
                ctypes.c_int(ddot_res.shape[1]),
                aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p),
            )

        nIP_loc += nIP_now
    # del W_tmp
    assert nIP_loc == naux

    K = np.zeros((nao, nao), dtype=np.float64)

    nIP_loc = 0
    for aoRg_holder in aoRg:

        if aoRg_holder is None:
            continue

        nIP_now = aoRg_holder.aoR.shape[1]
        nao_involved = aoRg_holder.aoR.shape[0]

        K_tmp = K1[nIP_loc : nIP_loc + nIP_now, :]

        ddot_res = np.ndarray((nao_involved, nao), buffer=ddot_res_buf)
        lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)

        if nao_involved == nao and np.allclose(aoRg_holder.ao_involved, ordered_ao_ind):
            K += ddot_res
        else:
            # K[aoRg_holder.ao_involved, :] += ddot_res
            fn_packadd_row(
                K.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(K.shape[0]),
                ctypes.c_int(K.shape[1]),
                ddot_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(ddot_res.shape[0]),
                ctypes.c_int(ddot_res.shape[1]),
                aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p),
            )

        nIP_loc += nIP_now
    # del K_tmp
    assert nIP_loc == naux

    #### step 3. get K, those part which W is not involved, with robust fitting

    if with_robust_fitting:

        K = -K

        ### calcualte those parts where V is involved

        V_R = mydf.V_R
        assert V_R is not None
        assert isinstance(V_R, np.ndarray)

        # lib_isdf.cwise_mul(V_R, Density_RgR, out=Density_RgR)

        K2 = K1
        K2.ravel()[:] = 0.0

        # fn_packcol = getattr(libisdf, "_buildK_packcol2", None)
        # assert fn_packcol is not None

        ddot_buf1 = np.ndarray((naux, max_nao_involved), buffer=ddot_res_buf)
        offset = naux * max_nao_involved * ddot_res_buf.dtype.itemsize
        V_tmp_buf = np.ndarray(
            (naux, max_ngrid_involved), buffer=ddot_res_buf, offset=offset
        )
        offset += V_tmp_buf.size * V_tmp_buf.dtype.itemsize
        pack_buf = np.ndarray(
            (naux, max_nao_involved), buffer=ddot_res_buf, offset=offset
        )
        offset += pack_buf.size * pack_buf.dtype.itemsize
        ddot_buf2 = np.ndarray(
            (naux, max_ngrid_involved), buffer=ddot_res_buf, offset=offset
        )

        ngrid_loc = 0

        for aoR_holder in aoR:

            if aoR_holder is None:
                continue

            ngrid_now = aoR_holder.aoR.shape[1]
            nao_involved = aoR_holder.aoR.shape[0]

            #### pack the density matrix ####

            if nao_involved == nao and np.allclose(
                aoR_holder.ao_involved, ordered_ao_ind
            ):
                Density_RgAO_packed = Density_RgAO
            else:
                # Density_RgAO_packed = Density_RgAO[:, aoR_holder.ao_involved]
                Density_RgAO_packed = np.ndarray((naux, nao_involved), buffer=pack_buf)
                fn_packcol1(
                    Density_RgAO_packed.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux),
                    ctypes.c_int(nao_involved),
                    Density_RgAO.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux),
                    ctypes.c_int(nao),
                    aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p),
                )

            # V_tmp = Density_RgR[:, ngrid_loc:ngrid_loc+ngrid_now] * V_R[:, ngrid_loc:ngrid_loc+ngrid_now]

            ddot_res2 = np.ndarray((naux, ngrid_now), buffer=ddot_buf2)
            lib.ddot(Density_RgAO_packed, aoR_holder.aoR, c=ddot_res2)
            Density_RgR = ddot_res2
            V_packed = np.ndarray((naux, ngrid_now), buffer=V_tmp_buf)
            fn_packcol2(
                V_packed.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(ngrid_now),
                V_R.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(naux),
                ctypes.c_int(ngrid),
                ctypes.c_int(ngrid_loc),
                ctypes.c_int(ngrid_loc + ngrid_now),
            )
            lib_isdf.cwise_mul(V_packed, Density_RgR, out=Density_RgR)
            V_tmp = Density_RgR

            ddot_res = np.ndarray((naux, nao_involved), buffer=ddot_buf1)
            lib.ddot(V_tmp, aoR_holder.aoR.T, c=ddot_res)

            if nao_involved == nao and np.allclose(
                aoR_holder.ao_involved, ordered_ao_ind
            ):
                K2 += ddot_res
            else:
                # K2[: , aoR_holder.ao_involved] += ddot_res
                fn_packadd_col(
                    K2.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(K2.shape[0]),
                    ctypes.c_int(K2.shape[1]),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ddot_res.shape[0]),
                    ctypes.c_int(ddot_res.shape[1]),
                    aoR_holder.ao_involved.ctypes.data_as(ctypes.c_void_p),
                )

            ngrid_loc += ngrid_now
        # del V_tmp

        assert ngrid_loc == ngrid

        K_add = np.zeros((nao, nao), dtype=np.float64)

        nIP_loc = 0
        for aoRg_holder in aoRg:

            if aoRg_holder is None:
                continue

            nIP_now = aoRg_holder.aoR.shape[1]
            nao_involved = aoRg_holder.aoR.shape[0]

            K_tmp = K2[
                nIP_loc : nIP_loc + nIP_now, :
            ]  # no need to pack, continguous anyway

            ddot_res = np.ndarray((nao_involved, nao), buffer=ddot_res_buf)
            lib.ddot(aoRg_holder.aoR, K_tmp, c=ddot_res)

            if nao == nao_involved and np.allclose(
                aoRg_holder.ao_involved, ordered_ao_ind
            ):
                K_add += ddot_res
            else:
                # K_add[aoRg_holder.ao_involved, :] += ddot_res
                fn_packadd_row(
                    K_add.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(K_add.shape[0]),
                    ctypes.c_int(K_add.shape[1]),
                    ddot_res.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(ddot_res.shape[0]),
                    ctypes.c_int(ddot_res.shape[1]),
                    aoRg_holder.ao_involved.ctypes.data_as(ctypes.c_void_p),
                )

            nIP_loc += nIP_now
        # del K_tmp
        assert nIP_loc == naux

        K_add += K_add.T

        K += K_add

    ######### finally delete the buffer #########

    del K1

    t2 = (logger.process_clock(), logger.perf_counter())

    # if mydf.verbose:
    _benchmark_time(t1, t2, "_contract_k_dm_quadratic", mydf)

    return K * ngrid / vol


def _contract_k_dm_quadratic_direct(mydf, dm, use_mpi=False):

    if use_mpi:
        # assert mydf.direct == True
        if not mydf.direct:
            raise ValueError("MPI is only supported for direct method.")
        from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, bcast, reduce

    t1 = (logger.process_clock(), logger.perf_counter())

    if dm.ndim == 3:
        assert dm.shape[0] <= 4
        # dm = dm[0]
    else:
        dm = dm.reshape(1, *dm.shape)

    aoR = mydf.aoR
    aoRg = mydf.aoRg

    # max_nao_involved = mydf.max_nao_involved
    max_ngrid_involved = mydf.max_ngrid_involved
    # max_nIP_involved = mydf.max_nIP_involved
    maxsize_group_naux = mydf.maxsize_group_naux

    ####### preparing the data #######

    nset, nao = dm.shape[0], dm.shape[1]
    cell = mydf.cell
    assert cell.nao == nao
    vol = cell.vol
    mesh = np.array(cell.mesh, dtype=np.int32)
    # mesh_int32 = mesh
    ngrid = np.prod(mesh)

    aoRg = mydf.aoRg
    assert isinstance(aoRg, list)
    aoR = mydf.aoR
    assert isinstance(aoR, list)

    # naux = mydf.naux
    nao = cell.nao
    aux_basis = mydf.aux_basis

    # grid_ordering = mydf.grid_ID_ordered

    if not hasattr(mydf, "coulG"):
        if mydf.omega is not None:
            assert mydf.omega >= 0.0
        # mydf.coulG = tools.get_coulG(cell, mesh=mesh, omega=mydf.omega)
        raise NotImplementedError("coulG is not implemented yet.")

    coulG = mydf.coulG
    coulG_real = coulG.reshape(*mesh)[:, :, : mesh[2] // 2 + 1].reshape(-1).copy()

    mydf.allocate_k_buffer(nset)
    build_k_buf = mydf.build_k_buf
    build_VW_buf = mydf.build_VW_in_k_buf

    group = mydf.group
    assert len(group) == len(aux_basis)

    ######### allocate buffer #########

    Density_RgAO_buf = mydf.Density_RgAO_buf

    nThread = lib.num_threads()
    # bufsize_per_thread = coulG_real.shape[0] * 2 + np.prod(mesh)
    # buf_build_V = np.ndarray(
    #     (nThread, bufsize_per_thread), dtype=np.float64, buffer=build_VW_buf
    # )
    # offset_now = buf_build_V.size * buf_build_V.dtype.itemsize

    bufsize_per_thread = coulG_real.shape[0] * 2 + mesh[0] * mesh[1] * (mesh[2] + 1)
    buf_build_V = np.ndarray(
        (nThread, bufsize_per_thread + EXTRA_ALLOC), dtype=np.float64
    )
    offset_now = 0

    build_K_bunchsize = min(maxsize_group_naux, mydf._build_K_bunchsize)

    offset_build_now = 0
    # offset_Density_RgR_buf = 0
    Density_RgR_buf = np.ndarray(
        (build_K_bunchsize, ngrid), buffer=build_k_buf, offset=offset_build_now
    )

    offset_build_now += Density_RgR_buf.size * Density_RgR_buf.dtype.itemsize
    offset_ddot_res_RgR_buf = offset_build_now
    ddot_res_RgR_buf = np.ndarray(
        (build_K_bunchsize, max_ngrid_involved),
        buffer=build_k_buf,
        offset=offset_ddot_res_RgR_buf,
    )

    offset_build_now += ddot_res_RgR_buf.size * ddot_res_RgR_buf.dtype.itemsize
    offset_K1_tmp1_buf = offset_build_now
    K1_tmp1_buf = np.ndarray(
        (maxsize_group_naux, nao), buffer=build_k_buf, offset=offset_K1_tmp1_buf
    )

    offset_build_now += K1_tmp1_buf.size * K1_tmp1_buf.dtype.itemsize
    offset_K1_tmp1_ddot_res_buf = offset_build_now
    K1_tmp1_ddot_res_buf = np.ndarray(
        (maxsize_group_naux, nao),
        buffer=build_k_buf,
        offset=offset_K1_tmp1_ddot_res_buf,
    )

    offset_build_now += K1_tmp1_ddot_res_buf.size * K1_tmp1_ddot_res_buf.dtype.itemsize

    offset_K1_final_ddot_buf = offset_build_now
    K1_final_ddot_buf = np.ndarray(
        (nao, nao), buffer=build_k_buf, offset=offset_K1_final_ddot_buf
    )

    ########### get involved C function ###########

    fn_packcol1 = getattr(libisdf, "_buildK_packcol", None)
    assert fn_packcol1 is not None
    fn_packcol2 = getattr(libisdf, "_buildK_packcol2", None)
    assert fn_packcol2 is not None
    fn_packadd_col = getattr(libisdf, "_buildK_packaddcol", None)
    assert fn_packadd_col is not None
    fn_packadd_row = getattr(libisdf, "_buildK_packaddrow", None)
    assert fn_packadd_row is not None

    # ordered_ao_ind = np.arange(nao)

    ######### begin work #########

    K1 = np.zeros((nset, nao, nao), dtype=np.float64)  # contribution from V matrix
    K2 = np.zeros((nset, nao, nao), dtype=np.float64)  # contribution from W matrix

    for group_id, atm_ids in enumerate(group):

        if use_mpi:
            if group_id % comm_size != rank:
                continue

        naux_tmp = 0
        aoRg_holders = []
        for atm_id in atm_ids:
            naux_tmp += aoRg[atm_id].aoR.shape[1]
            aoRg_holders.append(aoRg[atm_id])
        assert naux_tmp == aux_basis[group_id].shape[0]

        # aux_basis_tmp = aux_basis[group_id]

        #### 1. build the involved DM_RgR ####

        Density_RgAO_tmp = np.ndarray((nset, naux_tmp, nao), buffer=Density_RgAO_buf)
        offset_density_RgAO_buf = (
            Density_RgAO_tmp.size * Density_RgAO_buf.dtype.itemsize
        )
        Density_RgAO_tmp.ravel()[:] = 0.0
        Density_RgAO_tmp = __get_DensityMatrixonRgAO_qradratic(
            mydf, dm, aoRg_holders, "all", Density_RgAO_tmp, verbose=mydf.verbose
        )

        #### 2. build the V matrix ####

        W_tmp = None

        for iset in range(nset):

            calculate_W_tmp = iset == 0

            _W_tmp = _isdf_get_K_direct_kernel_1(
                mydf,
                coulG_real,
                group_id,
                Density_RgAO_tmp[iset],
                None,
                True,
                calculate_W_tmp,
                ##### buffer #####
                buf_build_V,
                build_VW_buf,
                offset_now,
                Density_RgR_buf,
                Density_RgAO_buf,
                offset_density_RgAO_buf,
                ddot_res_RgR_buf,
                K1_tmp1_buf,
                K1_tmp1_ddot_res_buf,
                K1_final_ddot_buf,
                ##### bunchsize #####
                build_K_bunchsize,
                ##### other info #####
                use_mpi=use_mpi,
                ##### out #####
                K1_or_2=K1[iset],
            )

            if calculate_W_tmp:
                W_tmp = _W_tmp.copy()

            _isdf_get_K_direct_kernel_1(
                mydf,
                coulG_real,
                group_id,
                Density_RgAO_tmp[iset],
                W_tmp,
                False,
                False,
                ##### buffer #####
                buf_build_V,
                build_VW_buf,
                offset_now,
                Density_RgR_buf,
                Density_RgAO_buf,
                offset_density_RgAO_buf,
                ddot_res_RgR_buf,
                K1_tmp1_buf,
                K1_tmp1_ddot_res_buf,
                K1_final_ddot_buf,
                ##### bunchsize #####
                build_K_bunchsize,
                ##### other info #####
                use_mpi=use_mpi,
                ##### out #####
                K1_or_2=K2[iset],
            )

    ######### finally delete the buffer #########

    if use_mpi:
        comm.Barrier()

    if use_mpi:
        K1 = reduce(K1, root=0)
        K2 = reduce(K2, root=0)
        K = np.zeros_like(K1)
        if rank == 0:
            for iset in range(nset):
                K[iset] = K1[iset] + K1[iset].T - K2[iset]
        else:
            K = None
        K = bcast(K, root=0)
    else:
        K = np.zeros_like(K1)
        for iset in range(nset):
            K[iset] = K1[iset] + K1[iset].T - K2[iset]

    del K1
    del K2

    t2 = (logger.process_clock(), logger.perf_counter())

    _benchmark_time(t1, t2, "_contract_k_dm_quadratic_direct", mydf)

    return K * ngrid / vol


def get_jk_dm_quadratic(
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

    ############ deal with occ-RI-K ############

    if getattr(dm, "__dict__", None) is not None:
        mo_coeff = dm.__dict__["mo_coeff"]
        mo_occ = dm.__dict__["mo_occ"]
        if mo_coeff is not None:
            assert mo_occ is not None
            if mo_coeff.ndim == 3:
                assert mo_coeff.shape[2] == mo_occ.shape[1]
                assert mo_occ.ndim == 2
            else:
                assert mo_coeff.shape[1] == mo_occ.shape[0]
                assert mo_coeff.ndim == 2
                assert mo_occ.ndim == 1
    else:
        dm = np.asarray(dm)
        if len(dm.shape) == 3:
            assert dm.shape[0] <= 4
        mo_occ = None
        mo_coeff = None

    if dm.ndim == 2:
        dm = dm.reshape(1, *dm.shape)

    assert dm.ndim == 3

    ############ end deal with occ-RI-K ############

    direct = mydf.direct
    use_mpi = mydf.use_mpi

    if use_mpi and not direct:
        raise NotImplementedError("ISDF does not support use_mpi and direct=False")

    if len(dm.shape) == 3:
        assert dm.shape[0] <= 4
        ## NOTE: 1 for RHF 2 for UHF 3/4 for GHF

    if hasattr(mydf, "Ls") and mydf.Ls is not None:
        from pyscf.isdf.isdf_tools_densitymatrix import symmetrize_dm

        dm = symmetrize_dm(dm, mydf.Ls)
    else:
        if hasattr(mydf, "kmesh") and mydf.kmesh is not None:
            from pyscf.isdf.isdf_tools_densitymatrix import symmetrize_dm

            dm = symmetrize_dm(dm, mydf.kmesh)

    if use_mpi:
        from pyscf.isdf.isdf_tools_mpi import rank, bcast

        dm = bcast(dm, root=0)
        if mo_coeff is not None:
            mo_coeff = bcast(mo_coeff, root=0)
        if mo_occ is not None:
            mo_occ = bcast(mo_occ, root=0)

    dm = lib.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)

    nset, nao = dm.shape[:2]

    ############ end deal with dm with tags ############

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
    k_real = gamma_point(kpt) and not np.iscomplexobj(dm)

    assert j_real
    assert k_real

    mem_now = lib.current_memory()[0]
    max_memory = max(2000, (mydf.max_memory - mem_now))

    log.debug1("max_memory = %d MB (%d in use)", max_memory, mem_now)

    ### TODO: improve the efficiency ###

    vj = np.zeros_like(dm)
    vk = np.zeros_like(dm)
    for iset in range(nset):
        if with_j and iset <= 1:
            vj[iset] = _contract_j_dm_ls(mydf, dm[iset], use_mpi)
        if with_k:
            if mydf.direct:
                if iset == 0:
                    vk = _contract_k_dm_quadratic_direct(mydf, dm, use_mpi=use_mpi)
                # vk[iset] = _contract_k_dm_quadratic_direct(mydf, dm[iset], use_mpi=use_mpi)
            else:
                vk[iset] = _contract_k_dm_quadratic(
                    mydf, dm[iset], mydf.with_robust_fitting, use_mpi=use_mpi
                )

    ##### the following code is added to deal with _ewald_exxdiv_for_G0 #####

    if not use_mpi or (use_mpi and rank == 0):

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

    if use_mpi:
        vj = bcast(vj, root=0)
        vk = bcast(vk, root=0)

    ##### end of dealing with _ewald_exxdiv_for_G0 #####

    t1 = log.timer("sr jk", *t1)

    return vj, vk


############# linear scaling implementation #############
