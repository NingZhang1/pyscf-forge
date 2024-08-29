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

############ global paramter ############

K_DIRECT_NAUX_BUNCHSIZE = 256

############ ISDF backend ############

import pyscf.isdf.BackEnd.isdf_backend as BACKEND

FLOAT64 = BACKEND.FLOAT64
TENSORTy = BACKEND.TENSORTy
ToNUMPY = BACKEND._toNumpy
ToTENSOR = BACKEND._toTensor
MAX = BACKEND._maximum
ABS = BACKEND._absolute
DOT = BACKEND._dot
TAKE = BACKEND._take
CLEAN = BACKEND._clean
INDEX_COPY = BACKEND._index_copy
INDEX_ADD = BACKEND._index_add
TAKE = BACKEND._take

############ ISDF utils ############

from pyscf.isdf.isdf_tools_local import (
    aoR_Holder,
)

#########################################


def _build_V_local_bas_kernel(
    aux_basis,
    # task info #
    group_id,
    p0,
    p1,
    # buffer #
    buffer,
    buffer_fft,
    ### some helper info ###
    group_grid_ID,
    grid_ordering,
    mesh,
    coulG_real,  # = coulG[:, :, mesh[2]//2+1].reshape(-1)
):
    ngrids = np.prod(mesh)
    basis_shape = (p1 - p0, ngrids)

    # unpack basis info #

    basis_buf = buffer.malloc(basis_shape, dtype=FLOAT64, name="basis_buf")
    CLEAN(basis_buf)
    INDEX_COPY(basis_buf, 1, group_grid_ID[group_id], aux_basis[group_id][p0:p1])
    basis_buf = basis_buf.reshape(p1 - p0, *mesh)

    # func #

    EINSUM_IJ_J_IJ = BACKEND._einsum_ij_j_ij

    # construct V #

    tmp = buffer_fft.rfft(basis_buf)
    tmp = tmp.reshape(p1 - p0, -1)
    EINSUM_IJ_J_IJ(tmp, coulG_real, out=tmp)
    tmp = tmp.reshape(p1 - p0, mesh[0], mesh[1], mesh[2] // 2 + 1)
    s = (p1 - p0, *mesh)
    V = buffer_fft.irfft(tmp, s=s)  # should be put on buffer_fft.real_buf
    V = V.reshape(p1 - p0, -1)

    buffer.free(count=1)

    V_reordered = buffer.malloc(basis_shape, dtype=FLOAT64, name="V_reordered")
    TAKE(V, grid_ordering, 1, out=V_reordered)

    return V_reordered


def _build_W_local_bas_kernel(V: TENSORTy, aux_basis: list[TENSORTy], W: TENSORTy):
    grid_loc = 0
    aux_ket_loc = 0
    for aux_ket in aux_basis:
        nket = aux_ket.shape[0]
        ngrids = aux_ket.shape[1]
        DOT(
            V[:, grid_loc : grid_loc + ngrids],
            aux_ket.T,
            c=W[:, aux_ket_loc : aux_ket_loc + nket],
        )
        grid_loc += ngrids
        aux_ket_loc += nket
    return W


############ build density matrix on grids ############


def _get_dm_RgR(
    dm_RgR: TENSORTy,
    # info to contract #
    dm_RgAO: TENSORTy,
    aoR_holders,
    # buffer #
    buffer,
):

    # check the input #

    assert dm_RgR.shape[0] == dm_RgAO.shape[0]

    nIP_involved = dm_RgR.shape[0]
    # ngrids_tot = dm_RgR.shape[1]
    nao = dm_RgAO.shape[1]

    for _aoR_holder in aoR_holders:

        aoR = _aoR_holder.aoR
        gridID_begin = _aoR_holder.global_gridID_begin
        ngrids_involved = aoR.shape[1]
        ao_involved = _aoR_holder.ao_involved
        nao_involved = aoR.shape[0]

        if nao_involved == nao:
            DOT(
                dm_RgAO,
                aoR,
                c=dm_RgR[:, gridID_begin : gridID_begin + ngrids_involved],
                # beta=1,
            )

        else:

            tmp = buffer.malloc((nIP_involved, nao_involved), dtype=FLOAT64, name="tmp")
            TAKE(dm_RgAO, ao_involved, 1, out=tmp)
            DOT(
                tmp,
                aoR,
                c=dm_RgR[:, gridID_begin : gridID_begin + ngrids_involved],
                # beta=1,
            )
            buffer.free(count=1)
    return dm_RgR


############ contract DM * V/W with aoR/aoRg ############


def _get_half_K(
    half_K: TENSORTy,
    # info to contract #
    dm_RgR: TENSORTy,  # density matrix on grids
    aoR_holders: list[aoR_Holder],
    # buffer #
    buffer,
):

    CLEAN(half_K)

    # check the input #

    assert half_K.shape[0] == dm_RgR.shape[0]

    nIP_involved = dm_RgR.shape[0]
    # ngrids_tot = dm_RgR.shape[1]
    nao = half_K.shape[1]

    for _aoR_holder in aoR_holders:

        aoR = _aoR_holder.aoR
        gridID_begin = _aoR_holder.global_gridID_begin
        ngrids_involved = aoR.shape[1]
        ao_involved = _aoR_holder.ao_involved
        nao_involved = aoR.shape[0]

        if nao_involved == nao:
            DOT(
                dm_RgR[:, gridID_begin : gridID_begin + ngrids_involved],
                aoR.T,
                c=half_K,
                beta=1,
            )

        else:

            tmp = buffer.malloc((nIP_involved, nao_involved), dtype=FLOAT64, name="tmp")
            DOT(
                dm_RgR[:, gridID_begin : gridID_begin + ngrids_involved],
                aoR.T,
                c=tmp,
            )
            INDEX_ADD(half_K, 1, ao_involved, tmp)
            buffer.free(count=1)

    return half_K
