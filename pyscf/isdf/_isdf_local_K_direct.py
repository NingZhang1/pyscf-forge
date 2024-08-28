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
ToNUMPY = BACKEND._toNumpy
ToTENSOR = BACKEND._toTensor
MAX = BACKEND._maximum
ABS = BACKEND._absolute
DOT = BACKEND._dot
TAKE = BACKEND._take
CLEAN = BACKEND._clean
INDEX_COPY = BACKEND._index_copy

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

    buffer.free(count=1)

    return V.reshape(p1 - p0, -1)
