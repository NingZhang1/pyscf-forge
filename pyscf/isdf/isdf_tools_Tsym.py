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

import sys
import numpy
import numpy as np
from itertools import product

from pyscf.pbc.gto import Cell
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.lib.kpts import KPoints

from pyscf.isdf.BackEnd.isdf_backend import _prod as PROD
from pyscf.isdf.BackEnd.isdf_backend import _zeros as ZEROS
from pyscf.isdf.BackEnd.isdf_backend import _imag as IMAG
from pyscf.isdf.BackEnd.isdf_backend import _absolute as ABS
from pyscf.isdf.BackEnd.isdf_backend import _maximum as MAX
from pyscf.isdf.BackEnd.isdf_backend import _is_realtype as IsRealTy
from pyscf.isdf.BackEnd.isdf_backend import _is_complextype as IsComplexTy
from pyscf.isdf.BackEnd.isdf_backend import TENSORTy


def symmetrize_mat(dm: TENSORTy, Ls):
    """

    generate translation symmetrized (density) matrix (by average)

    Args :
        dm : TENSORTy, density matrix or anyother matrix, shape = (nao, nao)
        Ls : list, supercell dimension, shape = (3,), or kmesh in k-sampling

    Returns :
        dm_symm : TENSORTy, symmetrized density matrix, shape = (nao, nao)
    """

    is_single_dm = False

    if dm.ndim == 2:
        is_single_dm = True
        dm = dm.reshape(1, dm.shape[0], dm.shape[1])

    ncell = PROD(Ls)
    nao = dm.shape[1]
    nset = dm.shape[0]
    nao_prim = nao // ncell
    dm_symm = ZEROS((nset, nao, nao), dtype=dm.dtype, like=dm, cpu=None)

    def _loc(i, j, k):
        return i * Ls[1] * Ls[2] + j * Ls[2] + k

    for i, j, k in product(range(Ls[0]), range(Ls[1]), range(Ls[2])):

        dm_symmetrized_buf = ZEROS(
            (nset, nao_prim, nao_prim), dtype=dm.dtype, like=dm, cpu=None
        )

        for i_row, j_row, k_row in product(range(Ls[0]), range(Ls[1]), range(Ls[2])):

            loc_row = _loc(i_row, j_row, k_row)
            loc_col = _loc(
                (i + i_row) % Ls[0], (j + j_row) % Ls[1], (k + k_row) % Ls[2]
            )

            b_begin = loc_row * nao_prim
            b_end = (loc_row + 1) * nao_prim

            k_begin = loc_col * nao_prim
            k_end = (loc_col + 1) * nao_prim

            dm_symmetrized_buf += dm[:, b_begin:b_end, k_begin:k_end]

        dm_symmetrized_buf /= ncell

        for i_row, j_row, k_row in product(range(Ls[0]), range(Ls[1]), range(Ls[2])):

            loc_row = _loc(i_row, j_row, k_row)
            loc_col = _loc(
                (i + i_row) % Ls[0], (j + j_row) % Ls[1], (k + k_row) % Ls[2]
            )

            b_begin = loc_row * nao_prim
            b_end = (loc_row + 1) * nao_prim

            k_begin = loc_col * nao_prim
            k_end = (loc_col + 1) * nao_prim

            dm_symm[:, b_begin:b_end, k_begin:k_end] = dm_symmetrized_buf

    if is_single_dm:
        return dm_symm[0]
    else:
        return dm_symm


def pack_JK(input_mat: TENSORTy, Ls, nao_prim, output=None):
    """
    pack matrix in real space
    """

    _FloatTy = input_mat.dtype

    assert IsComplexTy(_FloatTy)
    ncell = PROD(Ls)
    assert input_mat.shape[0] == nao_prim
    assert input_mat.shape[1] == nao_prim * ncell

    if output is None:
        output = ZEROS(
            (ncell * nao_prim, ncell * nao_prim),
            dtype=_FloatTy,
            like=input_mat,
            cpu=None,
        )
    else:
        assert output.shape == (ncell * nao_prim, ncell * nao_prim)

    def _loc(i, j, k):
        return i * Ls[1] * Ls[2] + j * Ls[2] + k

    for ix_row, iy_row, iz_row in product(range(Ls[0]), range(Ls[1]), range(Ls[2])):

        loc_row = _loc(ix_row, iy_row, iz_row)

        b_begin = loc_row * nao_prim
        b_end = (loc_row + 1) * nao_prim

        for ix_col, iy_col, iz_col in product(range(Ls[0]), range(Ls[1]), range(Ls[2])):

            loc_col = _loc(ix_col, iy_col, iz_col)

            k_begin = loc_col * nao_prim
            k_end = (loc_col + 1) * nao_prim

            loc_col2 = _loc(
                (ix_col - ix_row) % Ls[0],
                (iy_col - iy_row) % Ls[1],
                (iz_col - iz_row) % Ls[2],
            )

            k_begin2 = loc_col2 * nao_prim
            k_end2 = (loc_col2 + 1) * nao_prim

            output[b_begin:b_end, k_begin:k_end] = input_mat[:, k_begin2:k_end2]

    return output


def pack_JK_in_FFT_space(input_mat: TENSORTy, kmesh, nao_prim, output=None):
    """
    pack matrix in k-space
    """

    _FloatTy = input_mat.dtype

    ncomplex = kmesh[0] * kmesh[1] * (kmesh[2] // 2 + 1)
    assert IsComplexTy(_FloatTy)
    assert input_mat.shape[0] == nao_prim
    assert input_mat.shape[1] == nao_prim * ncomplex

    nkpts = PROD(kmesh)

    if output is None:
        output = ZEROS(
            (nao_prim, nao_prim * nkpts), dtype=_FloatTy, like=input_mat, cpu=None
        )
    else:
        assert output.shape == (nao_prim, nao_prim * nkpts) or output.shape == (
            nkpts,
            nao_prim,
            nao_prim,
        )

    output = output.reshape(nkpts, nao_prim, nao_prim)

    def _loc(i, j, k):
        return i * kmesh[1] * kmesh[2] + j * kmesh[2] + k

    loc = 0

    for ix, iy, iz in product(
        range(kmesh[0]), range(kmesh[1]), range(kmesh[2] // 2 + 1)
    ):
        loc1 = _loc(ix, iy, iz)
        loc2 = _loc(
            (kmesh[0] - ix) % kmesh[0],
            (kmesh[1] - iy) % kmesh[1],
            (kmesh[2] - iz) % kmesh[2],
        )
        if loc1 == loc2:  # self dual must be real!
            output[loc1] = input_mat[:, loc * nao_prim : (loc + 1) * nao_prim]
            imag_part = IMAG(output[loc1])
            if MAX(ABS(imag_part)) > 1e-8:
                print("Warning: max abs of imag_part = ", MAX(ABS(imag_part)))
        else:
            output[loc1] = input_mat[:, loc * nao_prim : (loc + 1) * nao_prim]
            output[loc2] = input_mat[:, loc * nao_prim : (loc + 1) * nao_prim].conj()
        loc += 1

    return output


def _kmesh_to_Kpoints(cell, mesh):

    kpts = []

    for i, j, k in product(range(mesh[0]), range(mesh[1]), range(mesh[2])):
        kpts.append(
            [
                1.0 / float(mesh[0]) * float(i),
                1.0 / float(mesh[1]) * float(j),
                1.0 / float(mesh[2]) * float(k),
            ]
        )

    kpts = np.array(kpts)

    return KPoints(cell, kpts)
