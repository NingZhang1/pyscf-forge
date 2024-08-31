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
from functools import partial

from pyscf.pbc.gto import Cell
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.lib.kpts import KPoints

from pyscf.isdf.BackEnd.isdf_backend import _prod as PROD
from pyscf.isdf.BackEnd.isdf_backend import _zeros as ZEROS
from pyscf.isdf.BackEnd.isdf_backend import _real as REAL
from pyscf.isdf.BackEnd.isdf_backend import _imag as IMAG
from pyscf.isdf.BackEnd.isdf_backend import _absolute as ABS
from pyscf.isdf.BackEnd.isdf_backend import _maximum as MAX
from pyscf.isdf.BackEnd.isdf_backend import _is_realtype as IsRealTy
from pyscf.isdf.BackEnd.isdf_backend import _is_complextype as IsComplexTy
from pyscf.isdf.BackEnd.isdf_backend import TENSORTy
from pyscf.isdf.BackEnd.isdf_backend import _toTensor as ToTensor
from pyscf.isdf.BackEnd.isdf_backend import _toNumpy as ToNUMPY
from pyscf.isdf.BackEnd.isdf_backend import _ifftn as IFFTN
from pyscf.isdf.BackEnd.isdf_backend import _fftn as FFTN
from pyscf.isdf.BackEnd.isdf_backend import _permute as PERMUTE
from pyscf.isdf.BackEnd.isdf_backend import _conjugate_ as CONJUGATE_
from pyscf.isdf.BackEnd.isdf_backend import NUM_THREADS
from pyscf.isdf.BackEnd.isdf_backend import _cast_to_complex as CAST_TO_COMPLEX

FFTN = partial(FFTN, threads=NUM_THREADS)


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
    dm_symm = ZEROS((nset, nao, nao), dtype=dm.dtype, like=dm, cpu=True)

    def _loc(i, j, k):
        return i * Ls[1] * Ls[2] + j * Ls[2] + k

    for i, j, k in product(range(Ls[0]), range(Ls[1]), range(Ls[2])):

        dm_symmetrized_buf = ZEROS(
            (nset, nao_prim, nao_prim), dtype=dm.dtype, like=dm, cpu=True
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

    assert IsRealTy(_FloatTy)
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


def _kmesh_to_Kpoints(cell, kmesh):

    kpts = cell.make_kpts(kmesh)

    return KPoints(cell, kpts)


def _make_kpts_kmesh(cell, kpts=None, kmesh=None):
    if kpts is None and kmesh is None:
        kmesh = np.asarray([1, 1, 1])
        kpts = _kmesh_to_Kpoints(cell, kmesh)
    elif kpts is not None:

        if not isinstance(kpts, KPoints):
            kpts = KPoints(cell, kpts)
        nkpts = kpts.nkpts

        from pyscf.pbc.tools.k2gamma import kpts_to_kmesh

        kmesh = kpts_to_kmesh(cell, kpts)
        kmesh = np.asarray(kmesh)

        if PROD(kmesh) != nkpts:
            raise ValueError("The number of k-points is not consistent with kmesh")
    else:
        if not isinstance(kmesh, np.ndarray):
            kmesh = np.asarray(kmesh)
        kpts = _kmesh_to_Kpoints(cell, kmesh)

    return kpts, kmesh


def _1e_operator_gamma2k(nao, kmesh, operator_gamma: TENSORTy):

    IsNumpy = isinstance(operator_gamma, np.ndarray)

    nao_prim = nao // PROD(kmesh)
    nkpts = PROD(kmesh)

    if operator_gamma.ndim == 3:
        assert operator_gamma.shape[0] == 1
        operator_gamma = operator_gamma[0]

    if operator_gamma.shape[0] != nao_prim:
        assert operator_gamma.shape[0] == nao_prim * nkpts
        assert operator_gamma.shape[1] == nao_prim * nkpts
        operator_gamma = operator_gamma[:nao_prim, :]
    else:
        assert operator_gamma.shape[1] == nao_prim * nkpts

    operator_gamma = ToTensor(operator_gamma)
    operator_gamma = operator_gamma.reshape(nao_prim, *kmesh, nao_prim)
    operator_gamma = PERMUTE(operator_gamma, (0, 4, 1, 2, 3))

    operator_k = FFTN(
        operator_gamma, s=tuple(kmesh), axes=(2, 3, 4), overwrite_input=False
    )
    operator_k = operator_k.reshape(nao_prim, nao_prim, nkpts)
    operator_k = PERMUTE(operator_k, (2, 0, 1))
    operator_k = CONJUGATE_(operator_k)

    # print(operator_k[1])

    if IsNumpy:
        operator_k = ToNUMPY(operator_k)
    return operator_k


def _1e_operator_k2gamma(nao, kmesh, operator_k: TENSORTy):

    IsNumpy = isinstance(operator_k, np.ndarray)

    operator_k = ToTensor(operator_k)

    nao_prim = nao // PROD(kmesh)

    assert operator_k.ndim == 3

    assert operator_k.shape[0] == PROD(kmesh)
    assert operator_k.shape[1] == nao_prim
    assert operator_k.shape[2] == nao_prim

    if PROD(kmesh) == 1:
        imag = IMAG(operator_k)
        if MAX(ABS(imag)) > 1e-8:
            print("Warning: max abs of imag_part = ", MAX(ABS(imag)))
        return REAL(operator_k).reshape(nao_prim, nao_prim)
    else:
        # symmetrization #
        for ix, iy, iz in product(range(kmesh[0]), range(kmesh[1]), range(kmesh[2] // 2 + 1)):
            loc1 = ix * kmesh[1] * kmesh[2] + iy * kmesh[2] + iz
            ix2 = (kmesh[0] - ix) % kmesh[0]
            iy2 = (kmesh[1] - iy) % kmesh[1]
            iz2 = (kmesh[2] - iz) % kmesh[2]
            loc2 = ix2 * kmesh[1] * kmesh[2] + iy2 * kmesh[2] + iz2
            if loc1 == loc2:
                op_1 = operator_k[loc1]
                # op_2 = operator_k[loc2]
                imag = IMAG(op_1)
                if MAX(ABS(imag)) > 1e-8:
                    print("Warning: In _1e_operator_k2gamma max abs of imag_part = ", MAX(ABS(imag)), ix, iy,iz)
                operator_k[loc1] =  CAST_TO_COMPLEX(REAL(op_1))
            else:
                op_1 = ToNUMPY(operator_k[loc1])
                op_2 = ToNUMPY(operator_k[loc2])
                # operator_k[loc1] = CAST_TO_COMPLEX(op_1)
                # operator_k[loc2] = CAST_TO_COMPLEX(op_2)
                diff = op_1 - op_2.conj()
                if np.max(np.abs(diff)) > 1e-8:
                    print("Warning: In _1e_operator_k2gamma max abs of diff = ", np.max(np.abs(diff)), ix, iy,iz)
                op_1 = (op_1 + op_2.conj()) / 2
                operator_k[loc1] = CAST_TO_COMPLEX(ToTensor(op_1))
                operator_k[loc2] = CAST_TO_COMPLEX(ToTensor(op_1.conj()))

    op_res = ToTensor(ToNUMPY(operator_k).copy())
    op_res = CONJUGATE_(op_res)
    op_res = PERMUTE(op_res, (1, 2, 0))
    op_res = op_res.reshape(nao_prim, nao_prim, *kmesh)
    op_res = IFFTN(op_res, s=tuple(kmesh), axes=(2, 3, 4), overwrite_input=False)
    op_res = PERMUTE(op_res, (0, 2, 3, 4, 1))
    op_res = op_res.reshape(nao_prim, PROD(kmesh) * nao_prim)

    imag = IMAG(op_res)
    if MAX(ABS(imag)) > 1e-8:
        print(
            "Warning: In _1e_operator_k2gamma max abs of imag_part = ", MAX(ABS(imag))
        )
    op_res = REAL(op_res)

    res = pack_JK(op_res, kmesh, nao_prim)

    if IsNumpy:
        res = ToNUMPY(res)

    return res
