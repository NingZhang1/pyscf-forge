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

import numpy, scipy

from pyscf import lib
from pyscf.df.df import DF
from pyscf.df.incore import _eig_decompose
from pyscf.isdf.isdf import ISDF
from pyscf.isdf.isdf_local import ISDF_Local
from pyscf import __config__
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

EINSUM_IK_JK_IJK = BACKEND._einsum_ik_jk_ijk
PERMUTE = BACKEND._permute
TAKE = BACKEND._take
ToTENSOR = BACKEND._toTensor
ToNUMPY = BACKEND._toNumpy

MAX_MEMORY = getattr(__config__, "df_outcore_max_memory", 2000)  # 2GB
# LINEAR_DEP_THR cannot be below 1e-7,
# see qchem default setting in https://manual.q-chem.com/5.4/sec_Basis_Customization.html
LINEAR_DEP_THR = getattr(__config__, "df_df_DF_lindep", 1e-7)


def aux_e2(mydf):
    """3-center AO integrals (ij|L), where L is the auxiliary basis (IPs in ISDF)."""

    tmp = EINSUM_IK_JK_IJK(mydf._aoRg, mydf._aoRg)
    nmo = mydf._aoRg.shape[0]
    tmp = tmp.reshape(nmo * nmo, -1)
    row_indices, col_indices = numpy.tril_indices(nmo)
    row_indices = ToTENSOR(numpy.array(row_indices, dtype=numpy.int64))
    col_indices = ToTENSOR(numpy.array(col_indices, dtype=numpy.int64))
    indices = row_indices * nmo + col_indices
    res = TAKE(tmp, indices, 0)

    return res


def aux_e1(mydf):
    """3-center 2-electron AO integrals (L|ij), where L is the auxiliary basis.

    Note aux_e1 is basically analogous to aux_e2 function. It can be viewed as
    the version of transposed aux_e2 tensor:
    """

    out = aux_e2(mydf)
    return out.T


def fill_2c2e(mydf):
    assert not isinstance(mydf._isdf_obj.W, str)  # do not support outcore
    return mydf._isdf_obj.W


def cholesky_eri(mydf, lindep=LINEAR_DEP_THR, verbose=0, fauxe2=aux_e2):

    j2c = fill_2c2e(mydf)
    naux = j2c.shape[0]

    j3c = fauxe2(mydf)

    j2c = ToNUMPY(j2c)
    j3c = ToNUMPY(j3c)

    j3c = j3c.reshape(-1, naux)

    # print(j2c.shape)
    # print(j2c.shape)

    try:
        low = scipy.linalg.cholesky(j2c, lower=True)
        j2c = None
        # cderi = scipy.linalg.solve_triangular(low, j3c.T, lower=True, overwrite_b=True)
        cderi = lib.dot(low.T, j3c.T)
    except scipy.linalg.LinAlgError:
        w, v = scipy.linalg.eigh(j2c)
        idx = w > LINEAR_DEP_THR
        v = v[:, idx] * numpy.sqrt(w[idx])
        cderi = lib.dot(v.conj().T, j3c.T)

    j3c = None

    return cderi
