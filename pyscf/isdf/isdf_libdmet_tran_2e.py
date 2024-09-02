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

import numpy, scipy
import numpy as np
import ctypes

############ pyscf module ############

from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact

############ isdf backends ############

import pyscf.isdf.BackEnd.isdf_backend as BACKEND
from pyscf.isdf.BackEnd.isdf_memory_allocator import SimpleMemoryAllocator

USE_GPU = BACKEND.USE_GPU

NUM_THREADS = BACKEND.NUM_THREADS
FLOAT64 = BACKEND.FLOAT64
TENSORTy = BACKEND.TENSORTy
ToNUMPY = BACKEND._toNumpy
ToTENSOR = BACKEND._toTensor
MALLOC = BACKEND._malloc
ZEROS = BACKEND._zeros
MAX = BACKEND._maximum
ABS = BACKEND._absolute
DOT = BACKEND._dot
TAKE = BACKEND._take
EINSUM_IK_JK_IJK = BACKEND._einsum_ik_jk_ijk
CLEAN = BACKEND._clean
ADD_TRANSPOSE_ = BACKEND._add_transpose_
INDEX_ADD = BACKEND._index_add

############ isdf utils ############

import pyscf.isdf.misc as misc
from pyscf.isdf.isdf_ao2mo import _get_moR
from pyscf.isdf.isdf_tools_local import (
    _range_partition,
)
from pyscf.isdf._isdf_local_K_kernel import (
    _build_V_local_bas_kernel,
    _build_W_local_bas_k_kernel,
)

from pyscf.isdf.isdf_ao2mo import AOPAIR_BLKSIZE


def eri_restore(eri, symmetry, nemb):
    """
    Restore eri with given permutation symmetry.
    """

    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

    spin_pair = eri.shape[0]
    if spin_pair == 1:
        eri_res = ao2mo.restore(symmetry, eri[0].real, nemb)
    else:
        if symmetry == 4:
            nemb_pair = nemb * (nemb + 1) // 2
            if eri.size == spin_pair * nemb_pair * nemb_pair:
                eri_res = eri.real.reshape(spin_pair, nemb_pair, nemb_pair)
            else:
                eri_res = np.empty((spin_pair, nemb_pair, nemb_pair))
                for s in range(spin_pair):
                    eri_res[s] = ao2mo.restore(symmetry, eri[s].real, nemb)
        elif symmetry == 1:
            if eri.size == spin_pair * nemb**4:
                eri_res = eri.real.reshape(spin_pair, nemb, nemb, nemb, nemb)
            else:
                eri_res = np.empty((spin_pair, nemb, nemb, nemb, nemb))
                for s in range(spin_pair):
                    eri_res[s] = ao2mo.restore(symmetry, eri[s].real, nemb)
        else:
            raise ValueError("Spin unrestricted ERI does not support 8-fold symmetry.")
    eri_res = np.asarray(eri_res, order="C")

    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

    global cputime_restore, walltime_restore
    cputime_restore += t2[0] - t1[0]
    walltime_restore += t2[1] - t1[1]

    return eri_res
