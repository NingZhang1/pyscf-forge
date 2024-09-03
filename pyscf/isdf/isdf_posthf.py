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

import numpy
import numpy as np
import ctypes

############ pyscf module ############

import pyscf
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.lib import logger
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact
import pyscf.pbc.gto as pbcgto
from pyscf.cc.rccsd import _ChemistsERIs, RCCSD

############ isdf utils ############

from pyscf.isdf.isdf_local import ISDF_Local
from pyscf.isdf.isdf import ISDF

####################################

### post-HF with ISDF ERIs (NOT THC-POSTHF!)

####################################

############ subroutines ---- deal with CC ############


def _make_isdf_eris_incore(mycc, my_isdf: ISDF_Local, mo_coeff=None):
    cput0 = (logger.process_clock(), logger.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]

    eri1 = my_isdf.ao2mo(mo_coeff, compact=False).reshape(nmo, nmo, nmo, nmo)
    eris.oooo = eri1[:nocc, :nocc, :nocc, :nocc].copy()
    eris.ovoo = eri1[:nocc, nocc:, :nocc, :nocc].copy()
    eris.ovov = eri1[:nocc, nocc:, :nocc, nocc:].copy()
    eris.oovv = eri1[:nocc, :nocc, nocc:, nocc:].copy()
    eris.ovvo = eri1[:nocc, nocc:, nocc:, :nocc].copy()
    eris.ovvv = eri1[:nocc, nocc:, nocc:, nocc:].copy()
    eris.vvvv = eri1[nocc:, nocc:, nocc:, nocc:].copy()
    logger.timer(mycc, "CCSD integral transformation", *cput0)

    cput1 = (logger.process_clock(), logger.perf_counter())

    _benchmark_time(cput0, cput1, "CCSD integral transformation", my_isdf)

    return eris


def RCCSD_isdf(mf, frozen=0, mo_coeff=None, mo_occ=None, run=True, cc2=False):
    mycc = RCCSD(mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
    mycc.cc2 = cc2
    # eris = mycc.ao2mo(mo_coeff)
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    eris_ccsd = _make_isdf_eris_incore(mycc, mf.with_df, mo_coeff=mo_coeff)
    # mycc.eris = eris
    if run:
        mycc.kernel(eris=eris_ccsd)
    return mycc, eris_ccsd
