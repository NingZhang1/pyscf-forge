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

import sys
import numpy as np
from functools import reduce
from copy import deepcopy

############ pyscf module ############

from pyscf.lib import logger
from pyscf import lib
from pyscf.pbc.gto import Cell
from pyscf.cc import ccsd, dfccsd

#### Xing's LNO ####

from lno.base import LNO
from lno.base.lno import (
    _LNOERIS,
    projection_construction,
    subspace_eigh,
    make_rdm1_mp2,
    natorb_compression,
)
from lno.cc.ccsd import _ChemistsERIs

############ isdf backends ############

import pyscf.isdf.BackEnd.isdf_backend as BACKEND
from pyscf.isdf.BackEnd.isdf_memory_allocator import SimpleMemoryAllocator

NUM_THREADS = BACKEND.NUM_THREADS

FLOAT64 = BACKEND.FLOAT64
ToNUMPY = BACKEND._toNumpy
ToTENSOR = BACKEND._toTensor
ZEROS = BACKEND._zeros
MAX = BACKEND._maximum
ABS = BACKEND._absolute
DOT = BACKEND._dot
TAKE = BACKEND._take
CLEAN = BACKEND._clean
MALLOC = BACKEND._malloc
CWISE_MUL = BACKEND._cwise_mul

# the function needed to make the backend and Xing's code compaible #


def fdot(*arg):
    _class_tmp = arg[0].__class__
    assert all([isinstance(a, _class_tmp) for a in arg])
    return_numpy = isinstance(arg[0], np.ndarray)
    arg = [ToTENSOR(a) for a in arg]
    res = reduce(DOT, arg)
    if return_numpy:
        return ToNUMPY(res)
    return res


############ isdf ############

from pyscf.isdf.isdf_local import ISDF_Local
import pyscf.isdf.misc as misc

# from pyscf.isdf.isdf_lno import _isdf_ao2mo

### the only func in build fpno needed to rewrite for isdf ###


def make_fpno1(
    mfcc,
    eris,
    orbfragloc,
    no_type,
    thresh_internal,
    thresh_external,
    frozen_mask=None,
    frag_target_nocc=None,
    frag_target_nvir=None,
):
    if not isinstance(eris, _LNOISDFINCOREERIS):
        from lno.base.lno import make_fpno1 as make_fpno1_original

        return make_fpno1_original(
            mfcc,
            eris,
            orbfragloc,
            no_type,
            thresh_internal,
            thresh_external,
            frozen_mask,
            frag_target_nocc,
            frag_target_nvir,
        )

    log = logger.Logger(mfcc.stdout, mfcc.verbose)

    mf = mfcc._scf
    nocc = np.count_nonzero(mf.mo_occ > 1e-10)
    nmo = mf.mo_occ.size
    orbocc0, orbocc1, orbvir1, orbvir0 = mfcc.split_mo()
    moeocc0, moeocc1, moevir1, moevir0 = mfcc.split_moe()
    nocc0, nocc1, nvir1, nvir0 = [m.size for m in [moeocc0, moeocc1, moevir1, moevir0]]
    # nlo = orbfragloc.shape[1]
    s1e = eris.s1e  # if eris.s1e is None else mf.get_ovlp()
    fock = eris.fock  # if eris.fock is None else mf.get_fock()
    # Lov = eris.Lov
    lovir = abs(fdot(orbfragloc.T, s1e, orbvir1)).max() > 1e-10

    if isinstance(thresh_external, float):
        thresh_ext_occ = thresh_ext_vir = thresh_external
    else:
        thresh_ext_occ, thresh_ext_vir = thresh_external

    # sanity check for no_type:
    if not lovir and no_type[0] != "i":
        log.warn('Input LOs span only occ but input no_type[0] is not "i".')
        raise ValueError
    if not lovir and no_type[1] == "i":
        log.warn('Input LOs span only occ but input no_type[1] is "i".')
        raise ValueError

    # split active occ/vir into internal(1) and external(2)
    m = fdot(orbfragloc.T, s1e, orbocc1)
    uocc1, uocc2 = projection_construction(m, thresh_internal)
    moefragocc1, orbfragocc1 = subspace_eigh(fock, fdot(orbocc1, uocc1))
    if lovir:
        m = fdot(orbfragloc.T, s1e, orbvir1)
        uvir1, uvir2 = projection_construction(m, thresh_internal)
        moefragvir1, orbfragvir1 = subspace_eigh(fock, fdot(orbvir1, uvir1))

    def moe_Ov(moefragocc):
        return (moefragocc[:, None] - moevir1).reshape(-1)

    def moe_oV(moefragvir):
        return (moeocc1[:, None] - moefragvir).reshape(-1)

    eov = moe_Ov(moeocc1)

    # Construct PT2 dm_vv
    if no_type[1] == "r":  # OvOv: IaJc,IbJc->ab
        u = fdot(orbocc1.T, s1e, orbfragocc1)
        ovov = eris.get_OvOv(u)  # to ISDF
        eia = ejb = moe_Ov(moefragocc1)
        e1_or_e2 = "e1"
        swapidx = "ab"
    elif no_type[1] == "e":  # Ovov: Iajc,Ibjc->ab
        u = fdot(orbocc1.T, s1e, orbfragocc1)
        ovov = eris.get_Ovov(u)  # to ISDF
        eia = moe_Ov(moefragocc1)
        # Ljb = Lov
        ejb = eov
        e1_or_e2 = "e1"
        swapidx = "ab"
    else:  # oVov: iCja,iCjb->ab
        u = fdot(orbvir1.T, s1e, orbfragvir1)
        ovov = eris.get_oVov(u)  # to ISDF
        eia = moe_oV(moefragvir1)
        # Ljb = Lov
        ejb = eov
        e1_or_e2 = "e2"
        swapidx = "ij"

    eiajb = (eia[:, None] + ejb).reshape(*ovov.shape)
    t2 = ovov / eiajb

    dmvv = make_rdm1_mp2(t2, "vv", e1_or_e2, swapidx)
    if lovir:
        dmvv = fdot(uvir2.T, dmvv, uvir2)

    # Lia = Ljb = ovov = eiajb = None
    ovov = eiajb = None

    # Construct PT2 dm_oo
    if no_type in ["ie", "ei"]:  # ie/ei share same t2
        if no_type[0] == "e":  # oVov: iAkb,jAkb->ij
            e1_or_e2 = "e1"
            swapidx = "ij"
        else:  # Ovov: Kaib,Kajb->ij
            e1_or_e2 = "e2"
            swapidx = "ab"
    else:
        t2 = None

        if no_type[0] == "r":  # oVoV: iAkB,jAkB->ij
            u = fdot(orbvir1.T, s1e, orbfragvir1)
            ovov = eris.get_oVoV(u)
            eia = ejb = moe_oV(moefragvir1)
            e1_or_e2 = "e1"
            swapidx = "ab"
        elif no_type[0] == "e":  # oVov: iAkb,jAkb->ij
            u = fdot(orbvir1.T, s1e, orbfragvir1)
            ovov = eris.get_oVov(u)
            eia = moe_oV(moefragvir1)
            # Ljb = Lov
            ejb = eov
            e1_or_e2 = "e1"
            swapidx = "ij"
        else:  # Ovov: Kaib,Kajb->ij
            u = fdot(orbocc1.T, s1e, orbfragocc1)
            ovov = eris.get_Ovov(u)
            eia = moe_Ov(moefragocc1)
            # Ljb = Lov
            ejb = eov
            e1_or_e2 = "e2"
            swapidx = "ab"

        eiajb = (eia[:, None] + ejb).reshape(*ovov.shape)
        t2 = ovov / eiajb

        # Lia = Ljb = ovov = eiajb = None
        ovov = eiajb = None

    dmoo = make_rdm1_mp2(t2, "oo", e1_or_e2, swapidx)
    dmoo = fdot(uocc2.T, dmoo, uocc2)

    t2 = None

    # Compress external space by PNO
    if frag_target_nocc is not None:
        frag_target_nocc -= orbfragocc1.shape[1]
    orbfragocc2, orbfragocc0 = natorb_compression(
        dmoo, orbocc1, thresh_ext_occ, uocc2, frag_target_nocc
    )
    orbfragocc12 = subspace_eigh(fock, np.hstack([orbfragocc2, orbfragocc1]))[1]
    if lovir:
        if frag_target_nvir is not None:
            frag_target_nvir -= orbfragvir1.shape[1]
        orbfragvir2, orbfragvir0 = natorb_compression(
            dmvv, orbvir1, thresh_ext_vir, uvir2, frag_target_nvir
        )
        orbfragvir12 = subspace_eigh(fock, np.hstack([orbfragvir2, orbfragvir1]))[1]
    else:
        orbfragvir2, orbfragvir0 = natorb_compression(
            dmvv, orbvir1, thresh_ext_vir, None, frag_target_nvir
        )
        orbfragvir12 = subspace_eigh(fock, orbfragvir2)[1]

    orbfrag = np.hstack(
        [orbocc0, orbfragocc0, orbfragocc12, orbfragvir12, orbfragvir0, orbvir0]
    )
    frzfrag = np.hstack(
        [
            np.arange(orbocc0.shape[1] + orbfragocc0.shape[1]),
            np.arange(nocc + orbfragvir12.shape[1], nmo),
        ]
    )

    return frzfrag, orbfrag


class _LNOISDFINCOREERIS(_LNOERIS):

    def _common_init_(self, mcc):
        log = logger.new_logger(mcc)
        _LNOERIS._common_init_(self, mcc)
        orbo, orbv = mcc.split_mo()[1:3]
        # print("orbo.shape = ", orbo.shape)
        # print("orbv.shape = ", orbv.shape)
        # self.Lov = get_Lov(mcc._scf, orbo, orbv, log=log)
        self._orbo = ToTENSOR(orbo)
        self._orbv = ToTENSOR(orbv)
        assert hasattr(mcc, "_isdf")
        self._isdf = mcc._isdf
        assert isinstance(self._isdf, ISDF_Local)

    def get_OvOv(self, u):
        return_numpy = isinstance(u, np.ndarray)
        mo_coeff_O = DOT(self._orbo, ToTENSOR(u))
        mo_coeff_v = deepcopy(self._orbv)
        res = self._isdf.ao2mo(
            (mo_coeff_O, mo_coeff_v, mo_coeff_O, mo_coeff_v), compact=False
        )
        if return_numpy:
            return ToNUMPY(res)
        return res

    def get_Ovov(self, u):
        return_numpy = isinstance(u, np.ndarray)
        mo_coeff_O = DOT(self._orbo, ToTENSOR(u))
        mo_coeff_v = deepcopy(self._orbv)
        mo_coeff_o = deepcopy(self._orbo)
        res = self._isdf.ao2mo(
            (mo_coeff_O, mo_coeff_v, mo_coeff_o, mo_coeff_v), compact=False
        )
        if return_numpy:
            return ToNUMPY(res)
        return res

    def get_oVov(self, u):
        return_numpy = isinstance(u, np.ndarray)
        mo_coeff_V = DOT(self._orbv, ToTENSOR(u))
        mo_coeff_v = deepcopy(self._orbv)
        mo_coeff_o = deepcopy(self._orbo)
        res = self._isdf.ao2mo(
            (mo_coeff_o, mo_coeff_V, mo_coeff_o, mo_coeff_v), compact=False
        )
        if return_numpy:
            return ToNUMPY(res)
        return res

    def get_oVoV(self, u):
        return_numpy = isinstance(u, np.ndarray)
        mo_coeff_V = DOT(self._orbv, ToTENSOR(u))
        mo_coeff_o = deepcopy(self._orbo)
        res = self._isdf.ao2mo(
            (mo_coeff_o, mo_coeff_V, mo_coeff_o, mo_coeff_V), compact=False
        )
        if return_numpy:
            return ToNUMPY(res)
        return res


def _make_isdf_eris(mcc, fock=None, s1e=None):
    if fock is None:
        fock = mcc.fock
    if s1e is None:
        s1e = mcc.s1e
    eris = _LNOISDFINCOREERIS(fock=fock, s1e=s1e)
    eris._common_init_(mcc)
    return eris


class LNO_ISDF(LNO):

    def ao2mo(self, fock=None, s1e=None):
        mf = self._scf
        if getattr(mf, "with_df", None):
            if not isinstance(mf.with_df, ISDF_Local):
                return super().ao2mo(fock, s1e)
        else:
            return super().ao2mo(fock, s1e)
        # ao2mo with ISDF #

        if fock is None:
            fock = self.fock
        if s1e is None:
            s1e = self.s1e

        log = logger.Logger(self.stdout, self.verbose)
        cput0 = (logger.process_clock(), logger.perf_counter())
        orbocc, orbvir = self.split_mo()[1:3]
        nocc = orbocc.shape[1]
        nvir = orbvir.shape[1]
        mf = self._scf
        # FIXME: more accurate mem estimate
        # mem_now = self.max_memory - lib.current_memory()[0]
        # mem_incore0 = (nocc*nvir)**2*8/1024**2.
        # mem_incore = mem_incore0 * 3 # 3 for tIajb, eIajb etc.
        eris = _make_isdf_eris(self, fock, s1e)
        cput1 = log.timer("Integral xform   ", *cput0)

        return eris

    make_fpno1 = make_fpno1
