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

############ pyscf module ############

from pyscf.lib import logger
from pyscf import lib
from pyscf.pbc.gto import Cell
from pyscf.cc import ccsd, dfccsd

#### Xing's LNO ####

from lno.base import LNO
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

############ isdf ############

from pyscf.isdf.isdf_local import ISDF_Local
import pyscf.isdf.misc as misc

### contraction vvvv ###


def _contract_vvvv_t2(
    myisdf: ISDF_Local, cell: Cell, t2, vir_moRg, out=None, verbose=None
):
    """Ht2 = numpy.einsum('ijcd,acdb->ijab', t2, vvvv)

    In the language of THC,

    ijcd,aP,cP,PQ,dQ,bQ->ijab, t2, vir_moRg,vir_moRg,W,vir_moRg,vir_moRg

    should be decomposed as

    ijcd,cP,dQ->ijPQ
    ijPQ,PQ->ijPQ
    ijPQ,aP,bQ->ijab

    note: ijPQ is too large, hence parallel over P!

    Args:
        vvvv : None or integral object
            if vvvv is None, contract t2 to AO-integrals using AO-direct algorithm
    """
    ### some predefined parameter ###

    ### logger ###

    log = logger.Logger(myisdf.stdout, myisdf.verbose)
    time1 = (logger.process_clock(), logger.perf_counter())

    ### preprocess ###

    if myisdf.with_robust_fitting:
        logger.warn(
            myisdf,
            "Robust fitting used in _contract_vvvv_t2, since only W matrix is used, larger error may be introduced!",
        )

    W = myisdf.W
    nvir = vir_moRg.shape[0]
    naux = W.shape[0]

    if out is None:
        Ht2 = ZEROS((nvir, nvir, nvir, nvir), dtype=FLOAT64)
    else:
        Ht2 = ToTENSOR(out)

    assert t2.shape == Ht2.shape

    Ht2 = Ht2.reshape(-1, nvir, nvir)
    t2 = t2.reshape(-1, nvir, nvir)

    # loop over ij #

    for ij in range(Ht2.shape[0]):
        # ijcd,cP,dQ->ijPQ
        tmp = t2[ij]
        tmp1 = DOT(vir_moRg.T, tmp)
        tmp = DOT(tmp1, vir_moRg)
        # ijPQ,PQ->ijPQ
        tmp = CWISE_MUL(tmp, W, out=tmp)
        # ijPQ,aP,bQ->ijab
        tmp1 = DOT(vir_moRg, tmp)
        tmp = DOT(tmp1, vir_moRg.T)
        Ht2[ij] = tmp

    time2 = (logger.process_clock(), logger.perf_counter())

    misc._benchmark_time(time1, time2, "_contract_vvvv_t2", myisdf, myisdf.rank)

    return Ht2.reshape(nvir, nvir, nvir, nvir)


class _ISDFChemistsERIs(_ChemistsERIs):

    def __init__(self, mol=None, isdf=None, occ_moRg=None, vir_moRg=None):
        super().__init__(mol)
        self._isdf = isdf
        self._occ_moRg = occ_moRg
        self._vir_moRg = vir_moRg

    def _contract_vvvv_t2(self, mycc, t2, direct=False, out=None, verbose=None):
        assert not direct
        return _contract_vvvv_t2(self._isdf, self.mol, t2, self._vir_moRg, out, verbose)


### special ao2mo designed for LNO ###

from pyscf.isdf.isdf_ao2mo import AOPAIR_BLKSIZE, _get_moR


def _isdf_ao2mo(
    myisdf, moeri_type: str, occ_moRg, vir_moRg, out=None, AOPAIR_BLKSIZE=AOPAIR_BLKSIZE
):
    assert moeri_type in ["oooo", "ovoo", "ovov", "ovvo", "oovv", "ovvv"]

    EINSUM_IK_JK_IJK = BACKEND._einsum_ik_jk_ijk

    # buffer #

    nocc = occ_moRg.shape[0]
    nvir = vir_moRg.shape[0]
    _map = {"o": nocc, "v": nvir}
    size_p = _map[moeri_type[0]]
    size_q = _map[moeri_type[1]]
    size_r = _map[moeri_type[2]]
    size_s = _map[moeri_type[3]]
    _map = {"o": occ_moRg, "v": vir_moRg}
    moRg_p = _map[moeri_type[0]]
    moRg_q = _map[moeri_type[1]]
    moRg_r = _map[moeri_type[2]]
    moRg_s = _map[moeri_type[3]]

    # memory requires is size_rs * (GRID_BUNCHSIZE) + size_rs * 2 * (GRID_BUNCHSIZE)
    # memory requires is size_qrs * (LOOP_SIZE) + GRID_BUNCHSIZE * size_q * (GRID_BUNCHSIZE)

    GRID_BUNCHIZE = int(AOPAIR_BLKSIZE // (size_r * size_s + size_p * size_q) // 5 * 8)
    GRID_BUNCHIZE = min(GRID_BUNCHIZE, myisdf.naux)
    LOOP_SIZE = int(AOPAIR_BLKSIZE // (size_q * size_r * size_s))
    LOOP_SIZE = min(LOOP_SIZE, size_p)

    size_max = size_r * size_s * GRID_BUNCHIZE + size_r * size_s * 2 * GRID_BUNCHIZE
    size_max2 = (
        size_r * size_s * GRID_BUNCHIZE
        + LOOP_SIZE * size_q * GRID_BUNCHIZE
        + LOOP_SIZE * size_q * size_r * size_s
    )
    size_max = max(size_max, size_max2)

    buffer = SimpleMemoryAllocator(size_max, FLOAT64)

    if moeri_type == "ovvv":
        row_indices, col_indices = np.tril_indices(nvir)
        row_indices = ToTENSOR(np.array(row_indices, dtype=np.int64))
        col_indices = ToTENSOR(np.array(col_indices, dtype=np.int64))
        indices = row_indices * nvir + col_indices
    else:
        indices = None

    # loop over blocks #

    W = myisdf.W

    size_rs = size_r * size_s
    if moeri_type == "ovvv":
        size_rs = nvir * (nvir + 1) // 2
    for p0, p1 in lib.prange(0, myisdf.naux, GRID_BUNCHIZE):
        # construct W * ket #
        moPairRgWKet = buffer.malloc((size_rs, p1 - p0), name="moPairRgWKet")
        CLEAN(moPairRgWKet)
        for q0, q1 in lib.prange(0, myisdf.naux, GRID_BUNCHIZE):
            moPairRg2 = buffer.malloc((size_r, size_s, q1 - q0), name="moPairRg2")
            EINSUM_IK_JK_IJK(moRg_r[:, q0:q1], moRg_s[:, q0:q1], out=moPairRg2)
            if moeri_type == "ovvv":
                moPairRgKet = buffer.malloc((size_rs, q1 - q0), name="moPairRg")
                moPairRg2 = moPairRg2.reshape((nvir * nvir, q1 - q0))
                TAKE(moPairRg2, indices, 0, out=moPairRgKet)
            else:
                moPairRgKet = moPairRg2.reshape((size_rs, q1 - q0))
            DOT(moPairRgKet, W[p0:p1, q0:q1].T, c=moPairRgWKet, beta=1)
            if moeri_type == "ovvv":
                buffer.free(count=2)
            else:
                buffer.free(count=1)
        # construct bra * W * ket #
        for pp0, pp1 in lib.prange(0, size_p, LOOP_SIZE):
            moPairRgBra = buffer.malloc(
                (pp1 - pp0, size_q, p0 - p1), name="moPairRgBra"
            )
            EINSUM_IK_JK_IJK(moRg_p[pp0:pp1, p0:p1], moRg_q[:, p0:p1], out=moPairRgBra)
            moPairRgBra = moPairRgBra.reshape(((pp1 - pp0) * size_q, p1 - p0))
            out_buffer = buffer.malloc((pp1 - pp0, size_q, size_rs), name="out_buffer")
            DOT(moPairRgBra, moPairRgWKet.T, c=out_buffer)
            if moeri_type == "ovvv":
                out[pp0:pp1] += out_buffer.reshape((pp1 - pp0, size_q, size_rs))
            else:
                out[pp0:pp1] += out_buffer.reshape((pp1 - pp0, size_q, size_r, size_s))
            buffer.free(count=2)
        buffer.free(count=1)


class MODIFIED_ISDFCCSD(dfccsd.RCCSD):
    def ao2mo(self, mo_coeff=None):
        return _make_isdf_eris(self, self._isdf, mo_coeff)


def _make_isdf_eris(cc, myisdf: ISDF_Local, mo_coeff=None):
    from pyscf.ao2mo import _ao2mo
    from pyscf.mp.mp2 import _mo_without_core
    from copy import deepcopy

    ## logger ##

    # log = logger.Logger(myisdf.stdout, myisdf.verbose)
    time0 = (logger.process_clock(), logger.perf_counter())

    ## some info for cc ##

    mo_coeff2 = deepcopy(mo_coeff)

    nocc = cc.nocc
    if mo_coeff2 is None:
        mo_coeff2 = cc.mo_coeff
    mo_coeff2 = _mo_without_core(cc, mo_coeff2)
    nmo = mo_coeff2.shape[1]
    nvir = nmo - nocc

    occ_moRg = _get_moR(myisdf.aoRg, mo_coeff2[:, :nocc])
    vir_moRg = _get_moR(myisdf.aoRg, mo_coeff2[:, nocc:])

    ## the original init ##

    eris = _ISDFChemistsERIs(
        mol=None, isdf=myisdf, occ_moRg=occ_moRg, vir_moRg=vir_moRg
    )
    eris._common_init_(cc, mo_coeff)
    nocc = eris.nocc
    nmo = eris.fock.shape[0]
    nvir = nmo - nocc
    nvir_pair = nvir * (nvir + 1) // 2
    with_df = cc.with_df
    assert isinstance(with_df, ISDF_Local)
    naux = eris.naux = with_df.naux

    eris.feri = lib.H5TmpFile()
    eris.oooo = eris.feri.create_dataset("oooo", (nocc, nocc, nocc, nocc), "f8")
    eris.ovoo = eris.feri.create_dataset(
        "ovoo", (nocc, nvir, nocc, nocc), "f8", chunks=(nocc, 1, nocc, nocc)
    )
    eris.ovov = eris.feri.create_dataset(
        "ovov", (nocc, nvir, nocc, nvir), "f8", chunks=(nocc, 1, nocc, nvir)
    )
    eris.ovvo = eris.feri.create_dataset(
        "ovvo", (nocc, nvir, nvir, nocc), "f8", chunks=(nocc, 1, nvir, nocc)
    )
    eris.oovv = eris.feri.create_dataset(
        "oovv", (nocc, nocc, nvir, nvir), "f8", chunks=(nocc, nocc, 1, nvir)
    )
    # nrow ~ 4e9/8/blockdim to ensure hdf5 chunk < 4GB

    mem_now = lib.current_memory()[0]
    max_memory = max(0, cc.max_memory - mem_now)
    vblk = max(nocc, int((max_memory * 0.15e6 / 8) / (nocc * nvir_pair)))
    vvblk = int(
        min(
            nvir_pair,
            4e8 / nocc,
            max(4, (max_memory * 0.8e6 / 8) / (vblk * nocc + naux)),
        )
    )
    eris.ovvv = eris.feri.create_dataset(
        "ovvv", (nocc, nvir, nvir_pair), "f8", chunks=(nocc, 1, vvblk)
    )

    time1 = logger.process_clock(), logger.perf_counter()

    misc._benchmark_time(time0, time1, "_make_isdf_eris", myisdf, myisdf.rank)

    _isdf_ao2mo(myisdf, "oooo", occ_moRg, vir_moRg, eris.oooo, AOPAIR_BLKSIZE)
    _isdf_ao2mo(myisdf, "ovoo", occ_moRg, vir_moRg, eris.ovoo, AOPAIR_BLKSIZE)
    _isdf_ao2mo(myisdf, "ovov", occ_moRg, vir_moRg, eris.ovov, AOPAIR_BLKSIZE)
    eris.ovvo[:] = eris.ovov.transpose(0, 1, 3, 2)
    _isdf_ao2mo(myisdf, "oovv", occ_moRg, vir_moRg, eris.oovv, AOPAIR_BLKSIZE)
    _isdf_ao2mo(myisdf, "ovvv", occ_moRg, vir_moRg, eris.ovvv, AOPAIR_BLKSIZE)

    time2 = logger.process_clock(), logger.perf_counter()

    misc._benchmark_time(time1, time2, "_make_isdf_eris", myisdf, myisdf.rank)

    return eris


########## CCSD solver #############


def CCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    import numpy
    from pyscf import lib
    from pyscf.soscf import newton_ah
    from pyscf import scf
    from lno.cc.ccsd import MODIFIED_CCSD, MODIFIED_DFCCSD, get_maskact

    log = logger.new_logger(mf)

    if isinstance(mf, newton_ah._CIAH_SOSCF) or not isinstance(mf, scf.hf.RHF):
        mf = scf.addons.convert_to_rhf(mf)

    if getattr(mf, "with_df", None):
        """auto-choose if using DFCCSD (storing Lvv) or CCSD (storing vvvv) by memory"""
        naux = mf.with_df.get_naoaux()
        if mo_occ is None:
            mo_occ = mf.mo_occ
        maskocc = mo_occ > 1e-10
        frozen, maskact = get_maskact(frozen, len(mo_occ))
        nvir = np.count_nonzero(~maskocc & maskact)
        nvir_pair = nvir * (nvir + 1) // 2
        if isinstance(mf.with_df, ISDF_Local):
            return MODIFIED_ISDFCCSD(mf, frozen, mo_coeff, mo_occ)
        if naux > nvir_pair:
            log.debug1("naux= %d > nvir_pair= %d -> using DFCCSD", naux, nvir_pair)
            return MODIFIED_CCSD(mf, frozen, mo_coeff, mo_occ)
        else:
            log.debug1("naux= %d <= nvir_pair= %d -> using DFCCSD", naux, nvir_pair)
            return MODIFIED_DFCCSD(mf, frozen, mo_coeff, mo_occ)

    elif np.iscomplexobj(mo_coeff) or np.iscomplexobj(mf.mo_coeff):
        log.error("Not implemented for now. Contact Hong-Zhou via hzyechem@gmail.com")
        raise NotImplementedError

    else:
        return MODIFIED_CCSD(mf, frozen, mo_coeff, mo_occ)


""" impurity solver for LNO-based CCSD/CCSD_T
"""


def impurity_solve(
    mf,
    mo_coeff,
    lo_coeff,
    ccsd_t=False,
    eris=None,
    frozen=None,
    log=None,
    verbose_imp=0,
):
    r"""Solve impurity problem and calculate local correlation energy.

    Args:
        mo_coeff (np.ndarray):
            MOs where the impurity problem is solved.
        lo_coeff (np.ndarray):
            LOs which the local correlation energy is calculated for.
        ccsd_t (bool):
            If True, CCSD(T) energy is calculated and returned as the third
            item (0 is returned otherwise).
        frozen (int or list; optional):
            Same syntax as `frozen` in MP2, CCSD, etc.

    Return:
        e_loc_corr_pt2, e_loc_corr_ccsd, e_loc_corr_ccsd_t:
            Local correlation energy at MP2, CCSD, and CCSD(T) level. Note that
            the CCSD(T) energy is 0 unless 'ccsd_t' is set to True.
    """

    # copy from lno.cc.ccsd #
    from lno.cc.ccsd import get_maskact, get_fragment_energy

    _fdot = np.dot
    fdot = lambda *args: reduce(_fdot, args)
    einsum = lib.einsum
    # ------------------ #

    log = logger.new_logger(mf if log is None else log)
    cput1 = (logger.process_clock(), logger.perf_counter())

    maskocc = mf.mo_occ > 1e-10
    nocc = np.count_nonzero(maskocc)
    nmo = mf.mo_occ.size

    frozen, maskact = get_maskact(frozen, nmo)

    orbfrzocc = mo_coeff[:, ~maskact & maskocc]
    orbactocc = mo_coeff[:, maskact & maskocc]
    orbactvir = mo_coeff[:, maskact & ~maskocc]
    orbfrzvir = mo_coeff[:, ~maskact & ~maskocc]
    nfrzocc, nactocc, nactvir, nfrzvir = [
        orb.shape[1] for orb in [orbfrzocc, orbactocc, orbactvir, orbfrzvir]
    ]
    nlo = lo_coeff.shape[1]
    s1e = mf.get_ovlp() if eris is None else eris.s1e
    prjlo = fdot(lo_coeff.T, s1e, orbactocc)

    log.debug(
        "    impsol:  %d LOs  %d/%d MOs  %d occ  %d vir",
        nlo,
        nactocc + nactvir,
        nmo,
        nactocc,
        nactvir,
    )

    # solve impurity problem
    mcc = CCSD(mf, mo_coeff=mo_coeff, frozen=frozen).set(verbose=verbose_imp)
    mcc._s1e = s1e
    if eris is not None:
        mcc._h1e = eris.h1e
        mcc._vhf = eris.vhf
    imp_eris = mcc.ao2mo()
    if isinstance(imp_eris.ovov, np.ndarray):
        ovov = imp_eris.ovov
    else:
        ovov = imp_eris.ovov[()]
    oovv = ovov.reshape(nactocc, nactvir, nactocc, nactvir).transpose(0, 2, 1, 3)
    ovov = None
    cput1 = log.timer_debug1("imp sol - eri    ", *cput1)
    # MP2 fragment energy
    t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
    cput1 = log.timer_debug1("imp sol - mp2 amp", *cput1)
    elcorr_pt2 = get_fragment_energy(oovv, t2, prjlo)
    cput1 = log.timer_debug1("imp sol - mp2 ene", *cput1)
    # CCSD fragment energy
    t1, t2 = mcc.kernel(eris=imp_eris, t1=t1, t2=t2)[1:]
    cput1 = log.timer_debug1("imp sol - cc  amp", *cput1)
    t2 += einsum("ia,jb->ijab", t1, t1)
    elcorr_cc = get_fragment_energy(oovv, t2, prjlo)
    cput1 = log.timer_debug1("imp sol - cc  ene", *cput1)
    if ccsd_t:
        from lno.cc.ccsd_t import kernel as CCSD_T

        t2 -= einsum("ia,jb->ijab", t1, t1)  # restore t2
        elcorr_cc_t = CCSD_T(mcc, imp_eris, prjlo, t1=t1, t2=t2, verbose=verbose_imp)
        cput1 = log.timer_debug1("imp sol - cc  (T)", *cput1)
    else:
        elcorr_cc_t = 0.0

    frag_msg = "  ".join(
        [
            f"E_corr(MP2) = {elcorr_pt2:.15g}",
            f"E_corr(CCSD) = {elcorr_cc:.15g}",
            f"E_corr(CCSD(T)) = {elcorr_cc_t:.15g}",
        ]
    )

    t1 = t2 = oovv = imp_eris = mcc = None

    return frag_msg, (elcorr_pt2, elcorr_cc, elcorr_cc_t)

from lno.cc.ccsd import LNOCCSD

class LNOCCSD_ISDF(LNOCCSD):
    def impurity_solve(self, mf, mo_coeff, lo_coeff, eris=None, frozen=None, log=None):
        return impurity_solve(
            mf,
            mo_coeff,
            lo_coeff,
            eris=eris,
            frozen=frozen,
            log=log,
            verbose_imp=self.verbose_imp,
            ccsd_t=self.ccsd_t,
        )
