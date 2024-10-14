# backend to test #

import pyscf.isdf.BackEnd._config as config

config.disable_fftw()
config.backend("torch")
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

MAX = BACKEND._maximum
ABS = BACKEND._absolute
ToTENSOR = BACKEND._toTensor

import numpy as np

from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.tools import super_cell

from pyscf import lib

from lno.cc import LNOCCSD

from lno.tools import guess_frozen

# isdf util #

from pyscf.isdf.isdf_tools_Tsym import _kmesh_to_Kpoints, _1e_operator_gamma2k
from pyscf.isdf import isdf_tools_cell
from pyscf.isdf.isdf import ISDF
from pyscf.isdf.isdf_local import ISDF_Local
from pyscf.isdf.isdf_to_df import DF_ISDF, GDF_ISDF
from pyscf.isdf.isdf_ao2mo import _get_moR
from pyscf.isdf.isdf_local_ao2mo import _general
from pyscf.isdf.isdf_lno_cc import LNOCCSD_ISDF, _contract_vvvv_t2, _make_isdf_eris
from pyscf.isdf.isdf_lno_cc import MODIFIED_ISDFCCSD
from pyscf.isdf.isdf_tools_local import _pack_aoR_holder

# test #

ke_cutoff = 256
basis = "gth-dzvp"

boxlen = 2.0
prim_a = np.array([[boxlen, 0.0, 0.0], [0.0, boxlen, 0.0], [0.0, 0.0, boxlen]])
atm = [
    ["Ne", (0.0, 0.0, 0.0)],
]

kmeshes = [
    [1, 1, 1],
]
VERBOSE = 10

cell = isdf_tools_cell.build_supercell(
    atm,
    prim_a,
    Ls=[4, 1, 1],
    ke_cutoff=ke_cutoff,
    basis=basis,
    pseudo="gth-pade",
    verbose=VERBOSE,
)
# frozen = guess_frozen(cell)

group = [[0], [1], [2], [3]]

isdf = ISDF_Local(
    cell, with_robust_fitting=False, limited_memory=True, build_V_K_bunchsize=56
)
isdf.build(c=45, m=5, rela_cutoff=1e-4, group=group)
df = GDF_ISDF(isdf)
df.build()

mf = scf.RHF(cell, exxdiv="ewald")
mf.with_df = isdf
# mf.with_df = df
# mf = scf.RHF(cell, exxdiv='ewald').density_fit()
mf.kernel()

nocc = 16
nvir = cell.nao_nr() - nocc

occ_coeff = mf.mo_coeff[:, :nocc]
vir_coeff = mf.mo_coeff[:, nocc:]


moeri_oooo1 = isdf.ao2mo((occ_coeff, occ_coeff, occ_coeff, occ_coeff), compact=False)
moeri_oooo2 = _general(isdf, (occ_coeff, occ_coeff, occ_coeff, occ_coeff))

diff = np.linalg.norm(moeri_oooo1 - moeri_oooo2)
print(diff)

moeri_ovvo1 = isdf.ao2mo((occ_coeff, vir_coeff, vir_coeff, occ_coeff), compact=False)
moeri_ovvo2 = _general(isdf, (occ_coeff, vir_coeff, vir_coeff, occ_coeff), debug=True)

diff = np.linalg.norm(moeri_ovvo1 - moeri_ovvo2)
print(diff)

# driver in lno.cc.LNOCCSD

## contract t2 and vvvv

moeri_vvvv = isdf.ao2mo((vir_coeff, vir_coeff, vir_coeff, vir_coeff), compact=False)
t2 = np.random.rand(nocc, nocc, nvir, nvir)
moRg_vir = _get_moR(isdf.aoRg, vir_coeff)
moRg_vir = _pack_aoR_holder(moRg_vir, nvir).aoR

benchmark = np.einsum("ijcd,acdb->ijab", t2, moeri_vvvv)
calculated = _contract_vvvv_t2(isdf, cell, t2, moRg_vir)
diff = np.linalg.norm(benchmark - calculated)
print(diff)

#### LNO-CCSD ####

from pyscf.pbc import gto, scf, mp, cc

params = {"lo_type": "pm", "no_type": "cim", "frag_lolist": "1o"}
mfcc = MODIFIED_ISDFCCSD(mf, None, mf.mo_coeff, mf.mo_occ)

eris = _make_isdf_eris(mfcc, isdf, mf.mo_coeff)

oooo = np.array(eris.oooo)
ovoo = np.array(eris.ovoo)
ovov = np.array(eris.ovov)
ovvo = np.array(eris.ovvo)
oovv = np.array(eris.oovv)
ovvv = lib.unpack_tril(np.array(eris.ovvv).reshape(nocc * nvir, -1)).reshape(
    nocc, nvir, nvir, nvir
)

moeri_oooo = isdf.ao2mo((occ_coeff, occ_coeff, occ_coeff, occ_coeff), compact=False)
moeri_ovoo = isdf.ao2mo((occ_coeff, vir_coeff, occ_coeff, occ_coeff), compact=False)
moeri_ovov = isdf.ao2mo((occ_coeff, vir_coeff, occ_coeff, vir_coeff), compact=False)
moeri_ovvo = isdf.ao2mo((occ_coeff, vir_coeff, vir_coeff, occ_coeff), compact=False)
moeri_oovv = isdf.ao2mo((occ_coeff, occ_coeff, vir_coeff, vir_coeff), compact=False)
moeri_ovvv = isdf.ao2mo((occ_coeff, vir_coeff, vir_coeff, vir_coeff), compact=False)

diff = np.linalg.norm(oooo - moeri_oooo)
print(diff)
diff = np.linalg.norm(ovoo - moeri_ovoo)
print(diff)
diff = np.linalg.norm(ovov - moeri_ovov)
print(diff)
diff = np.linalg.norm(ovvo - moeri_ovvo)
print(diff)
diff = np.linalg.norm(oovv - moeri_oovv)
print(diff)
diff = np.linalg.norm(ovvv - moeri_ovvv)
print(diff)
