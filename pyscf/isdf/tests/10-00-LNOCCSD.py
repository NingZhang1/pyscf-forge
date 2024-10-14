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
from pyscf.isdf.isdf_lno_cc import LNOCCSD_ISDF
from lno.tools import guess_frozen

# isdf util #

from pyscf.isdf.isdf_tools_Tsym import _kmesh_to_Kpoints, _1e_operator_gamma2k
from pyscf.isdf import isdf_tools_cell
from pyscf.isdf.isdf import ISDF
from pyscf.isdf.isdf_local import ISDF_Local
from pyscf.isdf.isdf_to_df import DF_ISDF, GDF_ISDF

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
mf.with_df = df
# mf = scf.RHF(cell, exxdiv='ewald').density_fit()
mf.kernel()
mmp = mp.RMP2(mf)
mmp.kernel()
mcc = cc.RCCSD(mf)
mcc.kernel()


def run1(mf, thresh, frozen, params=None):
    mfcc = LNOCCSD(mf, thresh=thresh, frozen=frozen)
    if isinstance(params, dict):
        mfcc.set(**params)
    mfcc.kernel()
    return mfcc


thresh = 0.0
params = {"lo_type": "pm", "no_type": "cim", "frag_lolist": "1o"}
mfcc = run1(mf, thresh, None, params=params)

def run2(mf, thresh, frozen, params=None):
    mfcc = LNOCCSD_ISDF(mf, thresh=thresh, frozen=frozen)
    if isinstance(params, dict):
        mfcc.set(**params)
    mfcc.kernel()
    return mfcc

thresh = 0.0
params = {"lo_type": "pm", "no_type": "cim", "frag_lolist": "1o"}
mfcc2 = run2(mf, thresh, None, params=params)