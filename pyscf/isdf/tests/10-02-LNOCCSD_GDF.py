# backend to test #

import pyscf.isdf.BackEnd._config as config

config.disable_fftw()
config.backend("torch")
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

MAX = BACKEND._maximum
ABS = BACKEND._absolute
ToTENSOR = BACKEND._toTensor

import numpy as np
import numpy

from pyscf.pbc import gto, scf, mp, cc
from pyscf.pbc.tools import super_cell
from pyscf import lib
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import dft as pbcdft
from pyscf.pbc.dft import multigrid

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

cell = pbcgto.Cell()

# Molecule
# boxlen = 12.4138
cell.a = numpy.array(
    [[5.9721579552, 0.0, 0.0], [0.0, 5.9721579552, 0.0], [0.0, 0.0, 21.6602222298]]
)
cell.atom = [
    ["C", (1.4930464485, 1.4930276945, 5.5505207056)],
    ["Mg", (0.0000000000, 0.0000000000, 1.0000000000)],
    ["Mg", (0.0000000000, 2.9860789776, 1.0000000000)],
    ["Mg", (2.9860789776, 0.0000000000, 1.0000000000)],
    ["Mg", (2.9860789776, 2.9860789776, 1.0000000000)],
    ["Mg", (1.4930210194, 1.4930383421, 3.1510006291)],
    ["Mg", (1.4930604908, 4.4791138260, 3.1015843767)],
    ["Mg", (4.4791078668, 1.4930453381, 3.1015875384)],
    ["Mg", (4.4791306128, 4.4791082497, 3.1031282373)],
    ["O", (1.4930390000, 1.4930390000, 1.0000000000)],
    ["O", (1.4930390000, 4.4791179776, 1.0000000000)],
    ["O", (4.4791179776, 1.4930390000, 1.0000000000)],
    ["O", (4.4791179776, 4.4791179776, 1.0000000000)],
    ["O", (-0.0067963799, -0.0067976478, 3.1562024896)],
    ["O", (-0.0067617311, 2.9928405345, 3.1566352624)],
    ["O", (2.9928271407, -0.0067655095, 3.1566170706)],
    ["O", (2.9928736174, 2.9928616433, 3.1562072754)],
    ["O", (1.4930871445, 1.4930252614, 6.6915399688)],
]

import os

dirnow = os.path.realpath(os.path.join(__file__, ".."))
basis_file_from_user = os.path.join(dirnow, "basis/gth-cc-pvdz.dat")

# cell.basis = "gth-dzvp"
cell.ke_cutoff = 384  # kinetic energy cutoff in a.u.
cell.max_memory = 8000  # in MB
cell.precision = 1e-6  # integral precision
cell.pseudo = "gth-hf-rev"
cell.verbose = 11
cell.use_loose_rcut = True  # integral screening based on shell radii
cell.use_particle_mesh_ewald = True  # use particle mesh ewald for nuclear repulsion
cell.build()

natm = cell.natm
group = [[i] for i in range(natm)]  # not the best way, but currently the only way!

### first search the parameter for ISDF ###

# exit(1)

def run1(mf, thresh, frozen, params=None):
    mfcc = LNOCCSD(mf, thresh=thresh, frozen=frozen)
    if isinstance(params, dict):
        mfcc.set(**params)
    mfcc.kernel()
    return mfcc

def run2(mf, thresh, frozen, params=None):
    mfcc = LNOCCSD_ISDF(mf, thresh=thresh, frozen=frozen)
    if isinstance(params, dict):
        mfcc.set(**params)
    mfcc.kernel()
    return mfcc


# for qr_cutoff in [1e-3, 3e-4, 1e-4]:
#isdf = ISDF_Local(
#    cell, with_robust_fitting=False, limited_memory=True, build_V_K_bunchsize=256
#)
#isdf.build(c=40, m=5, rela_cutoff=qr_cutoff, group=group)
mf = scf.RHF(cell).density_fit()
# mf.with_df = isdf
mf.kernel()

# LNO CCSD #

for thresh in [1e-3, 1e-4, 1e-5, 1e-6]:
    # thresh = 0.0
    # mf.with_df = isdf
    print(" ************ LNO CCSD with threshold %e ************" % (thresh))
    params = {"lo_type": "pm", "no_type": "cim", "frag_lolist": "1o"}
    mfcc2 = run1(mf, thresh, None, params=params)
