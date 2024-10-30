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
    [[11.9443159104, 0.0, 0.0], [0.0, 11.9443159104, 0.0], [0.0, 0.0, 21.6602222298]]
)
cell.atom = [
    ["C", (1.4930428341, 1.4930432491, 5.5284856213)],
    ["Mg", (0.0000000000, 0.0000000000, 1.0000000000)],
    ["Mg", (8.9582369328, 5.9721579552, 1.0000000000)],
    ["Mg", (2.9860789776, 5.9721579552, 1.0000000000)],
    ["Mg", (2.9860789776, 8.9582369328, 1.0000000000)],
    ["Mg", (5.9721579552, 0.0000000000, 1.0000000000)],
    ["Mg", (5.9721579552, 2.9860789776, 1.0000000000)],
    ["Mg", (5.9721579552, 5.9721579552, 1.0000000000)],
    ["Mg", (8.9582369328, 2.9860789776, 1.0000000000)],
    ["Mg", (5.9721579552, 8.9582369328, 1.0000000000)],
    ["Mg", (8.9582369328, 0.0000000000, 1.0000000000)],
    ["Mg", (2.9860789776, 2.9860789776, 1.0000000000)],
    ["Mg", (8.9582369328, 8.9582369328, 1.0000000000)],
    ["Mg", (0.0000000000, 8.9582369328, 1.0000000000)],
    ["Mg", (0.0000000000, 2.9860789776, 1.0000000000)],
    ["Mg", (0.0000000000, 5.9721579552, 1.0000000000)],
    ["Mg", (2.9860789776, 0.0000000000, 1.0000000000)],
    ["Mg", (1.4930343349, 1.4930374153, 3.1511187175)],
    ["Mg", (7.4651977897, 10.4517004415, 3.1114386205)],
    ["Mg", (7.4651969790, 7.4651968898, 3.1112699448)],
    ["Mg", (1.4930327857, 4.4811832720, 3.1056362779)],
    ["Mg", (7.4651966404, 4.4786922443, 3.1114383107)],
    ["Mg", (4.4811624032, 1.4930368520, 3.1056318890)],
    ["Mg", (10.4496839597, 4.4807048989, 3.1095898700)],
    ["Mg", (1.4930399629, 7.4651963524, 3.1116080311)],
    ["Mg", (10.4496925734, 10.4496968378, 3.1095846708)],
    ["Mg", (4.4807088948, 10.4496904623, 3.1095885984)],
    ["Mg", (4.4786954624, 7.4651978298, 3.1114386978)],
    ["Mg", (1.4930424686, 10.4492084126, 3.1056383774)],
    ["Mg", (10.4516980458, 7.4651970662, 3.1114406695)],
    ["Mg", (4.4807013212, 4.4807040921, 3.1095854125)],
    ["Mg", (7.4651986133, 1.4930389816, 3.1116104730)],
    ["Mg", (10.4492249714, 1.4930441995, 3.1056280117)],
    ["O", (10.4512759328, 1.4930390000, 1.0000000000)],
    ["O", (4.4791179776, 10.4512759328, 1.0000000000)],
    ["O", (7.4651969552, 1.4930390000, 1.0000000000)],
    ["O", (4.4791179776, 4.4791179776, 1.0000000000)],
    ["O", (10.4512759328, 4.4791179776, 1.0000000000)],
    ["O", (7.4651969552, 7.4651969552, 1.0000000000)],
    ["O", (7.4651969552, 10.4512759328, 1.0000000000)],
    ["O", (7.4651969552, 4.4791179776, 1.0000000000)],
    ["O", (4.4791179776, 7.4651969552, 1.0000000000)],
    ["O", (1.4930390000, 10.4512759328, 1.0000000000)],
    ["O", (10.4512759328, 7.4651969552, 1.0000000000)],
    ["O", (1.4930390000, 4.4791179776, 1.0000000000)],
    ["O", (1.4930390000, 7.4651969552, 1.0000000000)],
    ["O", (1.4930390000, 1.4930390000, 1.0000000000)],
    ["O", (4.4791179776, 1.4930390000, 1.0000000000)],
    ["O", (10.4512759328, 10.4512759328, 1.0000000000)],
    ["O", (8.9580238897, 5.9723685818, 3.1594644112)],
    ["O", (8.9570961456, 2.9857717252, 3.1600752389)],
    ["O", (8.9570985067, 0.0003085838, 3.1600738373)],
    ["O", (2.9857747423, 8.9570945142, 3.1600737733)],
    ["O", (5.9723690642, 5.9723696033, 3.1594610725)],
    ["O", (5.9732960489, 2.9857717265, 3.1600713997)],
    ["O", (5.9732976356, 0.0003062885, 3.1600713024)],
    ["O", (2.9857711850, 5.9732998192, 3.1600740469)],
    ["O", (2.9950544541, 2.9950623702, 3.1548459568)],
    ["O", (2.9950599166, -0.0089867212, 3.1548507549)],
    ["O", (0.0003060319, 8.9570959750, 3.1600745678)],
    ["O", (0.0003045736, 5.9732996331, 3.1600758940)],
    ["O", (-0.0089879078, 2.9950667318, 3.1548429191)],
    ["O", (-0.0089805134, -0.0089799254, 3.1548482730)],
    ["O", (5.9723706153, 8.9580251994, 3.1594625961)],
    ["O", (8.9580250454, 8.9580263132, 3.1594616543)],
    ["O", (1.4930495812, 1.4930407413, 6.6690780399)],
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


def run2(mf, thresh, frozen, params=None):
    mfcc = LNOCCSD_ISDF(mf, thresh=thresh, frozen=frozen)
    if isinstance(params, dict):
        mfcc.set(**params)
    mfcc.kernel()
    return mfcc


for qr_cutoff in [1e-3, 3e-4, 1e-4]:
    isdf = ISDF_Local(
        cell, with_robust_fitting=False, limited_memory=True, build_V_K_bunchsize=256
    )
    isdf.build(c=40, m=5, rela_cutoff=qr_cutoff, group=group)
    mf = scf.RHF(cell)
    mf.with_df = isdf
    mf.kernel()

    # LNO CCSD #

    for thresh in [1e-3, 1e-4, 1e-5, 1e-6]:
        # thresh = 0.0
        # mf.with_df = isdf
        print(" ************ LNO CCSD with threshold %e ************" % (thresh))
        params = {"lo_type": "pm", "no_type": "cim", "frag_lolist": "1o"}
        mfcc2 = run2(mf, thresh, None, params=params)
