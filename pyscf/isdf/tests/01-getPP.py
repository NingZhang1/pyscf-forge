import numpy as np
from pyscf import lib

from pyscf.lib.parameters import BOHR
from pyscf.pbc import df

# backend to test #

import pyscf.isdf.BackEnd._config as config

# config.backend("scipy")
config.backend("torch")
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

ABS = BACKEND._absolute
MAX = BACKEND._maximum
ToTENSOR = BACKEND._toTensor

# isdf util #

from pyscf.isdf.isdf_tools_Tsym import _kmesh_to_Kpoints, _1e_operator_gamma2k
from pyscf.isdf import isdf_tools_cell

#############################

ke_cutoff = 70
basis = "gth-szv"
# basis = "gth-dzvp"

boxlen = 3.57371000
prim_a = np.array([[boxlen, 0.0, 0.0], [0.0, boxlen, 0.0], [0.0, 0.0, boxlen]])
atm = [
    ["C", (0.0, 0.0, 0.0)],
    ["C", (0.8934275, 0.8934275, 0.8934275)],
    # ["C", (1.786855, 1.786855, 0.0)],
    # ["C", (2.6802825, 2.6802825, 0.8934275)],
    # ["C", (1.786855, 0.0, 1.786855)],
    # ["C", (2.6802825, 0.8934275, 2.6802825)],
    # ["C", (0.0, 1.786855, 1.786855)],
    # ["C", (0.8934275, 2.6802825, 2.6802825)],
]

kmeshes = [
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, 3],
    [1, 1, 4],
    [1, 2, 2],
    [1, 2, 3],
    [1, 3, 3],
    [2, 3, 3],
]
VERBOSE = 0

prim_cell = isdf_tools_cell.build_supercell(
    atm,
    prim_a,
    Ls=[1, 1, 1],
    ke_cutoff=ke_cutoff,
    basis=basis,
    pseudo="gth-pade",
    verbose=VERBOSE,
)

prim_mesh = prim_cell.mesh

for kmesh in kmeshes:

    mesh = [int(k * x) for k, x in zip(kmesh, prim_mesh)]
    print("kmesh:", kmesh, "mesh:", mesh)

    kpts = prim_cell.make_kpts(kmesh)

    from pyscf.pbc.df.fft import FFTDF

    scaled_kpts = prim_cell.get_scaled_kpts(kpts)
    df_tmp = FFTDF(prim_cell, kpts)
    PP_kpts = df_tmp.get_pp()

    cell = isdf_tools_cell.build_supercell(
        atm,
        prim_a,
        Ls=kmesh,
        ke_cutoff=ke_cutoff,
        mesh=mesh,
        basis=basis,
        pseudo="gth-pade",
        verbose=VERBOSE,
    )

    df_tmp2 = FFTDF(cell, np.zeros((1, 3)))

    PP = df_tmp2.get_pp(kpts=np.zeros((1, 3)))[0]

    PP = ToTENSOR(PP)
    PP_kpts = ToTENSOR(PP_kpts)
    PP_kpts2 = _1e_operator_gamma2k(cell, kmesh, PP)

    # assert MAX(ABS(PP_kpts - PP_kpts2)) < 1e-9

    print("diff = ", MAX(ABS(PP_kpts - PP_kpts2)))
    print("PASS")
