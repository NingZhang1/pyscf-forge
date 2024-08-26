import numpy as np
from pyscf import lib

from pyscf.lib.parameters import BOHR
from pyscf.pbc import df

# backend to test #

import pyscf.isdf.BackEnd._config as config

config.backend("scipy")
# config.backend("torch")
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

ABS = BACKEND._absolute
MAX = BACKEND._maximum
TENSORTy = BACKEND.TENSORTy

print(TENSORTy)

# isdf utils #

from pyscf.isdf.isdf_eval_gto import ISDF_eval_gto
from pyscf.isdf import isdf_tools_cell

#############################

ke_cutoff = 70
boxlen = 3.5668
basis = "gth-szv"

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

    from pyscf.pbc.df.fft import FFTDF

    df_tmp2 = FFTDF(cell, np.zeros((1, 3)))
    coords = df_tmp2.grids.coords

    aoR = ISDF_eval_gto(cell, coords=coords)
    print(aoR.shape)

    assert isinstance(aoR, TENSORTy)
