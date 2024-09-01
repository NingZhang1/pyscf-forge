# backend to test #

import pyscf.isdf.BackEnd._config as config

config.disable_fftw()
# config.backend("numpy")
# config.backend("scipy")
config.backend("torch")
# config.backend("torch_gpu")
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

MAX = BACKEND._maximum
ABS = BACKEND._absolute

# sys and pyscf #

from itertools import product
import numpy as np
from pyscf import lib

from pyscf.lib.parameters import BOHR
from pyscf.pbc import df

# isdf util #

from pyscf.isdf.isdf_tools_Tsym import _kmesh_to_Kpoints, _1e_operator_gamma2k
from pyscf.isdf import isdf_tools_cell
from pyscf.isdf.isdf import ISDF
from pyscf.isdf.isdf_local import ISDF_Local
from pyscf.isdf.isdf_local_k import ISDF_Local_K
from pyscf.isdf.isdf_tools_local import (
    _pack_aoR_holder,
)

#############################

ke_cutoff = 70
basis = "gth-dzvp"

boxlen = 3.57371000
prim_a = np.array([[boxlen, 0.0, 0.0], [0.0, boxlen, 0.0], [0.0, 0.0, boxlen]])
atm = [
    ["C", (0.0, 0.0, 0.0)],
    ["C", (0.8934275, 0.8934275, 0.8934275)],
    ["C", (1.786855, 1.786855, 0.0)],
    ["C", (2.6802825, 2.6802825, 0.8934275)],
    ["C", (1.786855, 0.0, 1.786855)],
    ["C", (2.6802825, 0.8934275, 2.6802825)],
    ["C", (0.0, 1.786855, 1.786855)],
    ["C", (0.8934275, 2.6802825, 2.6802825)],
]

kmeshes = [
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, 3],
    [1, 1, 4],
    [1, 2, 2],
    [2, 2, 2],
    # [1, 1, 9],
    [1, 1, 4],
    [1, 3, 3],  # NOTE: problematic
    [3, 3, 3],
    [4, 4, 4],
]
VERBOSE = 10

prim_cell = isdf_tools_cell.build_supercell(
    atm,
    prim_a,
    Ls=[1, 1, 1],
    ke_cutoff=ke_cutoff,
    basis=basis,
    pseudo="gth-pade",
    verbose=VERBOSE,
)

# prim_group = [[0, 1], [2, 3], [4, 5], [6, 7]]
prim_group = [[0], [1], [2], [3], [4], [5], [6], [7]]

prim_mesh = prim_cell.mesh

for kmesh in kmeshes[-4:-3]:

    mesh = [int(k * x) for k, x in zip(kmesh, prim_mesh)]
    print("kmesh:", kmesh, "mesh:", mesh)
    kpts = prim_cell.make_kpts(kmesh)
    print(kpts)
    prim_cell.max_memory = 200

    isdf = ISDF_Local_K(
        prim_cell,
        kmesh=kmesh,
        direct=False,
        with_robust_fitting=False,
        limited_memory=True,
        build_V_K_bunchsize=128,
    )
    isdf.build(c=30, m=5, rela_cutoff=1e-4, group=prim_group)
    # isdf._use_super_pp = True
    # isdf._use_FFTDF_pp = True

    # NOTE: debugging code #

    aoRg_packed = _pack_aoR_holder(isdf.aoRg, isdf.nao).todense(isdf.nao)
    aoRg1_packed = _pack_aoR_holder(isdf.aoRg1, isdf.naoPrim).todense(isdf.naoPrim)
    nIP_prim = aoRg_packed.shape[1]

    # check the symmetry #

    print("aoRg_packed", aoRg_packed.shape)
    print("aoRg1_packed", aoRg1_packed.shape)
    for ix, iy, iz in product(range(kmesh[0]), range(kmesh[1]), range(kmesh[2])):
        ix2 = (kmesh[0] - ix) % kmesh[0]
        iy2 = (kmesh[1] - iy) % kmesh[1]
        iz2 = (kmesh[2] - iz) % kmesh[2]
        loc = ix * kmesh[1] * kmesh[2] + iy * kmesh[2] + iz
        loc2 = ix2 * kmesh[1] * kmesh[2] + iy2 * kmesh[2] + iz2
        mat1 = aoRg1_packed[:, loc * nIP_prim : (loc + 1) * nIP_prim]
        mat2 = aoRg_packed[loc2 * isdf.naoPrim : (loc2 + 1) * isdf.naoPrim, :]
        diff = MAX(ABS(mat1 - mat2))
        print("diff", diff, " at ", ix, iy, iz)
    print("aoRg_packed", aoRg_packed.shape)
    print("aoRg1_packed", aoRg1_packed.shape)
    # exit(1)

    from pyscf.pbc.scf.khf import KRHF

    mf = KRHF(prim_cell, kpts=kpts)
    mf.with_df = isdf
    mf.max_cycle = 100
    mf.conv_tol = 1e-8
    mf.conv_tol_grad = 1e-3
    # mf.kpts.build()
    mf.kernel()

    # benchmark #
    # mf = scf.RHF(cell)
    # mf.kernel()
