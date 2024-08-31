# backend to test #

import pyscf.isdf.BackEnd._config as config

config.disable_fftw()
# config.backend("numpy")
# config.backend("scipy")
config.backend("torch")
# config.backend("torch_gpu")
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

ZEROS = BACKEND._zeros
FLOAT64 = BACKEND.FLOAT64
TAKE = BACKEND._take
EINSUM_IK_JK_IJK = BACKEND._einsum_ik_jk_ijk
DOT = BACKEND._dot
MAX = BACKEND._maximum
ABS = BACKEND._absolute

# sys and pyscf #

import numpy as np
from pyscf import lib

from pyscf.lib.parameters import BOHR
from pyscf.pbc import df

# isdf util #

from pyscf.isdf.isdf_tools_Tsym import _kmesh_to_Kpoints, _1e_operator_gamma2k
from pyscf.isdf import isdf_tools_cell
from pyscf.isdf.isdf import ISDF
from pyscf.isdf.isdf_local import ISDF_Local
from pyscf.isdf.isdf_tools_local import (
    _pack_aoR_holder,
)

#############################

ke_cutoff = 32
basis = "gth-szv"

boxlen = 3.57371000
prim_a = np.array([[boxlen, 0.0, 0.0], [0.0, boxlen, 0.0], [0.0, 0.0, boxlen]])
atm = [
    ["C", (0.0, 0.0, 0.0)],
    ["C", (0.8934275, 0.8934275, 0.8934275)],
    ["C", (1.786855, 1.786855, 0.0)],
    ["C", (2.6802825, 2.6802825, 0.8934275)],
    # ["C", (1.786855, 0.0, 1.786855)],
    # ["C", (2.6802825, 0.8934275, 2.6802825)],
    # ["C", (0.0, 1.786855, 1.786855)],
    # ["C", (0.8934275, 2.6802825, 2.6802825)],
]

kmeshes = [
    # [1, 1, 1],
    # [1, 1, 2],
    [1, 1, 4],
    # [1, 2, 2],
    # [2, 2, 2],
]  # -44.20339674 and -88.67568935
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

# NOTE: weird thing, when [[0, 1], [2, 3]] + 1 x 1 x 4 unit cell, the error is very large (1e-5)
# however, for [[0,1,2,3]] (5e-7) and [[0], [1], [2], [3]] (~1e-8) the error is not that large

# prim_group = [[0, 1], [2, 3], [4, 5], [6, 7]]
# prim_group = [[0], [1], [2], [3], [4], [5], [6], [7]]
# prim_group = [[0, 1, 2, 3]]
prim_group = [[0, 1], [2, 3]]
# prim_group = [[0], [1], [2], [3]]
# prim_group = [[0, 1, 2, 3]]

prim_mesh = prim_cell.mesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_scatter(x, y, z, title="3D Scatter Plot"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(x, y, z, c=z, cmap="viridis")

    plt.colorbar(scatter)

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    plt.title(title)

    plt.show()


for kmesh in kmeshes:

    mesh = [int(k * x) for k, x in zip(kmesh, prim_mesh)]
    print("kmesh:", kmesh, "mesh:", mesh)

    kpts = prim_cell.make_kpts(kmesh)

    cell, group = isdf_tools_cell.build_supercell_with_partition(
        atm,
        prim_a,
        Ls=kmesh,
        ke_cutoff=ke_cutoff,
        partition=prim_group,
        mesh=mesh,
        basis=basis,
        pseudo="gth-pade",
        verbose=VERBOSE,
    )
    cell.max_memory = 200
    print("group:", group)

    isdf = ISDF_Local(
        cell,
        with_robust_fitting=True,
        limited_memory=True,
        build_V_K_bunchsize=56,
        direct=True,
    )
    isdf.build(c=50, m=5, rela_cutoff=1e-5, group=group, global_IP_selection=False)

    ## groupID ##

    naux = isdf.naux
    ngrids = isdf.ngrids

    aux_basis_tensor = ZEROS((naux, ngrids), dtype=FLOAT64)

    iaux = 0
    igrid = 0
    for aux_basis in isdf.aux_basis:
        aux_basis_tensor[
            iaux : iaux + aux_basis.shape[0], igrid : igrid + aux_basis.shape[1]
        ] = aux_basis
        iaux += aux_basis.shape[0]
        igrid += aux_basis.shape[1]
    grid_ordering = isdf.gridID_ordering
    aux_basis_tensor_ordered = ZEROS((naux, ngrids), dtype=FLOAT64)
    # TAKE(aux_basis_tensor, grid_ordering, 1, out=aux_basis_tensor_ordered)
    aux_basis_tensor_ordered[:, grid_ordering] = aux_basis_tensor

    ## check not ordered ##

    aoRg_packed = _pack_aoR_holder(isdf.aoRg, isdf.nao)
    aoR_packed = _pack_aoR_holder(isdf.aoR, isdf.nao)
    print(aoRg_packed.aoR.shape)
    print(aoR_packed.aoR.shape)
    print(aux_basis_tensor.shape)

    aoPairRg = EINSUM_IK_JK_IJK(aoRg_packed.aoR, aoRg_packed.aoR)
    aoPairR = EINSUM_IK_JK_IJK(aoR_packed.aoR, aoR_packed.aoR)
    aoPairR2 = DOT(aoPairRg.reshape(isdf.nao * isdf.nao, -1), aux_basis_tensor).reshape(
        isdf.nao, isdf.nao, -1
    )
    diff = MAX(ABS(aoPairR - aoPairR2))
    print(diff)

    continue

    for ID, IPs in enumerate(isdf.IP_group):
        coords_IP = isdf.coords[IPs]
        x = coords_IP[:, 0]
        y = coords_IP[:, 1]
        z = coords_IP[:, 2]

        plot_3d_scatter(x, y, z, "group %d" % (ID))

    for atmID, IPs in enumerate(isdf.partition):
        coords_IP = isdf.coords[IPs]
        x = coords_IP[:, 0]
        y = coords_IP[:, 1]
        z = coords_IP[:, 2]

        plot_3d_scatter(x, y, z, "atm %d" % (atmID))
