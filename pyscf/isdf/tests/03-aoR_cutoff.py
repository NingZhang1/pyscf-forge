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

import numpy as np
from pyscf import lib

from pyscf.lib.parameters import BOHR
from pyscf.pbc import df

# isdf util #

from pyscf.isdf.isdf_tools_local import (
    build_cutoff_info,
    get_partition,
    get_aoR,
    _pack_aoR_holder,
)
from pyscf.isdf import isdf_tools_cell

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

kmeshes = [[1, 1, 1], [1, 1, 2], [1, 1, 4], [1, 1, 8]]
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

prim_mesh = prim_cell.mesh

precision = 1e-8

for kmesh in kmeshes:

    mesh = [int(k * x) for k, x in zip(kmesh, prim_mesh)]
    Ls = [int(k * x) for k, x in zip(kmesh, [3, 3, 3])]
    print("kmesh:", kmesh, "mesh:", mesh)
    ngrids = np.prod(mesh)

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

    distance_matrix, AtmConnectionInfo = build_cutoff_info(cell, precision, ngrids)

    print(distance_matrix)

    for atmID, x in enumerate(AtmConnectionInfo):
        print("atm ID = ", atmID)
        print(x)

    ### partition ###

    from pyscf.pbc.df.fft import FFTDF

    df_tmp = FFTDF(cell)

    coords = df_tmp.grids.coords.reshape(-1, 3).copy()

    partition = get_partition(cell, coords, AtmConnectionInfo, Ls=Ls)

    for atmid, x in enumerate(partition):
        print("the number of grids around %2d is %4d" % (atmid, len(x)))

    # print(partition[0])

    ## aoR ##

    if np.prod(kmesh) > 2:
        continue

    grid_ordering = []
    for x in partition:
        grid_ordering.extend(x)
    grid_ordering = np.array(grid_ordering, dtype=np.int32)

    aoR_holders = get_aoR(
        cell,
        AtmConnectionInfoList=AtmConnectionInfo,
        coords=coords,
        partition=partition,
    )

    # benchmark #

    from pyscf.isdf.isdf_eval_gto import ISDF_eval_gto

    weight = np.sqrt(cell.vol / coords.shape[0])
    aoR = ISDF_eval_gto(cell, coords=coords[grid_ordering]) * weight

    aoR2 = _pack_aoR_holder(aoR_holders, cell.nao).aoR

    diff = MAX(ABS(aoR2 - aoR))
    print("diff = ", diff)
    assert diff < 1e-8
