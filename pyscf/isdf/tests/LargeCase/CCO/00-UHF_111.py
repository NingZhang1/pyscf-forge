### read in configuration ###
from pyscf.isdf.tests.common.test_config import get_args

args = get_args()
direct = args.direct
with_robust_fitting = args.with_robust_fitting
backend = args.backend.lower()
assert backend in ["numpy", "scipy", "torch", "torch_gpu"]
aoR_cutoff = args.aoR_cutoff
bunchsize = args.bunchsize
robust_fitting_tune = args.robust_fitting_tune

print(" ----- configuration ----- ")
print("backend             = ", backend)
print("direct              = ", direct)
print("with_robust_fitting = ", with_robust_fitting)
print("aoR_cutoff          = ", aoR_cutoff)
print("bunchsize           = ", bunchsize)
print("robust_fitting_tune = ", robust_fitting_tune)
print(" ------------------------- ")

# backend to test #

import pyscf.isdf.BackEnd._config as config

config.disable_fftw()
config.backend(backend)
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

# other module #

import numpy as np
from pyscf import lib

from pyscf.isdf import isdf_local
from pyscf.isdf import misc
from pyscf.isdf import isdf_tools_cell

from pyscf.lib.parameters import BOHR

# CCO #

prim_a = (
    np.array(
        [
            [14.572056092 / 2, 0.000000000, 0.000000000],
            [0.000000000, 14.572056092 / 2, 0.000000000],
            [0.000000000, 0.000000000, 6.010273939],
        ]
    )
    * BOHR
)
atm = [
    ["Ca", (0.000000, 0.000000, 0.000000)],
    ["Cu", (1.927800, 1.927800, 1.590250)],
    ["O", (0.000000, 1.927800, 1.590250)],
    ["O", (1.927800, 0.000000, 1.590250)],
    # ["Ca", (0.000000, 3.855600, 0.000000)],
    # ["Cu", (1.927800, 5.783400, 1.590250)],
    # ["O", (0.000000, 5.783400, 1.590250)],
    # ["O", (1.927800, 3.855600, 1.590250)],
    # ["Ca", (3.855600, 0.000000, 0.000000)],
    # ["Cu", (5.783400, 1.927800, 1.590250)],
    # ["O", (3.855600, 1.927800, 1.590250)],
    # ["O", (5.783400, 0.000000, 1.590250)],
    # ["Ca", (3.855600, 3.855600, 0.000000)],
    # ["Cu", (5.783400, 5.783400, 1.590250)],
    # ["O", (3.855600, 5.783400, 1.590250)],
    # ["O", (5.783400, 3.855600, 1.590250)],
]

C_ARRAY = [15, 20, 25, 30, 35]
RELA_CUTOFF = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
SuperCell_ARRAY = [
    [1, 1, 1],
    [2, 2, 1],
    [4, 4, 1],
]
Ke_CUTOFF = [192, 256]
Basis = ["gth-dzvp"]

PARTITION = [
    [
        [0],
        [1],
        [2],
        [3],
        # [4],
        # [5],
        # [6],
        # [7],
        # [8],
        # [9],
        # [10],
        # [11],
        # [12],
        # [13],
        # [14],
        # [15],
    ]
]


if __name__ == "__main__":

    for supercell in SuperCell_ARRAY:
        DM_CACHED = None
        for ke_cutoff in Ke_CUTOFF:
            partition = PARTITION[0]
            for _basis_ in Basis:

                from pyscf.gto.basis import parse_nwchem

                fbas = "basis2.dat"
                atms = ["O", "Cu", "Ca"]
                basis = {atm: parse_nwchem.load(fbas, atm) for atm in atms}

                pseudo = {"Cu": "gth-pbe-q19", "O": "gth-pbe", "Ca": "gth-pbe"}

                prim_cell = isdf_tools_cell.build_supercell(
                    atm,
                    prim_a,
                    spin=1,
                    Ls=[1, 1, 1],
                    ke_cutoff=ke_cutoff,
                    basis=basis,
                    pseudo=pseudo,
                    verbose=10,
                )
                prim_mesh = prim_cell.mesh

                mesh = [
                    supercell[0] * prim_mesh[0],
                    supercell[1] * prim_mesh[1],
                    supercell[2] * prim_mesh[2],
                ]
                mesh = np.array(mesh, dtype=np.int32)

                cell, supercell_group = isdf_tools_cell.build_supercell_with_partition(
                    atm,
                    prim_a,
                    spin=np.product(supercell) % 2,
                    partition=partition,
                    Ls=supercell,
                    ke_cutoff=ke_cutoff,
                    mesh=mesh,
                    basis=basis,
                    pseudo=pseudo,
                    verbose=10,
                )

                cell.incore_anyway = False
                cell.max_memory = 200  # force to call with_df.get_jk

                for c, rela_cutoff in zip(C_ARRAY, RELA_CUTOFF):

                    print("--------------------------------------------")
                    print(
                        "C = %.2e, supercell = %s, kc_cutoff = %d, basis = %s, partition = %s"
                        % (c, str(supercell), ke_cutoff, basis, partition)
                    )

                    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())

                    pbc_isdf_info = isdf_local.ISDF_Local(
                        cell,
                        with_robust_fitting=with_robust_fitting,
                        direct=direct,
                        limited_memory=True,
                        build_V_K_bunchsize=bunchsize,
                        aoR_cutoff=aoR_cutoff,
                    )
                    pbc_isdf_info.build(
                        c=c, m=5, rela_cutoff=rela_cutoff, group=supercell_group
                    )
                    pbc_isdf_info.force_translation_symmetry(
                        supercell
                    )  # force symmetry, leading to Ferri phase
                    print("pbc_isdf_info.naux = ", pbc_isdf_info.naux)
                    print(
                        "effective c = ", float(pbc_isdf_info.naux) / pbc_isdf_info.nao
                    )

                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

                    print(
                        misc._benchmark_time(
                            t1, t2, "build_isdf", pbc_isdf_info, pbc_isdf_info.rank
                        )
                    )

                    ### perform scf ###

                    from pyscf.pbc import scf

                    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    mf = scf.UHF(cell)
                    mf = scf.addons.smearing_(mf, sigma=0.2, method="fermi")
                    mf.with_df = pbc_isdf_info
                    mf.max_cycle = 64
                    mf.conv_tol = 1e-7
                    mf.conv_tol_grad = 1e-2
                    pbc_isdf_info.direct_scf = mf.direct_scf
                    if DM_CACHED is not None:
                        mf.kernel(DM_CACHED)
                    else:
                        mf.kernel()
                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    print(
                        misc._benchmark_time(
                            t1, t2, "scf_isdf", pbc_isdf_info, pbc_isdf_info.rank
                        )
                    )

                    DM_CACHED = (
                        mf.make_rdm1()
                    )  # cache the density matrix as the initial guess

                    print("robust_fitting_tune = ", robust_fitting_tune)
                    print("with_robust_fitting = ", with_robust_fitting)
                    if robust_fitting_tune and not with_robust_fitting:

                        print(" ---------- perform robust fitting ---------- ")
                        pbc_isdf_info.rebuild(True, True)  # direct anyway
                        mf = scf.UHF(cell)
                        mf = scf.addons.smearing_(mf, sigma=0.2, method="fermi")
                        mf.with_df = pbc_isdf_info
                        mf.max_cycle = 64
                        mf.conv_tol = 1e-7
                        mf.conv_tol_grad = 1e-2
                        mf.kernel(DM_CACHED)
                        t3 = (lib.logger.process_clock(), lib.logger.perf_counter())
                        print(
                            misc._benchmark_time(
                                t2,
                                t3,
                                "scf_isdf_robust",
                                pbc_isdf_info,
                                pbc_isdf_info.rank,
                            )
                        )

                    del mf
                    del pbc_isdf_info

        ### UDF benchmark ###
        mf = scf.UHF(cell).density_fit()
        mf = scf.addons.smearing_(mf, sigma=0.2, method="fermi")
        mf.max_cycle = 64
        mf.conv_tol = 1e-8
        mf.kernel(DM_CACHED)
