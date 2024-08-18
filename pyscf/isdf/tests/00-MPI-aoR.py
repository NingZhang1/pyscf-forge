from functools import reduce
import numpy as np
from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member

from pyscf.isdf import isdf_tools_cell
from pyscf.isdf import isdf_local_k
from pyscf.isdf import isdf_jk
from pyscf.isdf import isdf_local_MPI
from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, allgather, bcast

MOL_STRUCTURE = """
Ti            2.3246330643        2.3246330643        1.4853414945
Ti            0.0000000000        0.0000000000       -0.0000000000
O             0.9065353261        3.7427308025        1.4853414945
O             3.7427308025        0.9065353261        1.4853414945
O             1.4180977382        1.4180977382        0.0000000000
O             3.2311683903        3.2311683903        0.0000000000
"""

atm = [
    ["Ti", (2.3246330643, 2.3246330643, 1.4853414945)],
    ["Ti", (0.0000000000, 0.0000000000, 0.0000000000)],
    ["O ", (0.9065353261, 3.7427308025, 1.4853414945)],
    ["O ", (3.7427308025, 0.9065353261, 1.4853414945)],
    ["O ", (1.4180977382, 1.4180977382, 0.0000000000)],
    ["O ", (3.2311683903, 3.2311683903, 0.0000000000)],
]
boxlen = [4.6492659759, 4.6492659759, 2.9706828877]

C_ARRAY = [
    15,
    25,
    30,
    35,
]  ## if rela_cutoff_QRCP is set, then c is used to when performing random projection, which can be relative large.
RELA_QR = [1e-2, 1e-3, 2e-4, 1e-4]
SuperCell_ARRAY = [
    # [2,2,1],
    [4, 4, 2],
]
Ke_CUTOFF = [192]
Basis = ["gth-cc-tzvp-Ye"]

prim_partition = [[0], [1], [2], [3], [4], [5]]

if __name__ == "__main__":

    prim_a = np.array(
        [[boxlen[0], 0.0, 0.0], [0.0, boxlen[1], 0.0], [0.0, 0.0, boxlen[2]]]
    )
    pseudo = "gth-hf-rev"

    for supercell in SuperCell_ARRAY:
        for basis in Basis:
            for ke_cutoff in Ke_CUTOFF:

                DM_CACHED = None

                from pyscf.gto.basis import parse_nwchem

                fbas = "basis.dat"
                atms = ["O", "Ti"]
                basis = {atm: parse_nwchem.load(fbas, atm) for atm in atms}

                prim_cell = isdf_tools_cell.build_supercell(
                    atm,
                    prim_a,
                    Ls=[1, 1, 1],
                    ke_cutoff=ke_cutoff,
                    basis=basis,
                    pseudo=pseudo,
                    spin=0,
                    verbose=10,
                )
                cell = prim_cell

                ### perform scf ###

                from pyscf.pbc import scf, dft
                from pyscf.pbc.dft import multigrid

                nk = supercell
                kpts = cell.make_kpts(nk)

                if rank != 0:
                    cell.verbose = 0
                    cell.build()

                for c, rela_qr in list(zip(C_ARRAY, RELA_QR))[-1:]:

                    if rank == 0:
                        print("--------------------------------------------")
                        print(
                            "C = %d, QR=%f, supercell = %s, kc_cutoff = %d, basis = %s"
                            % (c, rela_qr, str(supercell), ke_cutoff, basis)
                        )

                    ### create the isdf object ###

                    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    pbc_isdf_info = isdf_local_MPI.PBC_ISDF_Info_Quad_K_MPI(
                        cell,
                        kmesh=nk,
                        rela_cutoff_QRCP=rela_qr,
                        limited_memory=True,
                        build_K_bunchsize=36,  ## NOTE:control the memory cost in building K
                    )
                    pbc_isdf_info.verbose = 10
                    pbc_isdf_info.set_group()
                    pbc_isdf_info.build_partition_aoR()

                    #### dump attributes ####

                    filename = "attributes_%d_%d.dat" % (comm_size, rank)
                    pbc_isdf_info.dump_attributes(
                        filename=filename,
                        attr_lst=[
                            "partition_prim",
                            "partition",
                        ],
                    )
                    filename = "aoValues_%d_%d" % (comm_size, rank)
                    pbc_isdf_info.dump_aoR(filename)

                    if rank == 0:
                        pbc_isdf_info.build_partition_aoR()
                        filename = "attributes_benchmark.dat"
                        pbc_isdf_info.dump_attributes(
                            filename=filename,
                            attr_lst=[
                                "partition_prim",
                                "partition",
                            ],
                        )
                        filename = "aoValues_benchmark"
                        pbc_isdf_info.dump_aoR(filename)

                    exit(1)
