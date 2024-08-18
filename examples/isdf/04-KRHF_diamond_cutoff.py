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

MOL_STRUCTURE = """
C     0.      0.      0.
C     0.8917  0.8917  0.8917
C     1.7834  1.7834  0.
C     2.6751  2.6751  0.8917
C     1.7834  0.      1.7834
C     2.6751  0.8917  2.6751
C     0.      1.7834  1.7834
C     0.8917  2.6751  2.6751
"""

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
boxlen = 3.57371000
prim_a = np.array([[boxlen, 0.0, 0.0], [0.0, boxlen, 0.0], [0.0, 0.0, boxlen]])

C_ARRAY = [
    25
]  ## if rela_cutoff_QRCP is set, then c is used to when performing random projection, which can be relative large.
RELA_QR = [1e-3]
SuperCell_ARRAY = [
    [1, 1, 8],
]
Ke_CUTOFF = [70]
Basis = ["gth-dzvp"]
prim_partition = [[0, 1], [2, 3], [4, 5], [6, 7]]

if __name__ == "__main__":

    prim_a = np.array([[boxlen, 0.0, 0.0], [0.0, boxlen, 0.0], [0.0, 0.0, boxlen]])
    pseudo = "gth-pade"

    for supercell in SuperCell_ARRAY:
        for basis in Basis:
            for ke_cutoff in Ke_CUTOFF:

                DM_CACHED = None

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

                for c, rela_qr in list(zip(C_ARRAY, RELA_QR)):

                    print("--------------------------------------------")
                    print(
                        "C = %d, QR=%f, supercell = %s, kc_cutoff = %d, basis = %s"
                        % (c, rela_qr, str(supercell), ke_cutoff, basis)
                    )

                    ### create the isdf object ###

                    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    pbc_isdf_info = isdf_local_k.PBC_ISDF_Info_Quad_K(
                        cell,
                        kmesh=nk,
                        with_robust_fitting=True,
                        rela_cutoff_QRCP=rela_qr,
                        direct=True,
                        limited_memory=True,
                        build_K_bunchsize=128,  ## NOTE:control the memory cost in building K
                        # use_occ_RI_K=False
                    )
                    pbc_isdf_info.verbose = 4
                    pbc_isdf_info.build_IP_local(c=c, m=5, group=prim_partition)
                    pbc_isdf_info.set_build_K_cutoff(
                        rela_cutoff=1e-5, abs_cutoff=1e-6
                    )  # rela_cutoff is useless currently
                    print(
                        "effective c = ", float(pbc_isdf_info.naux) / pbc_isdf_info.nao
                    )
                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    print(isdf_jk._benchmark_time(t1, t2, "build ISDF", pbc_isdf_info))

                    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                    mf = scf.KRHF(cell, kpts)
                    mf.with_df = pbc_isdf_info
                    mf.max_cycle = 8
                    mf.conv_tol = 1e-8
                    mf.conv_tol_grad = 1e-3
                    mf.init_guess = "atom"
                    # mf.kernel()
                    if DM_CACHED is not None:
                        mf.kernel(DM_CACHED)
                    else:
                        mf.kernel()
                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

                    # dm0 = mf.init_guess_by_atom()
                    # vj, vk = pbc_isdf_info.get_jk(dm0)
                    # print("vj.shape = ", vj.shape)
                    # print("vk.shape = ", vk.shape)
                    # vj.tofile("vj_26.dat")
                    # vk.tofile("vk_26.dat")

                    print(isdf_jk._benchmark_time(t1, t2, "KRHF_bench", mf))
                    # DM_CACHED = mf.make_rdm1()
