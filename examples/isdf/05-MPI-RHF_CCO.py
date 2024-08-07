from functools import reduce
import numpy as np
from pyscf import lib
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf.gto.mole import *

from pyscf.isdf import isdf_tools_cell
from pyscf.isdf import isdf_local_k
from pyscf.isdf import isdf_jk
from pyscf.isdf import isdf_local_MPI
from pyscf.isdf.isdf_tools_mpi import rank, comm, comm_size, allgather, bcast

if __name__ == '__main__':

    C = 15
    from pyscf.lib.parameters import BOHR
    from pyscf.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
    
    verbose = 6
    if rank != 0:
        verbose = 0
    
    prim_a = np.array(
                    [[14.572056092/2, 0.000000000, 0.000000000],
                     [0.000000000, 14.572056092/2, 0.000000000],
                     [0.000000000, 0.000000000,  6.010273939],]) * BOHR
    atm = [
['Cu1',	(1.927800,	1.927800,	1.590250)],
['O1',	(1.927800,	0.000000,	1.590250)],
['O1',	(0.000000,	1.927800,	1.590250)],
['Ca',	(0.000000,	0.000000,	0.000000)],
    ]
    
    from pyscf.gto.basis import parse_nwchem
    fbas="basis2.dat" 
    atms = ['O', 'Cu', "Ca"]
    basis = {atm:parse_nwchem.load(fbas, atm) for atm in atms}
    pseudo = {'Cu1': 'gth-pbe-q19', 'Cu2': 'gth-pbe-q19', 'O1': 'gth-pbe', 'Ca': 'gth-pbe'}
    ke_cutoff = 128 
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo=pseudo)
    prim_mesh = prim_cell.mesh
    KE_CUTOFF = 128
        
    prim_mesh = prim_cell.mesh    
    prim_partition = [[0], [1], [2], [3]]    
    
    Ls = [2, 2, 1]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, 
                                                     Ls=Ls,
                                                     basis=basis, pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    if rank == 0:
        print("group_partition = ", group_partition)
    
    pbc_isdf_info = isdf_local_MPI.PBC_ISDF_Info_Quad_MPI(cell, aoR_cutoff=1e-8, verbose=verbose, limited_memory=True, build_K_bunchsize=16)
    pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition)
    pbc_isdf_info.Ls = Ls
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    from pyscf.pbc import scf

    if comm_size > 1:
        comm.Barrier()

    mf = scf.RHF(cell)
    mf = scf.addons.smearing_(mf, sigma=0.2, method='fermi')
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 16
    mf.conv_tol  = 1e-8
    mf.kernel()