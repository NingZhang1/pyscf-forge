import numpy as np
from pyscf import lib
from pyscf.gto.mole import *

from pyscf.isdf import isdf_tools_cell
from pyscf.isdf import isdf_local_k
from pyscf.isdf import isdf_jk
from pyscf.isdf import isdf_local

from pyscf.lib.parameters import BOHR
from pyscf.pbc import df

MOL_STRUCTURE = '''
                   C     0.      0.      0.
                   C     0.8917  0.8917  0.8917
                   C     1.7834  1.7834  0.
                   C     2.6751  2.6751  0.8917
                   C     1.7834  0.      1.7834
                   C     2.6751  0.8917  2.6751
                   C     0.      1.7834  1.7834
                   C     0.8917  2.6751  2.6751
                '''

#### NOTE: a full tests on combinations of parameters ####
                
C_ARRAY = [30]
RELA_CUTOFF = [1e-3]
SuperCell_ARRAY = [
    [1, 1, 1],
]

Ke_CUTOFF = [70]
boxlen = 3.5668
Basis = ['gth-dzvp']

PARTITION = [
    [[0,1],[2,3],[4,5],[6,7]],
]

if __name__ == '__main__':

    boxlen = 3.57371000
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    atm = [
        ['C', (0.        , 0.        , 0.    )],
        ['C', (0.8934275 , 0.8934275 , 0.8934275)],
        ['C', (1.786855  , 1.786855  , 0.    )],
        ['C', (2.6802825 , 2.6802825 , 0.8934275)],
        ['C', (1.786855  , 0.        , 1.786855)],
        ['C', (2.6802825 , 0.8934275 , 2.6802825)],
        ['C', (0.        , 1.786855  , 1.786855)],
        ['C', (0.8934275 , 2.6802825 , 2.6802825)],
    ]
    
    for supercell in SuperCell_ARRAY:
        ke_cutoff = Ke_CUTOFF[0]
        for partition in PARTITION:   ## test different partition of atoms
            for basis in Basis:
                for c, rela_cutoff in zip(C_ARRAY, RELA_CUTOFF):
                # for c in C_ARRAY:
                    print('--------------------------------------------')
                    print('C = %.2e, supercell = %s, kc_cutoff = %d, basis = %s, partition = %s' % (
                        c, str(supercell), ke_cutoff, basis, partition))

                    prim_cell = isdf_tools_cell.build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo="gth-pade", verbose=4)
                    prim_mesh = prim_cell.mesh
                    print("prim_mesh = ", prim_mesh)
            
                    mesh = [supercell[0] * prim_mesh[0], supercell[1] * prim_mesh[1], supercell[2] * prim_mesh[2]]
                    mesh = np.array(mesh, dtype=np.int32)
            
                    cell, supercell_group = isdf_tools_cell.build_supercell_with_partition(atm, prim_a, partition=partition, Ls = supercell, ke_cutoff=ke_cutoff, mesh=mesh, basis=basis, pseudo="gth-pade", verbose=15)

                    #cell.incore_anyway = False
                    #cell.max_memory    = 200   # force to call with_df.get_jk

                    t1 = (lib.logger.process_clock(),lib.logger.perf_counter())
                        
                    pbc_isdf_info = isdf_local.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, direct=False, rela_cutoff_QRCP=rela_cutoff)
                    pbc_isdf_info.build_IP_local(c=c, group=supercell_group, Ls=[supercell[0]*4, supercell[1]*4, supercell[2]*4])
                    print("pbc_isdf_info.naux = ", pbc_isdf_info.naux) 
                    print("effective c = ", float(pbc_isdf_info.naux) / pbc_isdf_info.nao) 
                    pbc_isdf_info.build_auxiliary_Coulomb()
                                                
                    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())

                    eri2 = pbc_isdf_info.get_eri(compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)

                    ### get eri ###
                    
                    mydf_eri = df.FFTDF(cell)
                    eri      = mydf_eri.get_eri(compact=False).reshape(cell.nao, cell.nao, cell.nao, cell.nao)
                    
                    diff_max = np.max(np.abs(eri - eri2))
                    print("diff_max = ", diff_max)