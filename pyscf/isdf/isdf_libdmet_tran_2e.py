#!/usr/bin/env python
# Copyright 2014-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Ning Zhang <ningzhang1024@gmail.com>
#

############ sys module ############

import numpy, scipy
import numpy as np
import ctypes

############ pyscf module ############

from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo.incore import iden_coeffs
from pyscf.pbc import tools
from pyscf.pbc.lib import kpts_helper
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, unique
from pyscf import __config__
from pyscf.pbc.df.fft_ao2mo import _format_kpts, _iskconserv, _contract_compact
libisdf = lib.load_library('libisdf')

############ isdf utils ############

from pyscf.isdf.isdf_tools_local import aoR_Holder
from pyscf.isdf.isdf_jk          import _benchmark_time 
from pyscf.isdf.isdf_local_k     import PBC_ISDF_Info_Quad_K

############ PROFILING ############

cputime_restore  = 0.0
cputime_embR     = 0.0
cputime_embpairR = 0.0
cputime_ddot     = 0.0
cputime_buildVW  = 0.0


walltime_restore = 0.0
walltime_embR    = 0.0
walltime_embpairR= 0.0
walltime_ddot    = 0.0
walltime_buildVW = 0.0

def reset_profile_get_eri_emb_time():
    
    global cputime_restore, cputime_embR, cputime_embpairR, cputime_ddot, cputime_buildVW
    global walltime_restore, walltime_embR, walltime_embpairR, walltime_ddot, walltime_buildVW
    
    cputime_restore  = 0.0
    cputime_embR     = 0.0
    cputime_embpairR = 0.0
    cputime_ddot     = 0.0
    cputime_buildVW  = 0.0
    
    walltime_restore = 0.0
    walltime_embR    = 0.0
    walltime_embpairR= 0.0
    walltime_ddot    = 0.0
    walltime_buildVW = 0.0

def log_profile_buildK_time(mydf, use_mpi=False):
    
    global cputime_restore, cputime_embR, cputime_embpairR, cputime_ddot, cputime_buildVW
    global walltime_restore, walltime_embR, walltime_embpairR, walltime_ddot, walltime_buildVW
    
    log = lib.logger.Logger(mydf.stdout, mydf.verbose)
    
    if not use_mpi:
        log.info('CPU time for restoring ERI  : %12.8f', cputime_restore)
        log.info('CPU time for emb_R          : %12.8f', cputime_embR)
        log.info('CPU time for emb_pair_R     : %12.8f', cputime_embpairR)
        log.info('CPU time for ddot           : %12.8f', cputime_ddot)
        log.info('CPU time for build_VW       : %12.8f', cputime_buildVW)
        log.info('Wall time for restoring ERI : %12.8f', walltime_restore)
        log.info('Wall time for emb_R         : %12.8f', walltime_embR)
        log.info('Wall time for emb_pair_R    : %12.8f', walltime_embpairR)
        log.info('Wall time for ddot          : %12.8f', walltime_ddot)
        log.info('Wall time for build_VW      : %12.8f', walltime_buildVW)
    else:
        if rank == 0:
            log.info('CPU time for restoring ERI  : %12.8f', cputime_restore)
            log.info('CPU time for emb_R          : %12.8f', cputime_embR)
            log.info('CPU time for emb_pair_R     : %12.8f', cputime_embpairR)
            log.info('CPU time for ddot           : %12.8f', cputime_ddot)
            log.info('CPU time for build_VW       : %12.8f', cputime_buildVW)
            log.info('Wall time for restoring ERI : %12.8f', walltime_restore)
            log.info('Wall time for emb_R         : %12.8f', walltime_embR)
            log.info('Wall time for emb_pair_R    : %12.8f', walltime_embpairR)
            log.info('Wall time for ddot          : %12.8f', walltime_ddot)
            log.info('Wall time for build_VW      : %12.8f', walltime_buildVW)
        comm.Barrier()

###################################

def _aoR_full_col(mydf):
    '''
    return aoR[:, :ngrid_prim] for the supercell system
    '''

    assert isinstance(mydf, PBC_ISDF_Info_Quad_K)

    fn_pack = getattr(libisdf, "_Pack_Matrix_SparseRow_DenseCol", None)
    assert fn_pack is not None
    
    prim_cell  = mydf.primCell
    prim_mesh  = prim_cell.mesh
    prim_ngrid = np.prod(prim_mesh)
    prim_natm  = mydf.natmPrim
    
    assert len(mydf.aoR) == prim_natm
    
    res = np.zeros((mydf.nao, prim_ngrid), dtype=np.float64)
    
    for i in range(prim_natm):
        aoR_i               = mydf.aoR[i]
        ao_involved_i       = aoR_i.ao_involved
        nao_i               = aoR_i.aoR.shape[0]
        global_grid_begin_i = aoR_i.global_gridID_begin
        ngrid_i             = aoR_i.aoR.shape[1]
                
        fn_pack(
            res.ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_int(res.shape[0]),
            ctypes.c_int(res.shape[1]),
            aoR_i.aoR.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_i),
            ctypes.c_int(ngrid_i),
            ao_involved_i.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(global_grid_begin_i),
            ctypes.c_int(global_grid_begin_i+ngrid_i)
        )
                
    return res

def _aoRg_full_col(mydf):
    '''
    return aoR[:, :ngrid_prim] for the supercell system
    '''

    assert isinstance(mydf, PBC_ISDF_Info_Quad_K)

    fn_pack = getattr(libisdf, "_Pack_Matrix_SparseRow_DenseCol", None)
    assert fn_pack is not None
    
    prim_cell  = mydf.primCell
    prim_mesh  = prim_cell.mesh
    prim_ngrid = np.prod(prim_mesh)
    prim_natm  = mydf.natmPrim
    prim_nIP   = mydf.nIP_Prim
    
    assert len(mydf.aoR) == prim_natm
    
    res = np.zeros((mydf.nao, prim_nIP), dtype=np.float64)
    
    for i in range(mydf.natmPrim):
        aoRg_i            = mydf.aoRg[i]
        ao_involved_i     = aoRg_i.ao_involved
        nao_i             = aoRg_i.aoR.shape[0]
        global_IP_begin_i = aoRg_i.global_gridID_begin
        nIP_i             = aoRg_i.aoR.shape[1]
                
        fn_pack(
            res.ctypes.data_as(ctypes.c_void_p), 
            ctypes.c_int(res.shape[0]),
            ctypes.c_int(res.shape[1]),
            aoRg_i.aoR.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nao_i),
            ctypes.c_int(nIP_i),
            ao_involved_i.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(global_IP_begin_i),
            ctypes.c_int(global_IP_begin_i+nIP_i)
        )
                
    return res

######## copy from libdmet ########

def eri_restore(eri, symmetry, nemb):
    """
    Restore eri with given permutation symmetry.
    """
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    spin_pair = eri.shape[0]
    if spin_pair == 1:
        eri_res = ao2mo.restore(symmetry, eri[0].real, nemb)
    else:
        if symmetry == 4:
            nemb_pair = nemb*(nemb+1) // 2
            if eri.size == spin_pair * nemb_pair * nemb_pair:
                eri_res = eri.real.reshape(spin_pair, nemb_pair, nemb_pair)
            else:
                eri_res = np.empty((spin_pair, nemb_pair, nemb_pair))
                for s in range(spin_pair):
                    eri_res[s] = ao2mo.restore(symmetry, eri[s].real, nemb)
        elif symmetry == 1:
            if eri.size == spin_pair * nemb**4:
                eri_res = eri.real.reshape(spin_pair, nemb, nemb, nemb, nemb)
            else:
                eri_res = np.empty((spin_pair, nemb, nemb, nemb, nemb))
                for s in range(spin_pair):
                    eri_res[s] = ao2mo.restore(symmetry, eri[s].real, nemb)
        else:
            raise ValueError("Spin unrestricted ERI does not support 8-fold symmetry.")
    eri_res = np.asarray(eri_res, order='C')
    
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter()) 
    
    global cputime_restore, walltime_restore
    cputime_restore  += t2[0] - t1[0]
    walltime_restore += t2[1] - t1[1]
    
    return eri_res

def get_emb_eri_isdf(mydf, C_ao_emb:np.ndarray=None, symmetry=4):
    
    ''' 
    get eri for embedding system
    '''
    
    #### profile ####
    
    global cputime_restore,  cputime_embR,  cputime_embpairR,  cputime_ddot,  cputime_buildVW
    global walltime_restore, walltime_embR, walltime_embpairR, walltime_ddot, walltime_buildVW
    reset_profile_get_eri_emb_time()
    
    #### preprocess #### 
    
    assert isinstance(mydf, PBC_ISDF_Info_Quad_K)
    assert not mydf.direct
    
    if C_ao_emb.ndim == 2:
        C_ao_emb = C_ao_emb.reshape(1, *C_ao_emb.shape)
    assert C_ao_emb.ndim  == 3
    assert C_ao_emb.dtype == np.float64  ## supercell basis
    
    nspin, nao_full, nemb = C_ao_emb.shape
    
    print("nspin    = ", nspin)
    print("nao_full = ", nao_full)
    print("nemb     = ", nemb)
    
    supercell = mydf.cell
    print("supercell.nao = ", supercell.nao)
    assert supercell.nao == nao_full
    
    ngrid      = mydf.ngrids
    vol        = supercell.vol
    mesh_prim  = mydf.primCell.mesh
    ngrid_prim = np.prod(mesh_prim)
    nao_prim   = mydf.nao_prim
    nIP_prim   = mydf.nIP_Prim
    kmesh      = mydf.kmesh
    nkpts      = np.prod(kmesh)
    nIP        = mydf.naux
    
    with_robust_fitting = mydf.with_robust_fitting
    
    #eri = np.zeros((nspin*(nspin+1)//2, nemb, nemb, nemb, nemb), dtype=np.float64) ## the ordering of spin is aa, bb, ab
    eri = np.zeros((nspin*(nspin+1)//2, nemb**2, nemb**2), dtype=np.float64) ## the ordering of spin is aa, bb, ab
    
    ### emb values on grid and IPs ###
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
    
    emb_R = []
    emb_Rg= []
    for i in range(nspin):
        emb_R.append([])
        emb_Rg.append([])
    
    if with_robust_fitting:
        aoR_fullcol  = _aoR_full_col(mydf)
        assert aoR_fullcol.shape  == (nao_full, ngrid_prim)
    aoRg_fullcol = _aoRg_full_col(mydf)
    assert aoRg_fullcol.shape == (nao_full, nIP_prim)
    
    aoR_tmp  = np.zeros_like(aoR_fullcol)
    aoRg_tmp = np.zeros_like(aoRg_fullcol)
    
    for kx in range(kmesh[0]):
        for ky in range(kmesh[1]):
            for kz in range(kmesh[2]):
                                
                for ix in range(kmesh[0]):
                    for iy in range(kmesh[1]):
                        for iz in range(kmesh[2]):
                            
                            ILOC  = ix*kmesh[1]*kmesh[2] + iy*kmesh[2] + iz
                            ix_   = (ix + kx) % kmesh[0]
                            iy_   = (iy + ky) % kmesh[1]
                            iz_   = (iz + kz) % kmesh[2]
                            ILOC_ = ix_*kmesh[1]*kmesh[2] + iy_*kmesh[2] + iz_
                            
                            if with_robust_fitting:
                                aoR_tmp[ILOC_*nao_prim:(ILOC_+1)*nao_prim,:] = aoR_fullcol[ILOC*nao_prim:(ILOC+1)*nao_prim,:]
                            aoRg_tmp[ILOC_*nao_prim:(ILOC_+1)*nao_prim,:] = aoRg_fullcol[ILOC*nao_prim:(ILOC+1)*nao_prim,:]
                            
                for i in range(nspin):
                    if with_robust_fitting:
                        emb_R[i].append(np.dot(C_ao_emb[i].T, aoR_tmp))
                    emb_Rg[i].append(np.dot(C_ao_emb[i].T, aoRg_tmp))
                            
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    cputime_embR  = t2[0] - t1[0]
    walltime_embR = t2[1] - t1[1]
    
    ### V_R term ###
    
    #V_R = mydf.V_R
    #assert V_R.shape == (nIP_prim, ngrid)
        
    tmp_V = np.zeros((nspin, nIP, nemb*nemb), dtype=np.float64)
    
    def _construct_tmp_V_W(Is_V=False):

        global cputime_embpairR, walltime_embpairR
        global cputime_ddot, walltime_ddot

        tmp_V.ravel()[:] = 0.0

        if Is_V:
            V = mydf.V_R
            ngrid_per_box = ngrid_prim
            _emb_R = emb_R
        else:
            V = mydf.W
            ngrid_per_box = nIP_prim
            _emb_R = emb_Rg
        
        for kx in range(kmesh[0]):
            for ky in range(kmesh[1]):
                for kz in range(kmesh[2]):
                    
                    ILOC = kx*kmesh[1]*kmesh[2] + ky*kmesh[2] + kz
                    
                    for i in range(nspin):
                        
                        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                        
                        _emb_pair = np.einsum('iP,jP->ijP', _emb_R[i][ILOC], _emb_R[i][ILOC])
                        _emb_pair = _emb_pair.reshape(nemb*nemb, ngrid_per_box)
                        # _tmp_V    = lib.ddot(V[:,ILOC*ngrid_per_box:(ILOC+1)*ngrid_per_box],_emb_pair.T)
                        
                        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                        cputime_embpairR += t2[0] - t1[0]
                        walltime_embpairR+= t2[1] - t1[1]

                        ## another pass to account for the transposition ##
                        
                        for ix in range(kmesh[0]):
                            for iy in range(kmesh[1]):
                                for iz in range(kmesh[2]):
                                    
                                    ix_ = (kx-ix+kmesh[0]) % kmesh[0]
                                    iy_ = (ky-iy+kmesh[1]) % kmesh[1]
                                    iz_ = (kz-iz+kmesh[2]) % kmesh[2]
                                    
                                    ILOC_ = ix_*kmesh[1]*kmesh[2] + iy_*kmesh[2] + iz_
                                    ILOC  = ix *kmesh[1]*kmesh[2] + iy *kmesh[2] + iz
                                    
                                    lib.ddot(
                                        a=V[:,ILOC_*ngrid_per_box:(ILOC_+1)*ngrid_per_box],
                                        b=_emb_pair.T,
                                        alpha=1.0,
                                        c=tmp_V[i][ILOC*nIP_prim:(ILOC+1)*nIP_prim,:],
                                        beta=1.0)

                        t3 = (lib.logger.process_clock(), lib.logger.perf_counter())
                        cputime_ddot += t3[0] - t2[0]
                        walltime_ddot+= t3[1] - t2[1]
    
    def _the_last_pass(plus):
        
        global cputime_embpairR, walltime_embpairR
        global cputime_ddot, walltime_ddot
        
        if plus:
            alpha = 1
        else:
            alpha =-1
            
        for ix in range(kmesh[0]):
            for iy in range(kmesh[1]):
                for iz in range(kmesh[2]):
                
                    ILOC = ix*kmesh[1]*kmesh[2] + iy*kmesh[2] + iz
                
                    if nspin == 1:
                        
                        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                        
                        emb_pair_Rg = np.einsum('iP,jP->ijP', emb_Rg[0][ILOC], emb_Rg[0][ILOC])
                        emb_pair_Rg = emb_pair_Rg.reshape(nemb*nemb, nIP_prim)

                        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                        cputime_embpairR += t2[0] - t1[0]
                        walltime_embpairR+= t2[1] - t1[1]

                        lib.ddot(
                            a = emb_pair_Rg,
                            b = tmp_V[0][ILOC*nIP_prim:(ILOC+1)*nIP_prim,:],
                            alpha = alpha,
                            c     = eri[0],
                            beta  = 1
                        )
                        
                        t3 = (lib.logger.process_clock(), lib.logger.perf_counter())
                        cputime_ddot += t3[0] - t2[0]
                        walltime_ddot+= t3[1] - t2[1]
                        
                    else:
                        if nspin == 2:
                            
                            t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
                            
                            emb_pair_Rg_alpha = np.einsum('iP,jP->ijP', emb_Rg[0][ILOC], emb_Rg[0][ILOC])
                            emb_pair_Rg_beta  = np.einsum('iP,jP->ijP', emb_Rg[1][ILOC], emb_Rg[1][ILOC])
                            
                            t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                            cputime_embpairR += t2[0] - t1[0]
                            walltime_embpairR+= t2[1] - t1[1]

                            lib.ddot(
                                a = emb_pair_Rg_alpha,
                                b = tmp_V[0][ILOC*nIP_prim:(ILOC+1)*nIP_prim,:],
                                alpha = alpha,
                                c     = eri[0],
                                beta  = 1
                            )

                            lib.ddot(
                                a = emb_pair_Rg_beta,
                                b = tmp_V[1][ILOC*nIP_prim:(ILOC+1)*nIP_prim,:],
                                alpha = alpha,
                                c     = eri[1],
                                beta  = 1
                            )

                            lib.ddot(
                                a = emb_pair_Rg_alpha,
                                b = tmp_V[1][ILOC*nIP_prim:(ILOC+1)*nIP_prim,:],
                                alpha = alpha,
                                c     = eri[2],
                                beta  = 1
                            )
                            
                            t3 = (lib.logger.process_clock(), lib.logger.perf_counter())
                            cputime_ddot += t3[0] - t2[0]
                            walltime_ddot+= t3[1] - t2[1]

                        else:
                            raise ValueError("nspin > 2 is not supported")
    
    if with_robust_fitting:
        
        _construct_tmp_V_W(True)
        _the_last_pass(plus=True)
        nspinpair = nspin*(nspin+1)//2
        
        for i in range(nspinpair):
            eri[i] += eri[i].T
    
    ### W term ###
    
    _construct_tmp_V_W(False)
    if with_robust_fitting:
        _the_last_pass(plus=False)
    else:
        _the_last_pass(plus=True)
    
    #### post process ####
    
    # reshape the eri 
    
    eri = eri.reshape(nspin*(nspin+1)//2, nemb, nemb, nemb, nemb)
    eri = eri_restore(eri, symmetry, nemb)
    
    log_profile_buildK_time(mydf, mydf.use_mpi)
    
    return eri * ngrid / vol
 

class aoR_Holder2(aoR_Holder):
    def __init__(self, aoR, ao_involved, grid_involved, global_gridID_begin, global_gridID_end):
        super(aoR_Holder2, self).__init__(aoR, ao_involved, 0, global_gridID_end-global_gridID_begin, global_gridID_begin, global_gridID_end)
        self.grid_involved = grid_involved

    def size(self):
        return super(aoR_Holder2, self).size() + self.grid_involved.nbytes

def _allocate_buffer_get_embR(mydf:PBC_ISDF_Info_Quad_K, nemb, nspin=1, IsIP=False):
    max_nao = mydf.max_nao_involved
    if IsIP:
        max_grid        = mydf.max_nIP_involved
        ngrid           = mydf.nIP_Prim
    else:
        max_grid        = mydf.max_ngrid_involved
        ngrid           = mydf.nGridPrim
    size = nspin * nemb * ngrid + nemb * max_grid + nemb * max_nao
    return np.zeros(size, dtype=np.float64)

def _get_embR_box(mydf:PBC_ISDF_Info_Quad_K, C_ao_emb, 
                  ix, iy, iz, 
                  kmesh,
                  build_V_bunchsize = 64,
                  IsIP              = False,
                  aoR_cutoff        = 1e-8,
                  aoPair_cutoff     = 1e-10,
                  buffer            = None):
    
    '''
    get embeded orbital values on given box,
    return list[aoR_holder] with size nspin with aoR_cutoff
    '''
    
    nspin, nao_full, nemb = C_ao_emb.shape
    C_ao_emb_T            = C_ao_emb.transpose(0, 2, 1).copy()
    
    ILOC = ix*kmesh[1]*kmesh[2] + iy*kmesh[2] + iz
    
    ###### allocate buffer ###### 
    
    if buffer is None:
        buffer = _allocate_buffer_get_embR(mydf, nemb, nspin, IsIP)
    
    max_nao = mydf.max_nao_involved
    
    if IsIP:
        max_grid        = mydf.max_nIP_involved
        ngrid           = mydf.nIP_Prim
        global_id_begin = ILOC * ngrid
    else:
        max_grid        = mydf.max_ngrid_involved
        ngrid           =  mydf.nGridPrim
        global_id_begin = ILOC * ngrid
    
    res         = np.ndarray((nspin, nemb, ngrid), dtype=np.float64, buffer=buffer, offset=0)
    offset_now  = res.size * res.itemsize 
    ddot_buf    = np.ndarray((nemb, max_grid),     dtype=np.float64, buffer=buffer, offset=offset_now)
    offset_now += ddot_buf.size * ddot_buf.itemsize
    pack_buf    = np.ndarray((nemb, max_nao),      dtype=np.float64, buffer=buffer, offset=offset_now)
    
    if IsIP:
        permutation = mydf._get_permutation_column_aoRg(ix, iy, iz)
    else:
        permutation = mydf._get_permutation_column_aoR(ix, iy, iz)
    
    fn_packcol = getattr(libisdf, "_buildK_packcol", None)
    assert fn_packcol is not None
    
    ###### calculate ######
    
    for i in range(mydf.natmPrim):
        
        _permutation_ = permutation[i]

        if IsIP:
            aoR_i = mydf.aoRg[i]
        else:
            aoR_i = mydf.aoR[i]

        global_IP_begin_i = aoR_i.global_gridID_begin
        ngrid_i           = aoR_i.aoR.shape[1]
        nao_involved_i    = aoR_i.aoR.shape[0]

        ddot_res = np.ndarray((nemb, ngrid_i),        dtype=np.float64, buffer=ddot_buf)
        pack_res = np.ndarray((nemb, nao_involved_i), dtype=np.float64, buffer=pack_buf)

        for j in range(nspin):
            fn_packcol(
                pack_res.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nemb),
                ctypes.c_int(nao_involved_i),
                C_ao_emb_T[j].ctypes.data_as(ctypes.c_void_p),
                ctypes.c_int(nemb),
                ctypes.c_int(nao_full),
                _permutation_.ctypes.data_as(ctypes.c_void_p)
            )
            lib.ddot(pack_res, aoR_i.aoR, c=ddot_res)
            res[j, :, global_IP_begin_i:global_IP_begin_i+ngrid_i] = ddot_res

    ###### cutoff ######

    res2 = []

    grid_cutoff = np.sqrt(aoPair_cutoff)

    for i in range(nspin):
        ## find the max of each row ##
        res_i = res[i]
        res_i_max = np.max(np.abs(res_i), axis=1)
        ## find the row id with max > aoR_cutoff ##
        res_row_max_id = np.where(res_i_max > aoR_cutoff)[0]
        ## cutoff ##
        res_i_new = res_i[res_row_max_id].copy()
        ## find the max of each col ##
        res_i_max = np.max(np.abs(res_i_new), axis=0)
        ## find the col id with max > grid_cutoff ##
        res_col_max_id = np.where(res_i_max > grid_cutoff)[0]
        ## cutoff ##
        res_i_new = res_i_new[:, res_col_max_id].copy()
        res2.append(aoR_Holder2(res_i_new, res_row_max_id, res_col_max_id, global_id_begin, global_id_begin+ngrid))

    return res2

def get_emb_eri_isdf_fast(mydf, 
                          C_ao_emb:np.ndarray=None, 
                          symmetry=4,
                          aoR_cutoff=1e-8, 
                          aoPair_cutoff=1e-12,
                          loop_bunchsize=56):
    
    ''' 
    get eri for embedding system with cutoff
    '''
    
    #### preprocess #### 
    
    assert isinstance(mydf, PBC_ISDF_Info_Quad_K)
    
    log = lib.logger.Logger(mydf.stdout, mydf.verbose)
    
    # assert not mydf.direct direct mode is allowed # 
    
    if C_ao_emb.ndim == 2:
        C_ao_emb = C_ao_emb.reshape(1, *C_ao_emb.shape)
    assert C_ao_emb.ndim  == 3
    assert C_ao_emb.dtype == np.float64  ## supercell basis
    
    nspin, nao_full, nemb = C_ao_emb.shape
    supercell = mydf.cell
    assert supercell.nao == nao_full
    
    ngrid      = mydf.ngrids
    vol        = supercell.vol
    mesh_prim  = mydf.primCell.mesh
    ngrid_prim = np.prod(mesh_prim)
    nao_prim   = mydf.nao_prim
    nIP_prim   = mydf.nIP_Prim
    kmesh      = mydf.kmesh
    nkpts      = np.prod(kmesh)
    nIP        = mydf.naux
    
    with_robust_fitting = mydf.with_robust_fitting
    direct              = mydf.direct
    
    cell = mydf.cell
    mesh = np.array(cell.mesh, dtype=np.int32)
    
    ### emb values on grid and IPs ###
    
    ###### allocate buffer #######
    
    t1 = (lib.logger.process_clock(), lib.logger.perf_counter())    
    
    emb_R = []
    emb_Rg= []
    for i in range(nspin):
        emb_R.append([])
        emb_Rg.append([])
    
    if with_robust_fitting:
        buffer = _allocate_buffer_get_embR(mydf, nemb, nspin, False)
    else:
        buffer = _allocate_buffer_get_embR(mydf, nemb, nspin, True)
    
    for kx in range(kmesh[0]):
        for ky in range(kmesh[1]):
            for kz in range(kmesh[2]):
                if with_robust_fitting:
                    tmp = _get_embR_box(mydf, C_ao_emb, kx, ky, kz, kmesh, IsIP=False, aoR_cutoff=aoR_cutoff, aoPair_cutoff=aoPair_cutoff, buffer=buffer)
                    for i in range(nspin):
                        emb_R[i].append(tmp[i])
                        
                        log.info("emb values on grids box %2d %2d %2d for spin %1d, nao : %4d/%4d, grid points %6d/%6d"% (kx, ky, kz, i, tmp[i].aoR.shape[0], nemb, tmp[i].aoR.shape[1], ngrid_prim))
                
                tmp = _get_embR_box(mydf, C_ao_emb, kx, ky, kz, kmesh, IsIP=True, aoR_cutoff=aoR_cutoff, aoPair_cutoff=aoPair_cutoff, buffer=buffer)
                for i in range(nspin):
                    emb_Rg[i].append(tmp[i])
                
                    log.info("emb values on IPs   box %2d %2d %2d for spin %1d, nao : %4d/%4d, IPs         %6d/%6d"% (kx, ky, kz, i, tmp[i].aoR.shape[0], nemb, tmp[i].aoR.shape[1], nIP_prim))
                    
    
    t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
    cputime_embR  = t2[0] - t1[0]
    walltime_embR = t2[1] - t1[1]
    
    ###############################
    
    ## extract the info for further calculation ##
    
    max_nemb_involved = 0
    
    for i in range(nspin):
        if with_robust_fitting:
            max_tmp = np.max([x.aoR.shape[0] for x in emb_R[i]])
            max_nemb_involved = max(max_nemb_involved, max_tmp)
        max_tmp = np.max([x.aoR.shape[0] for x in emb_Rg[i]])
        max_nemb_involved = max(max_nemb_involved, max_tmp)
        
    max_ngrid_invovled = 0
    
    for i in range(nspin):
        if with_robust_fitting:
            max_tmp = np.max([x.aoR.shape[1] for x in emb_R[i]])
            max_ngrid_invovled = max(max_ngrid_invovled, max_tmp)
        max_tmp = np.max([x.aoR.shape[1] for x in emb_Rg[i]])
        max_ngrid_invovled = max(max_ngrid_invovled, max_tmp)    
    
    max_nIP_involved = 0
    for i in range(nspin):
        max_tmp = np.max([x.aoR.shape[1] for x in emb_Rg[i]])
        max_nIP_involved = max(max_nIP_involved, max_tmp)
    
    log.info("max_nemb_involved = %4d, max_ngrid_invovled = %4d" % (max_nemb_involved, max_ngrid_invovled))
    log.info("max_nIP_involved  = %4d" % (max_nIP_involved))
    
    not_invovled_box = []
    for i in range(nspin):
        if with_robust_fitting:
            not_invovled_box.append([_id_ for (_id_,holder) in enumerate(emb_R[i]) if holder.aoR.shape[0] == 0 or holder.aoR.shape[1] == 0])
        else:
            not_invovled_box.append([_id_ for (_id_,holder) in enumerate(emb_Rg[i]) if holder.aoR.shape[0] == 0 or holder.aoR.shape[1] == 0])
        if len(not_invovled_box[-1]) > 0:
            log.info("Warning: some boxes are not involved in the calculation for %s" % (not_invovled_box))
    
    max_nIP_group = 0
    for i in range(nspin):
        max_tmp       = np.max([x.shape[0] for x in mydf.aux_basis])
        max_nIP_group = max(max_nIP_group, max_tmp)
        
    ###############################
    
    nnemb = max_nemb_involved * (max_nemb_involved + 1) // 2
    
    if with_robust_fitting:
        buffer_tmp_embpair_on_grids = np.zeros((nnemb, max_ngrid_invovled), dtype=np.float64)
        buffer_packVW               = np.zeros((loop_bunchsize, max_ngrid_invovled), dtype=np.float64)
    else:
        buffer_tmp_embpair_on_grids = np.zeros((nnemb, max_nIP_involved),   dtype=np.float64)
        buffer_packVW               = np.zeros((loop_bunchsize, max_nIP_involved),   dtype=np.float64)
    
    if direct:
        
        nThread    = lib.num_threads()
        coulG      = mydf.coulG
        coulG_real = coulG.reshape(*mesh)[:, :, :mesh[2]//2+1].reshape(-1).copy()
        bufsize_per_thread = coulG_real.shape[0] * 2 + np.prod(mesh)
        buffer_build_V     = np.ndarray((nThread, bufsize_per_thread), dtype=np.float64)

        if with_robust_fitting:
            buffer_V_tmp = np.zeros((loop_bunchsize, np.prod(mesh)), dtype=np.float64)
            buffer_W_tmp = np.zeros((max_nIP_group,  nIP), dtype=np.float64)
    
    ### other info ###
    
    aux_basis = mydf.aux_basis
    
    ## function for build V and W  ##
    
    def _build_V_W(group_id, p0, p1, V_, W_):
        
        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        assert direct
        assert V_.shape[0] == p1-p0
        aux_basis_tmp = aux_basis[group_id]
        assert W_.shape[0] == aux_basis_tmp.shape[0]
        
        aux_basis_grid_ID = mydf.partition_group_to_gridID[group_id]
        
        from pyscf.isdf._isdf_get_K_direct import construct_V
    
        construct_V(aux_basis[group_id][p0:p1, :],
                    buf_build_V,
                    V_,
                    aux_basis_grid_ID,
                    mydf.grid_ID_ordered,
                    mesh,
                    coulG_real)
        
        aux_ket_shift = 0
        grid_shift    = 0
            
        ILOC = 0
        for ix in range(kmesh[0]):
            for iy in range(kmesh[1]):
                for iz in range(kmesh[2]):
                    for j in range(len(group)):
                        aux_basis_ket  = mydf.aux_basis[j]
                        ngrid_now      = aux_basis_ket.shape[1]
                        naux_ket       = aux_basis_ket.shape[0]
                        if not skip:
                            W_[p0:p1, aux_ket_shift:aux_ket_shift+naux_ket] = lib.ddot(
                            V_[:, grid_shift:grid_shift+ngrid_now], aux_basis_ket.T)
                        aux_ket_shift += naux_ket
                        grid_shift    += ngrid_now 
                    ILOC += 1
        
        
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        
        cputime_buildVW  += t2[0] - t1[0]
        walltime_buildVW += t2[1] - t1[1]
    
    ## function for final contraction ##
    
    ###### loop over groups first and then loop over bunches ######
    
    eri = np.zeros((nspin*(nspin+1)//2, nnemb, nnemb), dtype=np.float64) ## the ordering of spin is aa, bb, ab
    
    group = mydf.group
    assert len(group) == len(aux_basis)
    
    ip_id_now = 0
    for group_id, atm_ids in enumerate(group):
        
        naux_tmp = aux_basis[group_id].shape[0]
        
        if direct:
            W_tmp = np.ndarray((naux_tmp, nIP), dtype=np.float64, buffer=buffer_W_tmp)
        else:
            W = mydf.W
            V = mydf.V_R
        
        for p0, p1 in lib.prange(0, naux_tmp, loop_bunchsize):
            if direct:
                V_tmp = np.ndarray((p1-p0, np.prod(mesh)), dtype=np.float64, buffer=buffer_V_tmp)
                _build_V_W(group_id, p0, p1, V_tmp, W_tmp)
            else:
                V_tmp = V[ip_id_now+p0:ip_id_now+p1, :]
                W_tmp = W[ip_id_now+p0:ip_id_now+p1, :]
            
            
        

    
if __name__ == "__main__":

    from isdf_tools_cell import build_supercell, build_supercell_with_partition
    C = 25
    
    verbose = 10
    import pyscf.pbc.gto as pbcgto
    
    cell   = pbcgto.Cell()
    boxlen = 3.5668
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
    atm = [
        ['C', (0.     , 0.     , 0.    )],
        ['C', (0.8917 , 0.8917 , 0.8917)],
        ['C', (1.7834 , 1.7834 , 0.    )],
        ['C', (2.6751 , 2.6751 , 0.8917)],
        ['C', (1.7834 , 0.     , 1.7834)],
        ['C', (2.6751 , 0.8917 , 2.6751)],
        ['C', (0.     , 1.7834 , 1.7834)],
        ['C', (0.8917 , 2.6751 , 2.6751)],
    ] 
    
    KE_CUTOFF = 70
    basis = 'gth-szv'
    
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], basis=basis, ke_cutoff=KE_CUTOFF)
    prim_mesh = prim_cell.mesh
    # prim_partition = [[0], [1], [2], [3], [4], [5], [6], [7]]
    # prim_partition = [[0,1,2,3,4,5,6,7]]
    prim_partition = [[0,1],[2,3],[4,5],[6,7]]
    
    Ls = [1, 2, 2]
    kpts = prim_cell.make_kpts(Ls)
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, 
                                                     Ls=Ls,
                                                     basis=basis, 
                                                     #pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    
    pbc_isdf_info = PBC_ISDF_Info_Quad_K(prim_cell, kmesh=Ls, with_robust_fitting=True, aoR_cutoff=1e-8, 
                                         # direct=True, 
                                         direct=False, 
                                         rela_cutoff_QRCP=1e-4,
                                         limited_memory=True, build_K_bunchsize=32)
    pbc_isdf_info.build_IP_local(c=C, m=5, group=prim_partition, Ls=[Ls[0]*10, Ls[1]*10, Ls[2]*10])
    pbc_isdf_info.verbose = 10    
    pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
    
    # print("grid_segment = ", pbc_isdf_info.grid_segment)
    
    from pyscf.pbc import scf

    mf = scf.KRHF(prim_cell, kpts)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 16
    mf.conv_tol = 1e-7
    
    mf.kernel()

    nao_full = pbc_isdf_info.cell.nao
    nao_emb  = nao_full // 5
    C_ao_emb = np.random.rand(nao_full, nao_emb)
    
    eri_emb = get_emb_eri_isdf(pbc_isdf_info, C_ao_emb, symmetry=4)
    
    supercell = pbc_isdf_info.cell
    
    from pyscf.isdf.isdf_local import PBC_ISDF_Info_Quad
    
    pbc_isdf_info2 = PBC_ISDF_Info_Quad(supercell, with_robust_fitting=True, 
                                        aoR_cutoff=1e-8, 
                                        direct=False, 
                                        # direct=True, 
                                        limited_memory=True, build_K_bunchsize=32,
                                        use_occ_RI_K=False, rela_cutoff_QRCP=1e-4)
        
    pbc_isdf_info2.build_IP_local(c=C, m=5, group=group_partition)
    pbc_isdf_info2.build_auxiliary_Coulomb()    
    
    eri_emb_benchmark = pbc_isdf_info2.ao2mo(C_ao_emb)
    
    assert eri_emb.shape == eri_emb_benchmark.shape
    
    diff = np.linalg.norm(eri_emb - eri_emb_benchmark)
    print("diff     = ", diff)
    max_diff = np.max(np.abs(eri_emb - eri_emb_benchmark))
    print("max_diff = ", max_diff)
    
    print("eri_emb.shape = ", eri_emb.shape)
    print("eri_emb           = ", eri_emb[0,0],eri_emb[0,1])
    print("eri_emb_benchmark = ", eri_emb_benchmark[0,0], eri_emb_benchmark[0,1])
    for i in range(eri_emb.shape[0]):
        for j in range(eri_emb.shape[1]):
            print(eri_emb[i,j], eri_emb_benchmark[i,j], eri_emb[i,j]/eri_emb_benchmark[i,j])
