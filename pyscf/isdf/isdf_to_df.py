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

import numpy
import contextlib

from pyscf.pbc.df.df import GDF
from pyscf.df.df import DF
from pyscf.isdf.isdf import ISDF
from pyscf.isdf.isdf_local import ISDF_Local
from pyscf.lib import logger
from pyscf.isdf.isdf_to_df_incore import cholesky_eri
import pyscf.isdf.BackEnd.isdf_backend as BACKEND
from pyscf import lib

ToNUMPY = BACKEND._toNumpy


class DF_ISDF(DF):
    r"""
    the density fitting object adapted from (local) ISDF without robust fitting
    """

    def __init__(self, isdf_obj: ISDF):
        cell = isdf_obj.cell
        super().__init__(cell, auxbasis="ISDF interpolation points")
        self._isdf_obj = isdf_obj
        self._aoRg = isdf_obj.aoRg_full().aoR
        # self._aoRg = ToNUMPY(self._aoRg)
        # add a common factor #
        factor = (isdf_obj.ngrids / isdf_obj.cell.vol) ** 0.25
        # print("factor = ", factor)
        self._aoRg *= factor
        # print some warning #
        if hasattr(isdf_obj, "outcore") and isdf_obj.outcore:
            raise NotImplementedError("outcore is not supported")
        if isdf_obj.with_robust_fitting:
            logger.warning(
                self,
                "robust fitting is used with possibly small c, DF_ISDF use only the W matrix, the error may be too large",
            )
        self.kpts = isdf_obj.kpts

    def build(self):
        self._cderi = cholesky_eri(self)
        self._aoRg = ToNUMPY(
            self._aoRg
        )  # NOTE all the involved tensors are converted to numpy array, to be consistent with pyscf
        return self

    def reset(self, mol=None):
        raise NotImplementedError

    @contextlib.contextmanager
    def range_coulomb(self, omega):
        raise NotImplementedError

class GDF_ISDF(GDF):
    r"""
    the density fitting object adapted from (local) ISDF without robust fitting
    """

    def __init__(self, isdf_obj: ISDF):
        cell = isdf_obj.cell
        super().__init__(cell)
        self._isdf_obj = isdf_obj
        self._aoRg = isdf_obj.aoRg_full().aoR
        # self._aoRg = ToNUMPY(self._aoRg)
        # add a common factor #
        factor = (isdf_obj.ngrids / isdf_obj.cell.vol) ** 0.25
        # print("factor = ", factor)
        self._aoRg *= factor
        # print some warning #
        if hasattr(isdf_obj, "outcore") and isdf_obj.outcore:
            raise NotImplementedError("outcore is not supported")
        if isdf_obj.with_robust_fitting:
            logger.warning(
                self,
                "robust fitting is used with possibly small c, DF_ISDF use only the W matrix, the error may be too large",
            )
        self.kpts = isdf_obj.kpts

    def get_pp(self, kpts=None):
        return self._isdf_obj.get_pp(kpts)

    def build(self, j_only=None, with_j3c=True, kpts_band=None):
        self._cderi = cholesky_eri(self)
        self._cderi = self._cderi.reshape(1, *self._cderi.shape)
        self._aoRg = ToNUMPY(
            self._aoRg
        )  # NOTE all the involved tensors are converted to numpy array, to be consistent with pyscf
        return self

    def reset(self, mol=None):
        raise NotImplementedError

    @contextlib.contextmanager
    def range_coulomb(self, omega):
        raise NotImplementedError
    
    def sr_loop(self, kpti_kptj=numpy.zeros((2,3)), max_memory=2000,
                compact=True, blksize=None, aux_slice=None):
        assert numpy.allclose(kpti_kptj, numpy.zeros((2,3)))
        # assert compact
        cell = self.cell
        nao = cell.nao_nr()
        if cell.dimension == 2 and cell.low_dim_ft_type != 'inf_vacuum':
            raise RuntimeError('ERIs of PBC-2D systems are not positive '
                               'definite. Current API only supports positive '
                               'definite ERIs.')
        if blksize is None:
            blksize = self.blockdim

        if compact:
            for LpqR in self.loop(blksize):
                yield LpqR, numpy.zeros_like(LpqR), 1
        else:
            for LpqR in self.loop(blksize):
                naux = LpqR.shape[0]
                LpqR = lib.unpack_tril(LpqR).reshape(naux, nao**2)
                yield LpqR, numpy.zeros_like(LpqR), 1
    
    def loop(self, blksize=None):
        if self._cderi is None:
            self.build()
        if blksize is None:
            blksize = self.blockdim

        from pyscf.df import addons
        import h5py
        from pyscf.ao2mo.outcore import _load_from_h5g
        
        with addons.load(self._cderi, self._dataname) as feri:
            if isinstance(feri, numpy.ndarray):
                naoaux = feri.shape[1]
                for b0, b1 in self.prange(0, naoaux, blksize):
                    yield numpy.asarray(feri[0, b0:b1], order='C')

            else:
                raise NotImplementedError
                if isinstance(feri, h5py.Group):
                    # starting from pyscf-1.7, DF tensor may be stored in
                    # block format
                    naoaux = feri['0'].shape[0]
                    def load(aux_slice):
                        b0, b1 = aux_slice
                        return _load_from_h5g(feri, b0, b1)
                else:
                    naoaux = feri.shape[0]
                    def load(aux_slice):
                        b0, b1 = aux_slice
                        return numpy.asarray(feri[b0:b1])

                for dat in lib.map_with_prefetch(load, self.prange(0, naoaux, blksize)):
                    yield dat
                    dat = None