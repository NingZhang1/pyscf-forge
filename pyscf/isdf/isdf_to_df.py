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

from pyscf.df.df import DF
from pyscf.isdf.isdf import ISDF
from pyscf.isdf.isdf_local import ISDF_Local
from pyscf.lib import logger
from pyscf.isdf.isdf_to_df_incore import cholesky_eri
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

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
