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

import copy
import numpy as np
import numpy
import ctypes

############ pyscf module ############

from pyscf import lib
from pyscf.lib import logger
import pyscf.pbc.gto as pbcgto
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts import KPoints
from pyscf.pbc.lib.kpts_helper import is_zero, gamma_point, member
from pyscf.gto.mole import ATOM_OF, NCTR_OF, ANG_OF
from pyscf.pbc.dft import multigrid
from pyscf.pbc import df

############ isdf utils ############

############ isdf backends ############

import pyscf.isdf.BackEnd.isdf_backend

############ global variables ############

BASIS_CUTOFF = 1e-18  # too small may lead to numerical instability

############ subroutines --- select IP and build aux basis ############
