# backend to test #

import pyscf.isdf.BackEnd._config as config

config.disable_fftw()
# config.backend("numpy")
# config.backend("scipy")
config.backend("torch")
# config.backend("torch_gpu")
import pyscf.isdf.BackEnd.isdf_backend as BACKEND

ZEROS = BACKEND._zeros
FLOAT64 = BACKEND.FLOAT64
ToNUMPY = BACKEND._toNumpy

# sys and pyscf #

import numpy as np
from pyscf import lib

from pyscf.isdf.isdf_tools_Tsym import (
    symmetrize_mat,
    _1e_operator_gamma2k,
    _1e_operator_k2gamma,
)


KMESH = [
    [1, 1, 1],
    [1, 1, 2],
    [1, 1, 3],
    [1, 2, 2],
    [1, 3, 3],
    [2, 2, 2],
    [3, 3, 3],
]

NAO = [3, 5, 7, 11]


for kmesh in KMESH:
    nkpts = np.prod(kmesh)
    for nao in NAO:
        dm = ZEROS((nkpts * nao, nkpts * nao), dtype=FLOAT64)
        dm = symmetrize_mat(dm, kmesh)
        dm_kpts = _1e_operator_gamma2k(nkpts * nao, kmesh, dm)
        dm2 = _1e_operator_k2gamma(nkpts * nao, kmesh, dm_kpts)

        dm = ToNUMPY(dm)
        dm2 = ToNUMPY(dm2)

        assert dm.__array_interface__["data"][0] != dm2.__array_interface__["data"][0]

        np.testing.assert_allclose(dm, dm2, atol=1e-8)
