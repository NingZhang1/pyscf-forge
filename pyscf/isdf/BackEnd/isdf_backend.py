from pyscf.isdf.BackEnd._config import USE_NUMPY, USE_SCIPY, USE_TORCH, USE_TORCH_GPU, ENABLE_FFTW
from pyscf.isdf.BackEnd._config import MULTI_THREADING, USE_GPU

if not MULTI_THREADING:
    import os

    os.environ["OMP_NUM_THREADS"] = 1

from pyscf.isdf.BackEnd._num_threads import num_threads

NUM_THREADS = num_threads()

assert isinstance(USE_NUMPY, int)
assert isinstance(USE_SCIPY, int)
assert isinstance(USE_TORCH, int)
assert isinstance(USE_TORCH_GPU, int)
assert isinstance(USE_GPU, int)
assert USE_NUMPY >= 0
assert USE_SCIPY >= 0
assert USE_TORCH >= 0
assert USE_TORCH_GPU >= 0

assert USE_NUMPY + USE_SCIPY + USE_TORCH + USE_TORCH_GPU == 1

# import different backend #

if USE_NUMPY:
    import pyscf.isdf.BackEnd._numpy as backend
elif USE_SCIPY:
    import pyscf.isdf.BackEnd._scipy as backend
elif USE_TORCH_GPU:
    import pyscf.isdf.BackEnd._torch as backend

    USE_GPU = 1
else:
    import pyscf.isdf.BackEnd._torch as backend

    USE_GPU = 0

# assign python interface #

# type system #
INT32 = backend.INT32Ty
INT64 = backend.INT64Ty
FLOAT32 = backend.FLOAT32Ty
FLOAT64 = backend.FLOAT64Ty
COMPLEX64 = backend.COMPLEX64Ty
COMPLEX128 = backend.COMPLEX128Ty
ITEM_SIZE = {
    INT32: 4,
    INT64: 8,
    FLOAT32: 4,
    FLOAT64: 8,
    COMPLEX64: 8,
    COMPLEX128: 16,
}
ToNUMPYTy = backend.ToNUMPYTy
TENSORTy = backend.TENSORTy
_zeros = backend.zeros
_real = backend.real
_imag = backend.imag
_permute = backend.permute
_conjugate = backend.conjugate
_conjugate_ = backend.conjugate_
_is_realtype = backend.is_realtype
_is_complextype = backend.is_complextype
# func interface #
_toTensor = backend.toTensor
_toNumpy = backend.toNumpy
_malloc = backend.malloc
_fftn = backend.fftn
_ifftn = backend.ifftn
_rfftn = backend.rfftn
_irfftn = backend.irfftn
_dot = backend.dot
_qr_col_pivoting = backend.qr_col_pivoting
_qr = backend.qr
_index_add = backend.index_add
_index_copy = backend.index_copy
_take = backend.take
_maximum = backend.maximum
_minimum = backend.minimum
_absolute = backend.absolute
_Frobenius_norm = backend.Frobenius_norm
_einsum_ij_j_ij = backend.einsum_ij_j_ij
_einsum_i_ij_ij = backend.einsum_i_ij_ij
_einsum_ik_jk_ijk = backend.einsum_ik_jk_ijk
_eigh = backend.eigh
_square = backend.square
_square_ = backend.square_
_cholesky = backend.cholesky
_solve_cholesky = backend.solve_cholesky

# some simple utils use numpy's impl #

import numpy

_prod = numpy.prod
