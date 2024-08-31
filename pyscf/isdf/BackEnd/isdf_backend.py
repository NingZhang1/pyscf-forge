from pyscf.isdf.BackEnd._config import (
    USE_NUMPY,
    USE_SCIPY,
    USE_TORCH,
    USE_TORCH_GPU,
    ENABLE_FFTW,
)
from pyscf.isdf.BackEnd._config import MULTI_THREADING, USE_GPU

if not MULTI_THREADING:
    import os

    os.environ["OMP_NUM_THREADS"] = 1

from pyscf.isdf.BackEnd._num_threads import num_threads

NUM_THREADS = num_threads()

print("LOADING ISDF BACKEND")
print("NUM_THREADS   :", NUM_THREADS)
print("USE_NUMPY     :", USE_NUMPY)
print("USE_SCIPY     :", USE_SCIPY)
print("USE_TORCH     :", USE_TORCH)
print("USE_TORCH_GPU :", USE_TORCH_GPU)

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
    # import torch
    # torch.set_num_threads(NUM_THREADS) # NOTE: problematic on pauling
    USE_GPU = 1
else:
    import pyscf.isdf.BackEnd._torch as backend
    # import torch
    # torch.set_num_threads(NUM_THREADS)
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
_clean = backend.clean
_maximum = backend.maximum
_minimum = backend.minimum
_absolute = backend.absolute
_Frobenius_norm = backend.Frobenius_norm
_cwise_mul = backend.cwise_mul
_einsum_ij_j_ij = backend.einsum_ij_j_ij
_einsum_i_ij_ij = backend.einsum_i_ij_ij
_einsum_ik_jk_ijk = backend.einsum_ik_jk_ijk
_einsum_ij_ij_j = backend.einsum_ij_ij_j
_eigh = backend.eigh
_square = backend.square
_square_ = backend.square_
_add_transpose_ = backend.add_transpose_
_cholesky = backend.cholesky
_solve_cholesky = backend.solve_cholesky

# some simple utils use numpy's impl #

import numpy

_prod = numpy.prod

# some special functions designed for ISDF #

if USE_NUMPY:
    import pyscf.isdf.BackEnd._isdf_numpy as isdf_special_func
elif USE_SCIPY:
    import pyscf.isdf.BackEnd._isdf_scipy as isdf_special_func
elif USE_TORCH_GPU:
    import pyscf.isdf.BackEnd._isdf_torch as isdf_special_func
else:
    import pyscf.isdf.BackEnd._isdf_torch as isdf_special_func

_distance_translation = isdf_special_func.distance_translation
_add_to_indexed_submatrix_ = isdf_special_func.add_to_indexed_submatrix_
_copy_indexed_submatrix = isdf_special_func.copy_indexed_submatrix
_cast_to_complex = isdf_special_func.cast_to_complex