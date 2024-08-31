import numpy
import scipy
from pyscf.isdf.BackEnd._config import ENABLE_FFTW, ENABLE_PYSCF_LIB
import pyscf.isdf.BackEnd._pyfftw as _pyfftw
from pyscf.isdf.BackEnd._malloc import __malloc
import pyscf.isdf.BackEnd._numpy_shared_func as SHARED_FUNC_LIB

if ENABLE_PYSCF_LIB:
    try:
        from pyscf import lib

        PYSCF_LIB_FOUND = True
    except ImportError:
        PYSCF_LIB_FOUND = False

# type #

INT32Ty = numpy.int32
INT64Ty = numpy.int64
FLOAT32Ty = numpy.float32
FLOAT64Ty = numpy.float64
COMPLEX64Ty = numpy.complex64
COMPLEX128Ty = numpy.complex128
TENSORTy = numpy.ndarray
ToNUMPYTy = SHARED_FUNC_LIB.ToNUMPYTy
is_realtype = SHARED_FUNC_LIB.is_realtype
is_complextype = SHARED_FUNC_LIB.is_complextype

# toTensor #


def toTensor(data, cpu=True):
    assert cpu
    return numpy.asarray(data)


def toNumpy(data):
    if isinstance(data, list):
        return numpy.array(data)
    if not isinstance(data, numpy.ndarray):
        raise ValueError("Data is not a numpy array")
    return data


malloc = __malloc

# create tensors #

zeros = SHARED_FUNC_LIB.zeros
real = SHARED_FUNC_LIB.real
imag = SHARED_FUNC_LIB.imag
permute = SHARED_FUNC_LIB.permute
conjugate = SHARED_FUNC_LIB.conjugate
conjugate_ = SHARED_FUNC_LIB.conjugate_

# FFT #


if _pyfftw.FFTW_FOUND and ENABLE_FFTW:

    rfftn = _pyfftw.rfftn
    irfftn = _pyfftw.irfftn
    fftn = _pyfftw.fftn
    ifftn = _pyfftw.ifftn

else:

    def rfftn(x, s=None, axes=None, overwrite_input=None, threads=None, out=None):
        return scipy.fft.rfftn(x, s, axes, overwrite_x=overwrite_input, workers=threads)

    def irfftn(x, s=None, axes=None, overwrite_input=None, threads=None, out=None):
        return scipy.fft.irfftn(
            x, s, axes, overwrite_x=overwrite_input, workers=threads
        )

    def fftn(x, s=None, axes=None, overwrite_input=None, threads=None, out=None):
        return scipy.fft.fftn(x, s, axes, overwrite_x=overwrite_input, workers=threads)

    def ifftn(x, s=None, axes=None, overwrite_input=None, threads=None, out=None):
        return scipy.fft.ifftn(x, s, axes)


# matmul #

if ENABLE_PYSCF_LIB and PYSCF_LIB_FOUND:

    def dot(a, b, alpha=1, c=None, beta=0):
        return lib.dot(a, b, alpha=alpha, c=c, beta=beta)

else:

    def dot(a, b, alpha=1, c=None, beta=0):
        if c is not None:
            c *= beta
            c += alpha * numpy.dot(a, b)
            return c
        else:
            return alpha * numpy.dot(a, b)


# QR #

QR_MODE_MAPPING = {
    "full": "full",
    "complete": "full",
    "reduced": "economic",
    "r": "r",
    "raw": "raw",
    "economic": "economic",
}


def qr_col_pivoting(A, tol=1e-8, max_rank=None, mode="r"):
    if mode != "r":
        return scipy.linalg.qr(A, mode=QR_MODE_MAPPING[mode], pivoting=True)
    else:
        r, P = scipy.linalg.qr(A, mode=QR_MODE_MAPPING[mode], pivoting=True)
        return None, r, P


def qr(A, mode="full"):
    return scipy.linalg.qr(A, mode=QR_MODE_MAPPING[mode], pivoting=False)


# index #

index_add = SHARED_FUNC_LIB.index_add
index_copy = SHARED_FUNC_LIB.index_copy
take = SHARED_FUNC_LIB.take
clean = SHARED_FUNC_LIB.clean

# min/max/abs/norm #

maximum = SHARED_FUNC_LIB.maximum
minimum = SHARED_FUNC_LIB.minimum
absolute = SHARED_FUNC_LIB.absolute
Frobenius_norm = SHARED_FUNC_LIB.Frobenius_norm

# special einsum #

cwise_mul = SHARED_FUNC_LIB.cwise_mul
einsum_ij_j_ij = SHARED_FUNC_LIB.einsum_ij_j_ij
einsum_i_ij_ij = SHARED_FUNC_LIB.einsum_i_ij_ij
einsum_ik_jk_ijk = SHARED_FUNC_LIB.einsum_ik_jk_ijk
einsum_ij_ij_j = SHARED_FUNC_LIB.einsum_ij_ij_j

# eigh #


def eigh(a):
    return scipy.linalg.eigh(a)


# square #

square = SHARED_FUNC_LIB.square
square_ = SHARED_FUNC_LIB.square_

add_transpose_ = SHARED_FUNC_LIB.add_transpose_

# cholesky #


def cholesky(a, lower=True, overwrite_a=True, out=None):
    return scipy.linalg.cholesky(
        a, lower=lower, overwrite_a=overwrite_a, check_finite=False
    )


# solve #


def solve_cholesky(
    a, b, lower=True, overwrite_a=True, overwrite_b=True, check_finite=False
):
    C = cholesky(a, lower, overwrite_a, out=None)
    B = scipy.linalg.cho_solve(
        (C, lower), b, overwrite_b=overwrite_b, check_finite=check_finite
    )
    return B
