# this file shared the functions shared by numpy and scipy backend
# to avoid mutual import

import numpy

ToNUMPYTy = {
    numpy.float32: numpy.float32,
    numpy.float64: numpy.float64,
    numpy.int32: numpy.int32,
    numpy.int64: numpy.int64,
    numpy.complex64: numpy.complex64,
    numpy.complex128: numpy.complex128,
}


def is_realtype(a):
    return numpy.issubdtype(a.dtype, numpy.floating)


def is_complextype(a):
    return numpy.issubdtype(a.dtype, numpy.complexfloating)


def zeros(shape, dtype=numpy.float64, like=None, cpu=True):
    assert cpu or cpu is None
    if like is not None:
        if dtype is None:
            dtype = like.dtype
    return numpy.zeros(shape, dtype=dtype, like=like)


def real(a, force_outofplace=False):
    if force_outofplace:
        return numpy.real(a).copy()
    else:
        return numpy.real(a)


def imag(a, force_outofplace=False):
    if force_outofplace:
        return numpy.imag(a).copy()
    else:
        return numpy.imag(a)


def permute(a, axes):
    return numpy.transpose(a, axes)


def conjugate(a, out=None):
    return numpy.conjugate(a, out=out)


def conjugate_(a):
    return conjugate(a, out=a)


def index_add(A, dim, index, source, alpha=1):
    A[(slice(None),) * dim + (index,)] += alpha * source
    return A


def index_copy(A, dim, index, source):
    # equivalent to A[:,:,...,index,...:,...] = source
    A[(slice(None),) * dim + (index,)] = source
    return A


def take(a, indices, axis=None, out=None):
    return numpy.take(a, indices, axis=axis, out=out)


def maximum(a, axis=None, out=None):
    return numpy.max(a, axis=axis, out=out)


def minimum(a, axis=None, out=None):
    return numpy.min(a, axis=axis, out=out)


def absolute(a, out=None):
    return numpy.abs(a, out=out)


def Frobenius_norm(a):
    assert a.ndim == 2
    return numpy.linalg.norm(a)


def einsum_ij_j_ij(a, b, out=None):
    if out is None:
        return a * b
    else:
        numpy.multiply(a, b, out=out)
        return out


def einsum_i_ij_ij(a, b, out=None):
    a_reshaped = a[:, numpy.newaxis]
    if out is None:
        return a_reshaped * b
    else:
        numpy.multiply(a_reshaped, b, out=out)
        return out


def einsum_ik_jk_ijk(a, b, out=None):
    if out is None:
        return numpy.einsum("ik,jk->ijk", a, b)
    else:
        return numpy.einsum("ik,jk->ijk", a, b, out=out)


def square(a, out=None):
    return numpy.square(a, out=out)


def square_(a):
    return square(a, out=a)
