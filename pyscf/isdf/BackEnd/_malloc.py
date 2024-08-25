import numpy as np


try:
    import pyfftw

    def __malloc(shape, dtype, buf=None, offset=0, gpu=False):
        assert not gpu
        if buf is None:
            res = pyfftw.empty_aligned(shape, dtype=dtype)
            return np.asarray(res)
        else:
            return np.ndarray(shape, dtype=dtype, buffer=buf, offset=offset)

except ImportError:

    def __malloc(shape, dtype, buf=None, offset=0, gpu=False):
        assert not gpu
        if buf is None:
            return np.ndarray(shape, dtype=dtype)
        else:
            return np.ndarray(shape, dtype=dtype, buffer=buf, offset=offset)
