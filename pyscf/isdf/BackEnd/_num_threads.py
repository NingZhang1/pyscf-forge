import os
import multiprocessing
from BackEnd._config import ENABLE_PYSCF_LIB, MULTI_THREADING

_PYSCF_FOUND = False

if ENABLE_PYSCF_LIB:

    try:
        from pyscf import lib

        _PYSCF_FOUND = True

    except ImportError:
        _PYSCF_FOUND = False

else:
    _PYSCF_FOUND = False

if _PYSCF_FOUND:

    def num_threads():
        if not MULTI_THREADING:
            return lib.num_threads(1)
        else:
            return lib.num_threads()

else:

    def num_threads():
        if not MULTI_THREADING:
            return 1
        if "OMP_NUM_THREADS" in os.environ:
            return int(os.environ["OMP_NUM_THREADS"])
        else:
            return multiprocessing.cpu_count()
