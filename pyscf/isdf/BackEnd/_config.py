ENABLE_FFTW = True
ENABLE_PYSCF_LIB = True
FORCE_PYSCF_LIB = False

# multithreading #

MULTI_THREADING = True


def enable_multi_threading():
    global MULTI_THREADING
    MULTI_THREADING = True


def disable_multi_threading():
    global MULTI_THREADING
    MULTI_THREADING = False


# configuration specific to torch backend #

USE_GPU = False
FFT_CPU_USE_TORCH_ANYWAY = False
QR_PIVOTING_GPU_ANYWAY = False


def disable_gpu():
    global USE_GPU
    USE_GPU = False


def enable_gpu():
    global USE_GPU
    USE_GPU = True


def enable_torch_fft_cpu():
    global FFT_CPU_USE_TORCH_ANYWAY
    FFT_CPU_USE_TORCH_ANYWAY = True


def disable_torch_fft_cpu():
    global FFT_CPU_USE_TORCH_ANYWAY
    FFT_CPU_USE_TORCH_ANYWAY = False


def enable_torch_gpu_qr_pivoting():
    global QR_PIVOTING_GPU_ANYWAY
    QR_PIVOTING_GPU_ANYWAY = True


def disable_torch_gpu_qr_pivoting():
    global QR_PIVOTING_GPU_ANYWAY
    QR_PIVOTING_GPU_ANYWAY = False


def enable_fftw():
    global ENABLE_FFTW
    ENABLE_FFTW = True


def disable_fftw():
    global ENABLE_FFTW
    ENABLE_FFTW = False


def enable_pyscf_lib():
    global ENABLE_PYSCF_LIB
    ENABLE_PYSCF_LIB = True


def disable_pyscf_lib():
    global ENABLE_PYSCF_LIB
    ENABLE_PYSCF_LIB = False


def force_pyscf_lib():
    global FORCE_PYSCF_LIB
    FORCE_PYSCF_LIB = True


def disable_pyscf_lib_force():
    global FORCE_PYSCF_LIB
    FORCE_PYSCF_LIB = False


# determine which backend to use #

USE_SCIPY = 1
USE_TORCH = 0
USE_NUMPY = 0
USE_TORCH_GPU = 0


def backend(backend="scipy"):
    backend = backend.lower()
    global USE_SCIPY, USE_TORCH, USE_NUMPY, USE_TORCH_GPU, USE_GPU
    USE_GPU = 0
    if backend == "scipy":
        USE_SCIPY = 1
        USE_TORCH = 0
        USE_NUMPY = 0
        USE_TORCH_GPU = 0
    elif backend == "torch":
        USE_SCIPY = 0
        USE_TORCH = 1
        USE_NUMPY = 0
        USE_TORCH_GPU = 0
    elif backend == "numpy":
        USE_SCIPY = 0
        USE_TORCH = 0
        USE_NUMPY = 1
        USE_TORCH_GPU = 0
    elif backend == "torch_gpu":
        USE_SCIPY = 0
        USE_TORCH = 0
        USE_NUMPY = 0
        USE_TORCH_GPU = 1
        USE_GPU = 1
    else:
        raise ValueError("Invalid backend: {}".format(backend))
