from pyscf.isdf.BackEnd._num_threads import num_threads

try:
    import pyfftw

    pyfftw.interfaces.cache.enable()
    pyfftw.config.NUM_THREADS = num_threads()
    pyfftw.config.PLANNER_EFFORT = "FFTW_ESTIMATE"

    FFTW_FOUND = True

    # interface #

    def rfftn(
        x, s=None, axes=None, overwrite_input=False, threads=pyfftw.config.NUM_THREADS
    ):
        return pyfftw.interfaces.scipy_fft.rfftn(
            x, s, axes, overwrite_x=overwrite_input, workers=threads
        )

    def irfftn(
        x, s=None, axes=None, overwrite_input=False, threads=pyfftw.config.NUM_THREADS
    ):
        return pyfftw.interfaces.scipy_fft.irfftn(
            x, s, axes, overwrite_x=overwrite_input, workers=threads
        )

    def fftn(
        x, s=None, axes=None, overwrite_input=False, threads=pyfftw.config.NUM_THREADS
    ):
        return pyfftw.interfaces.scipy_fft.fftn(
            x, s, axes, overwrite_x=overwrite_input, workers=threads
        )

    def ifftn(
        x, s=None, axes=None, overwrite_input=False, threads=pyfftw.config.NUM_THREADS
    ):
        return pyfftw.interfaces.scipy_fft.ifftn(
            x, s, axes, overwrite_x=overwrite_input, workers=threads
        )

except ImportError:

    FFTW_FOUND = False
