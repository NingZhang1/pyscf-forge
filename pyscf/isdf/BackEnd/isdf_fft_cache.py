import pyscf.isdf.BackEnd.isdf_backend as BackEnd
import numpy as np

NUM_THREADS = BackEnd.NUM_THREADS
ToTENSOR = BackEnd._toTensor
ToNUMPY = BackEnd._toNumpy
USE_GPU = BackEnd.USE_GPU
USE_TORCH = BackEnd.USE_TORCH == 1 or BackEnd.USE_TORCH_GPU == 1

if BackEnd.ENABLE_FFTW:

    try:
        import pyfftw

        FFTW_FOUND = True
    except ImportError:
        FFTW_FOUND = False
else:
    FFTW_FOUND = False

if FFTW_FOUND:

    # implement FFTW obj #

    class DynamicCached3DRFFT:
        def __init__(self, initial_shape, num_threads=NUM_THREADS):

            self.num_threads = num_threads
            self.plans = {}
            self._maxsize = 0
            self.real_buffer = None
            self.complex_buffer = None
            self._allocate_buffers(initial_shape)

        @property
        def memory(self):
            return self.real_buffer.nbytes + self.complex_buffer.nbytes

        def _allocate_buffers(self, shape):

            self.current_shape = shape
            assert len(shape) == 3 or len(shape) == 4, "shape must have 3 or 4 elements"

            size = np.prod(shape)
            if size > self._maxsize:
                del self.real_buffer
                del self.complex_buffer
                self.real_buffer = ToNUMPY(pyfftw.empty_aligned(size, dtype="float64"))
                if len(shape) == 3:
                    complex_shape = (shape[0], shape[1], shape[2] // 2 + 1)
                else:
                    complex_shape = (shape[0], shape[1], shape[2], shape[3] // 2 + 1)
                size_complex = np.prod(complex_shape)
                self.complex_buffer = ToNUMPY(
                    pyfftw.empty_aligned(size_complex, dtype="complex128")
                )
                self._maxsize = size
                self._rebuild_plans()

        def _rebuild_plans(self):
            shapes = set(shape for (shape, _) in self.plans.keys())
            self.plans = {}
            for shape in shapes:
                self._create_plan(shape, "FFTW_FORWARD")
                self._create_plan(shape, "FFTW_BACKWARD")

        def _create_plan(self, shape, direction):
            if len(shape) == 3:
                complex_shape = (shape[0], shape[1], shape[2] // 2 + 1)
                axis = (0, 1, 2)
            else:
                complex_shape = (shape[0], shape[1], shape[2], shape[3] // 2 + 1)
                axis = (1, 2, 3)
            real_buf = np.ndarray(shape, dtype=np.float64, buffer=self.real_buffer)
            complex_buf = np.ndarray(
                complex_shape, dtype=np.complex128, buffer=self.complex_buffer
            )
            if direction == "FFTW_FORWARD":

                self.plans[(shape, direction)] = pyfftw.FFTW(
                    real_buf,
                    complex_buf,
                    axes=axis,
                    direction="FFTW_FORWARD",
                    flags=["FFTW_ESTIMATE"],
                    threads=self.num_threads,
                )
            else:  # 'FFTW_BACKWARD'
                self.plans[(shape, direction)] = pyfftw.FFTW(
                    complex_buf,
                    real_buf,
                    axes=axis,
                    direction="FFTW_BACKWARD",
                    flags=["FFTW_ESTIMATE"],
                    threads=self.num_threads,
                )

        def _get_or_create_plan(self, shape, direction):

            self._allocate_buffers(shape)

            if (shape, direction) not in self.plans:
                self._create_plan(shape, direction)

            return self.plans[(shape, direction)]

        def rfft(self, input_array):
            input_array = ToNUMPY(input_array)
            shape = input_array.shape
            plan = self._get_or_create_plan(shape, "FFTW_FORWARD")
            real_buf = np.ndarray(shape, dtype=np.float64, buffer=self.real_buffer)
            if (
                input_array.__array_interface__["data"][0]
                != real_buf.__array_interface__["data"][0]
            ):
                real_buf[:] = input_array
            # else:
            #     print("rfft: input_array is real_buf")
            return ToTENSOR(plan())

        def irfft(self, input_array, s=None):
            input_array = ToNUMPY(input_array)
            if s is None:
                raise ValueError("s must be provided")
            plan = self._get_or_create_plan(s, "FFTW_BACKWARD")
            if len(s) == 3:
                assert input_array.shape == (s[0], s[1], s[2] // 2 + 1)
            else:
                assert input_array.shape == (s[0], s[1], s[2], s[3] // 2 + 1)
            complex_buf = np.ndarray(
                input_array.shape, dtype=np.complex128, buffer=self.complex_buffer
            )
            if (
                input_array.__array_interface__["data"][0]
                != complex_buf.__array_interface__["data"][0]
            ):
                raise ValueError("input_array is not complex_buf")
                complex_buf[:] = input_array
            # else:
            #     print("irfft: input_array is complex_buf")
            return ToTENSOR(plan())

else:

    if USE_TORCH:

        import torch

        class DynamicCached3DRFFT:
            def __init__(self, initial_shape, num_threads=None, device="cpu"):
                self.num_threads = None
                self.plans = {}
                self.device = device
                self._maxsize = 0
                self.real_buffer = None
                self.complex_buffer = None
                self._allocate_buffers(initial_shape)

            @property
            def memory(self):
                return self.real_buffer.nbytes + self.complex_buffer.nbytes

            def _allocate_buffers(self, shape):
                self.current_shape = shape
                assert (
                    len(shape) == 3 or len(shape) == 4
                ), "shape must have 3 or 4 elements"
                size = torch.prod(torch.tensor(shape))
                if size > self._maxsize:
                    del self.real_buffer
                    del self.complex_buffer
                    self.real_buffer = torch.empty(
                        size, dtype=torch.float64, device=self.device
                    )
                    if len(shape) == 3:
                        complex_shape = (shape[0], shape[1], shape[2] // 2 + 1)
                    else:
                        complex_shape = (
                            shape[0],
                            shape[1],
                            shape[2],
                            shape[3] // 2 + 1,
                        )
                    size_complex = torch.prod(torch.tensor(complex_shape))
                    self.complex_buffer = torch.empty(
                        size_complex, dtype=torch.complex128, device=self.device
                    )
                    self._maxsize = size
                    self._rebuild_plans()

            def _rebuild_plans(self):
                # PyTorch doesn't use plans like FFTW, so we'll just clear the cache
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            def _get_or_create_plan(self, shape, direction):
                self._allocate_buffers(shape)
                # PyTorch doesn't need explicit plans, so we just return None
                return None

            def rfft(self, input_array):
                if self.device == "cuda":
                    input_array = input_array.cuda()
                shape = input_array.shape
                self._get_or_create_plan(shape, "FFTW_FORWARD")
                real_buf = self.real_buffer[: torch.prod(torch.tensor(shape))].view(
                    shape
                )
                if input_array.data_ptr() != real_buf.data_ptr():
                    real_buf.copy_(input_array)
                # allocate out #
                if len(shape) == 3:
                    complex_shape = (shape[0], shape[1], shape[2] // 2 + 1)
                else:
                    assert len(shape) == 4
                    complex_shape = (shape[0], shape[1], shape[2], shape[3] // 2 + 1)
                out = self.complex_buffer[
                    : torch.prod(torch.tensor(complex_shape))
                ].view(complex_shape)
                # perform #
                return torch.fft.rfftn(
                    real_buf, dim=(-3, -2, -1), norm="backward", out=out
                )

            def irfft(self, input_array, s=None):
                if s is None:
                    raise ValueError("s must be provided")
                if self.device == "cuda":
                    input_array = input_array.cuda()
                self._get_or_create_plan(s, "FFTW_BACKWARD")
                if len(s) == 3:
                    assert input_array.shape == (s[0], s[1], s[2] // 2 + 1)
                else:
                    assert input_array.shape == (s[0], s[1], s[2], s[3] // 2 + 1)
                complex_buf = self.complex_buffer[
                    : torch.prod(torch.tensor(input_array.shape))
                ].view(input_array.shape)
                if input_array.data_ptr() != complex_buf.data_ptr():
                    raise ValueError("input_array is not complex_buf")
                    complex_buf.copy_(input_array)
                # allocate out #
                out = self.real_buffer[: torch.prod(torch.tensor(s))].view(s)
                # perform #
                return torch.fft.irfftn(
                    complex_buf, s=s[-3:], dim=(-3, -2, -1), norm="backward", out=out
                )

    else:

        RFFT = BackEnd._rfftn
        IRFFT = BackEnd._irfftn

        class DynamicCached3DRFFT:
            def __init__(self, initial_shape, num_threads=NUM_THREADS):
                pass

            @property
            def memory(self):
                return 0

            def rfft(self, input_array):
                shape = input_array.shape
                if len(shape) == 3:
                    return RFFT(input_array, axes=(0, 1, 2))
                else:
                    return RFFT(input_array, axes=(1, 2, 3))

            def irfft(self, input_array, s=None):
                if s is None:
                    raise ValueError("s must be provided")
                if len(s) == 3:
                    return IRFFT(input_array, s=s, axes=(0, 1, 2))
                else:
                    return IRFFT(input_array, s=s[1:], axes=(1, 2, 3))


if USE_GPU:

    USE_TORCH_GPU = BackEnd.USE_TORCH_GPU
    assert USE_TORCH_GPU == 1 and not FFTW_FOUND

    class DynamicCached3DRFFT_GPU(DynamicCached3DRFFT):
        def __init__(self, initial_shape, num_threads=None):
            super().__init__(initial_shape, num_threads=num_threads, device="cuda")
