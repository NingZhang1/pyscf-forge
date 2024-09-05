from pyscf.isdf.BackEnd.isdf_backend import _malloc as MALLOC
from pyscf.isdf.BackEnd.isdf_backend import FLOAT64, USE_GPU, ITEM_SIZE
from pyscf.isdf.BackEnd.isdf_backend import _prod as PRODUCT
from pyscf.isdf.BackEnd.isdf_backend import _clean as CLEAN


class SimpleMemoryAllocator:
    def __init__(self, total_size, dtype=FLOAT64, gpu=False):
        self.total_size = total_size  # in terms of dtype
        self.dtype = dtype
        self._itemsize = ITEM_SIZE[dtype]
        self.gpu = gpu
        if gpu:
            assert USE_GPU == 1, "GPU is not enabled in the backend"
        self.offset = 0
        self.allocations = []
        self.buffer = MALLOC((total_size,), dtype=dtype, gpu=gpu)

        self._name_format = "_CHUNK_%d"
        self._unnamed_chunk_count = 0

    def malloc(self, shape, dtype=FLOAT64, name=None):

        if name is None:
            name = self._name_format % self._unnamed_chunk_count
            self._unnamed_chunk_count += 1

        size = PRODUCT(shape)

        # Check if there's enough memory
        if self.offset + size > self.total_size:
            print("offnow  = ", self.offset)
            print("size    = ", size)
            print("totsize = ", self.total_size)
            raise MemoryError("Not enough memory in the allocator")

        arr = MALLOC(
            shape,
            dtype=dtype,
            gpu=self.gpu,
            buf=self.buffer,
            offset=self.offset * self._itemsize,
        )
        print("malloc %d at %d with name %s" % (size, self.offset, name))
        # Update offset and allocation records
        self.offset += size
        self.allocations.append((name, size, arr))

        return arr

    def free(self, name=None, count=1):
        if name is None:
            # Free the last 'count' memory chunks
            for _ in range(count):
                if self.allocations:
                    name, size, _ = self.allocations.pop()
                    self.offset -= size
                    print("free   %d at %d with name %s" % (size, self.offset, name))
                else:
                    raise ValueError("No allocations to free")
        else:
            if isinstance(name, str):
                name = [name]
            assert len(name) == count
            # Free memory by name
            if len(self.allocations) < count:
                raise ValueError("Not enough allocations to verify")

            # Verify the names of the last few memory chunks
            for i in range(1, count + 1):
                if self.allocations[-i][0] not in name:
                    raise ValueError(
                        f"Allocation name mismatch: expected {name}, got {self.allocations[-i][0]}"
                    )

            # Free memory
            freed_size = sum(alloc[1] for alloc in self.allocations[-count:])
            self.allocations = self.allocations[:-count]
            self.offset -= freed_size

    def clean(self):
        CLEAN(self.buffer)

    def free_all(self):
        self.free(count=len(self.allocations))

    def __str__(self):
        return f"SimpleMemoryAllocator(total_size={self.total_size}, used={self.offset}, free={self.total_size - self.offset})"
