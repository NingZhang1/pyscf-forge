import unittest
import pyscf.isdf.BackEnd._config as CONFIG

CONFIG.backend("torch_gpu")
CONFIG.disable_pyscf_lib()
from pyscf.isdf.BackEnd.isdf_memory_allocator import (
    SimpleMemoryAllocator,
)


class TestSimpleMemoryAllocator(unittest.TestCase):
    def setUp(self):
        self.total_size = 1000
        self.allocator = SimpleMemoryAllocator(self.total_size, gpu=True)

    def test_initialization(self):
        self.assertEqual(self.allocator.total_size, self.total_size)
        self.assertEqual(self.allocator.offset, 0)
        self.assertEqual(len(self.allocator.allocations), 0)

    def test_malloc(self):
        arr1 = self.allocator.malloc((10, 10))
        self.assertEqual(self.allocator.offset, 100)
        self.assertEqual(len(self.allocator.allocations), 1)

        arr2 = self.allocator.malloc((5, 5), name="test_arr")
        self.assertEqual(self.allocator.offset, 125)
        self.assertEqual(len(self.allocator.allocations), 2)
        self.assertEqual(self.allocator.allocations[-1][0], "test_arr")

        self.allocator.free_all()

    def test_free(self):
        self.allocator.malloc((10, 10), name="arr1")
        self.allocator.malloc((5, 5), name="arr2")
        self.allocator.malloc((3, 3), name="arr3")

        self.allocator.free("arr3")
        self.assertEqual(self.allocator.offset, 125)
        self.assertEqual(len(self.allocator.allocations), 2)

        self.allocator.free(count=2)
        self.assertEqual(self.allocator.offset, 0)
        self.assertEqual(len(self.allocator.allocations), 0)

        self.allocator.free_all()

    def test_memory_error(self):
        with self.assertRaises(MemoryError):
            self.allocator.malloc((1000, 1000))  # This should exceed total_size

    def test_value_error(self):
        with self.assertRaises(ValueError):
            self.allocator.free("non_existent_arr")

        with self.assertRaises(ValueError):
            self.allocator.free(count=5)  # Trying to free more allocations than exist

    def test_str_representation(self):
        self.allocator.malloc((10, 10))
        expected_str = (
            f"SimpleMemoryAllocator(total_size={self.total_size}, used=100, free=900)"
        )
        self.assertEqual(str(self.allocator), expected_str)


if __name__ == "__main__":
    unittest.main()
