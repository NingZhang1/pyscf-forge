import numpy as np


def ToNUMPY(arr):
    return np.array(arr)


def _find_common_elements_positions(arr1, arr2):
    arr1 = ToNUMPY(arr1)
    arr2 = ToNUMPY(arr2)

    # Create a dictionary to store elements and their positions in arr1
    elem_positions = {elem: i for i, elem in enumerate(arr1)}

    # Find common elements and their positions
    common_positions = [
        (elem_positions[elem], i)
        for i, elem in enumerate(arr2)
        if elem in elem_positions
    ]

    # If common elements exist, return two arrays of positions; otherwise, return empty arrays
    if common_positions:
        common_positions = np.array(common_positions).T
        return common_positions[0], common_positions[1]
    else:
        return np.array([], dtype=int), np.array([], dtype=int)


def test_find_common_elements_positions():
    # Test case 1: Identical arrays
    arr1 = [1, 2, 3, 4, 5]
    arr2 = [1, 2, 3, 4, 5]
    pos1, pos2 = _find_common_elements_positions(arr1, arr2)
    assert np.array_equal(pos1, [0, 1, 2, 3, 4])
    assert np.array_equal(pos2, [0, 1, 2, 3, 4])
    print("Test case 1 passed: Identical arrays")

    # Test case 2: Partially overlapping arrays
    arr1 = [1, 2, 3, 4, 5]
    arr2 = [3, 4, 5, 6, 7]
    pos1, pos2 = _find_common_elements_positions(arr1, arr2)
    assert np.array_equal(pos1, [2, 3, 4])
    assert np.array_equal(pos2, [0, 1, 2])
    print("Test case 2 passed: Partially overlapping arrays")

    # Test case 3: No common elements
    arr1 = [1, 2, 3]
    arr2 = [4, 5, 6]
    pos1, pos2 = _find_common_elements_positions(arr1, arr2)
    assert len(pos1) == 0 and len(pos2) == 0
    print("Test case 3 passed: No common elements")

    # Test case 4: Empty arrays
    arr1 = []
    arr2 = []
    pos1, pos2 = _find_common_elements_positions(arr1, arr2)
    assert len(pos1) == 0 and len(pos2) == 0
    print("Test case 4 passed: Empty arrays")

    print("All test cases passed!")


if __name__ == "__main__":
    test_find_common_elements_positions()
