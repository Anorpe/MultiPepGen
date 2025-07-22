import pytest
import numpy as np
from multipepgen.utils import postprocessing

def test_one_hot_max_basic():
    arr = np.array([0.1, 0.5, 0.2])
    result = postprocessing.one_hot_max(arr)
    assert np.array_equal(result, np.array([0, 1, 0]))
    arr2 = np.array([3, 1, 3])
    result2 = postprocessing.one_hot_max(arr2)
    assert np.array_equal(result2, np.array([1, 0, 1]))
    # Empty array should raise
    with pytest.raises(ValueError):
        postprocessing.one_hot_max([])

def test_one_hot_max_matrix_basic():
    mat = np.array([[0.1, 0.5, 0.2], [2, 2, 1]])
    result = postprocessing.one_hot_max_matrix(mat)
    expected = np.array([[0, 1, 0], [1, 1, 0]])
    assert np.array_equal(result, expected)
    # Not 2D or empty should raise
    with pytest.raises(ValueError):
        postprocessing.one_hot_max_matrix([])
    with pytest.raises(ValueError):
        postprocessing.one_hot_max_matrix(np.array([1, 2, 3])) 