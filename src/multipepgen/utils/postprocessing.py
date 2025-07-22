"""
Utility module for postprocessing data in MultiPepGen.

Includes functions to process model predictions.
"""

import numpy as np

def one_hot_max(array):
    """
    Converts an array to a binary array, where only the maximum value(s) is set to one and the rest to zero.

    Parameters
    ----------
    array : array-like
        Input 1D array.

    Returns
    -------
    np.ndarray
        Binary array with one at the position(s) of the maximum value.

    Raises
    ------
    ValueError
        If the input array is empty.

    Example
    -------
    >>> one_hot_max([0.1, 0.5, 0.2])
    array([0, 1, 0])
    >>> one_hot_max(np.array([3, 1, 3]))
    array([1, 0, 1])
    """
    array = np.asarray(array)
    if array.size == 0:
        raise ValueError("Input array must not be empty.")
    return (array == array.max()).astype(int)

def one_hot_max_matrix(matrix):
    """
    Apply the one_hot_max function row-wise to a matrix.

    Parameters
    ----------
    matrix : array-like
        2D input matrix (n_rows, n_cols).

    Returns
    -------
    np.ndarray
        2D binary matrix of the same shape as input.

    Raises
    ------
    ValueError
        If the input matrix is empty or not 2D.

    Example
    -------
    >>> one_hot_max_matrix([[0.1, 0.5, 0.2], [2, 2, 1]])
    array([[0, 1, 0],
           [1, 1, 0]])
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("Input must be a non-empty 2D array.")
    return np.apply_along_axis(one_hot_max, 1, matrix)
    