"""
Utility module for postprocessing data in MultiPepGen.

Includes functions to process model predictions.
"""

import numpy as np

def escalon(array):
  """

    Converts an array to a binary array, where all of its values are set to zero except the maximum value which is converted to one

    parameters
    ----------
    array: list to transform

    return
    -------
    array: array

  """
  maximo = max(array)
  escalon = []
  for i in array:
    if i == maximo:
      escalon.append(1)
    else:
      escalon.append(0)

  return np.array(escalon)


def escalon_matrix(matrix):
  """

    Apply the "escalon" function on a matrix

    parameters
    ----------
    matrix: matrix to tranform

    return
    -------
    matrix: list

  """
  escalon_matrix = []
  for array in matrix:
    escalon_matrix.append(np.array(escalon(array)))
  return escalon_matrix