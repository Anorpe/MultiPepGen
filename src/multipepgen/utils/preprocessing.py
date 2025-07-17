"""
Utility module for preprocessing and postprocessing data in MultiPepGen.

Includes functions to prepare data before training and to process model predictions.
"""

from sklearn.preprocessing import OneHotEncoder
import numpy as np

valid_aminoacids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P','S','T','W','Y', 'V','_'] #+ ['B','Z','J']

  # Initialize the OneHotEncoder
X = np.array(valid_aminoacids)
ohe = OneHotEncoder(sparse_output = False)
ohe.fit(X.reshape(-1,1))

def complete_sequence(sequence,max_len):
  """

  Complete the sequence with "_" character until it has a number of max_len

  parameters
  ----------
  sequence: sequence
  max_len: maximum row length

  return
  -------
  sequence: string

  """
  agregate = int(max_len - len(sequence))

  return sequence + "_"*agregate

def encoding(data):
  """

    applies a one hot encoding to a set of sequences

    parameters
    ----------
    data: Dataframe containing the sequences

    return
    -------
    ohes: array

  """
  
  sequences = list(complete_sequence(i,35) for i in data['sequence'])
  ohes = []
  for sequence in sequences:
      
      coding = ohe.transform(np.array(list(sequence)).reshape(-1,1))
      ohes.append(coding.reshape((35,21,1)))
    
  return np.array(ohes,dtype = "float32")


def preprocess_data(data):
    """
    Realiza el preprocesamiento necesario sobre los datos de entrada antes del entrenamiento.
    Por ejemplo: limpieza, normalización, codificación, etc.
    
    Args:
        data: Datos de entrada (por ejemplo, DataFrame de pandas o array de numpy)
    Returns:
        Datos preprocesados, listos para el modelo.
    """
    # TODO: Implementar lógica de preprocesamiento
    return data
