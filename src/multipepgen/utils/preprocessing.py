"""
Utility module for preprocessing and postprocessing data in MultiPepGen.

Includes functions to prepare data before training and to process model predictions.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import OneHotEncoder
from multipepgen.config import LABELS


valid_aminoacids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P','S','T','W','Y', 'V','_'] #+ ['B','Z','J']
set_valid_aminoacids = set(valid_aminoacids)


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


def preprocess_data(df, batch_size=32):
    """
    Realiza el preprocesamiento necesario sobre los datos de entrada antes del entrenamiento.
    
    Args:
        df: Datos de entrada : DataFrame
    Retorna
    -------
      dataset : tf.Dataset
    """
    data = df.drop(LABELS, axis = 'columns')
    target = np.array(df[LABELS].values,dtype = "float32")
    data_ohe = encoding(data)
    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(data_ohe,dtype = tf.float32), target))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset





def amp_filter(seq, valid_aas = set_valid_aminoacids , min_aas = 7, max_aas = 35, min_unique_aas = 3, verbose = False):
    """
    Input:
        -seq: aminoacid sequence.
        -min_aas: minimum number of aas for sequence.
        -max_aas: maximum number of aas for sequence.
        -valid_aas: a list with the aminoacids to be included.
        -min_unique_aas = minimum number of unique aas for sequence.

    Output:
        >False if 'seq' does not comply with any of the criteria.
        >True if 'seq' complies with all of the criteria.
    """

    if len(seq) > max_aas or len(seq) < min_aas:
        if verbose:
            print("Discarded ", seq, " due to length ", len(seq))
        return False
    set_diff = set(seq)-valid_aas
    if len(set_diff)>0:
        if verbose:
            print("Discarded ", seq, " due to ", set_diff)
        return False


    if len(set(seq)) < min_unique_aas:
        if verbose:
            print("Discarded ", seq, " due to low number of different aas")
        return False
    return True


def replace_bzj(seq, verbose = False):
    res = seq.replace('B', random.sample(['N','D'], 1)[0])
    res = res.replace('Z', random.sample(['Q','E'], 1)[0])
    res = res.replace('J', random.sample(['I','L'], 1)[0])
    if res != seq and verbose:
        print(seq, "--->", res)
    return res


def amp_filter_df(df, valid_aas = set_valid_aminoacids , min_aas = 7, max_aas = 35, min_unique_aas = 3, verbose = False):
    rows = []
    for index, sequence in df.iterrows():
        if amp_filter(sequence["sequence"], valid_aas, min_aas, max_aas, min_unique_aas, verbose):
            rows.append({"sequence": sequence["sequence"]})
    df_result = pd.DataFrame(rows)
    df_result = df_result.drop_duplicates(["sequence"])
    return df_result