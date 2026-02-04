"""
Utility module for data preprocessing in MultiPepGen.

Includes functions to prepare data before training and to process model predictions.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.preprocessing import OneHotEncoder
from multipepgen.config import LABELS, VALID_AMINOACIDS, SET_VALID_AMINOACIDS, DEFAULT_CONFIG
from multipepgen.utils.logger import logger

DEF_MAX_LEN = DEFAULT_CONFIG['data']['sequence_length']

# Initialize the OneHotEncoder
X = np.array(VALID_AMINOACIDS)
ohe = OneHotEncoder(sparse_output=False)
ohe.fit(X.reshape(-1, 1))

def complete_sequence(sequence: str, max_len: int) -> str:
    """
    Pads the input sequence with '_' characters until it reaches the specified maximum length.

    Args:
        sequence (str): The amino acid sequence to pad.
        max_len (int): The desired length of the output sequence.

    Returns:
        str: The padded sequence of length max_len.

    Raises:
        ValueError: If the input sequence is longer than max_len.
    """
    if len(sequence) > max_len:
        raise ValueError(f"Input sequence length ({len(sequence)}) exceeds max_len ({max_len}).")
    return sequence + '_' * (max_len - len(sequence))

def encode_sequences(df: pd.DataFrame, max_len: int = DEF_MAX_LEN) -> np.ndarray:
    """
    Applies one-hot encoding to a set of amino acid sequences in a DataFrame.

    Each sequence is padded to max_len and encoded using a fitted OneHotEncoder.

    Args:
        df (pd.DataFrame): DataFrame containing a 'sequence' column with amino acid sequences.
        max_len (int, optional): Maximum sequence length for padding. Defaults to 35.

    Returns:
        np.ndarray: Array of shape (n_samples, max_len, n_aminoacids, 1) with one-hot encoded sequences.

    Raises:
        KeyError: If the DataFrame does not contain a 'sequence' column.
        ValueError: If any sequence is longer than max_len.
    """
    if 'sequence' not in df.columns:
        raise KeyError("DataFrame must contain a 'sequence' column.")
    sequences = [complete_sequence(seq, max_len) for seq in df['sequence']]
    encoded = [
        ohe.transform(np.array(list(seq)).reshape(-1, 1)).reshape((max_len, len(VALID_AMINOACIDS), 1))
        for seq in sequences
    ]
    return np.array(encoded, dtype="float32")

def preprocess_data(df: pd.DataFrame, batch_size: int = 32, max_len: int = DEF_MAX_LEN) -> tf.data.Dataset:
    """
    Preprocesses input data for model training, including one-hot encoding and batching.

    Args:
        df (pd.DataFrame): Input DataFrame containing features and target labels.
        batch_size (int, optional): Batch size for the TensorFlow dataset. Defaults to 32.
        max_len (int, optional): Maximum sequence length for encoding. Defaults to 35.

    Returns:
        tf.data.Dataset: A shuffled and batched TensorFlow dataset with encoded features and targets.

    Raises:
        KeyError: If LABELS are not present in the DataFrame.
    """
    data = df.drop(LABELS, axis='columns')
    targets = np.array(df[LABELS].values, dtype="float32")
    data_ohe = encode_sequences(data, max_len)
    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(data_ohe, dtype=tf.float32), targets))
    dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)
    return dataset

def filter_amp(
    sequence: str,
    valid_aas: set = SET_VALID_AMINOACIDS,
    min_aas: int = 7,
    max_aas: int = DEF_MAX_LEN,
    min_unique_aas: int = 3,
    verbose: bool = False
) -> bool:
    """
    Checks if an amino acid sequence meets length and diversity criteria.

    Args:
        sequence (str): Amino acid sequence to be evaluated.
        valid_aas (set, optional): Set of valid amino acids. Defaults to SET_VALID_AMINOACIDS.
        min_aas (int, optional): Minimum allowed sequence length. Defaults to 7.
        max_aas (int, optional): Maximum allowed sequence length. Defaults to 35.
        min_unique_aas (int, optional): Minimum number of unique amino acids required. Defaults to 3.
        verbose (bool, optional): If True, prints the reason for discarding. Defaults to False.

    Returns:
        bool: True if the sequence meets all criteria, False otherwise.
    """
    if not (min_aas <= len(sequence) <= max_aas):
        if verbose:
            logger.debug(f"Discarded {sequence} due to length {len(sequence)}")
        return False
    set_diff = set(sequence) - valid_aas
    if set_diff:
        if verbose:
            logger.debug(f"Discarded {sequence} due to containing {set_diff}")
        return False
    if len(set(sequence)) < min_unique_aas:
        if verbose:
            logger.debug(f"Discarded {sequence} due to low amino acid diversity")
        return False
    return True

def replace_bzj(sequence: str, verbose: bool = False) -> str:
    """
    Replaces ambiguous amino acid codes B, Z, and J with random valid alternatives.

    B is replaced by N or D, Z by Q or E, and J by I or L.

    Args:
        sequence (str): Amino acid sequence to process.
        verbose (bool, optional): If True, prints the change made. Defaults to False.

    Returns:
        str: Modified sequence with ambiguous codes replaced.
    """
    res = sequence.replace('B', random.choice(['N', 'D']))
    res = res.replace('Z', random.choice(['Q', 'E']))
    res = res.replace('J', random.choice(['I', 'L']))
    if res != sequence and verbose:
        logger.info(f"Substituted: {sequence} ---> {res}")
    return res

def filter_amp_df(
    df: pd.DataFrame,
    valid_aas: set = SET_VALID_AMINOACIDS,
    min_aas: int = 7,
    max_aas: int = DEF_MAX_LEN,
    min_unique_aas: int = 3,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Filters a DataFrame of amino acid sequences using filter_amp criteria.

    Args:
        df (pd.DataFrame): DataFrame with a 'sequence' column containing amino acid sequences.
        valid_aas (set, optional): Set of valid amino acids. Defaults to SET_VALID_AMINOACIDS.
        min_aas (int, optional): Minimum allowed sequence length. Defaults to 7.
        max_aas (int, optional): Maximum allowed sequence length. Defaults to 35.
        min_unique_aas (int, optional): Minimum number of unique amino acids required. Defaults to 3.
        verbose (bool, optional): If True, prints the reason for discarding. Defaults to False.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid and unique sequences.

    Raises:
        KeyError: If the DataFrame does not contain a 'sequence' column.
    """
    if 'sequence' not in df.columns:
        raise KeyError("DataFrame must contain a 'sequence' column.")
    filtered = [
        {"sequence": str(row["sequence"])}
        for _, row in df.iterrows()
        if filter_amp(str(row["sequence"]), valid_aas, min_aas, max_aas, min_unique_aas, verbose)
    ]
    df_result = pd.DataFrame(filtered).drop_duplicates(["sequence"])
    return df_result