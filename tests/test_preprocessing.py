import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
from multipepgen.utils import preprocessing

# Test for complete_sequence
def test_complete_sequence_padding():
    assert preprocessing.complete_sequence('ACD', 5) == 'ACD__'
    assert preprocessing.complete_sequence('', 3) == '___'
    assert preprocessing.complete_sequence('ACDE', 4) == 'ACDE'
    with pytest.raises(ValueError):
        preprocessing.complete_sequence('ACDEFG', 3)

# Test for encode_sequences
def test_encode_sequences_shape():
    df = pd.DataFrame({'sequence': ['ACD', 'GGG']})
    arr = preprocessing.encode_sequences(df, max_len=5)
    assert arr.shape == (2, 5, len(preprocessing.VALID_AMINOACIDS), 1)
    assert arr.dtype == np.float32
    # Test error if 'sequence' column missing
    with pytest.raises(KeyError):
        preprocessing.encode_sequences(pd.DataFrame({'seq': ['ACD']}))
    # Test error if sequence too long
    df2 = pd.DataFrame({'sequence': ['A'*10]})
    with pytest.raises(ValueError):
        preprocessing.encode_sequences(df2, max_len=5)

# Test for preprocess_data
def test_preprocess_data_tf_dataset():
    df = pd.DataFrame({
        'sequence': ['ACD', 'GGG'],
        'microbiano': [1, 0],
        'bacteriano': [0, 1],
        'antigramneg': [0, 0],
        'antigrampos': [1, 1],
        'fungico': [0, 0],
        'viral': [0, 0],
        'cancer': [0, 1],
    })
    ds = preprocessing.preprocess_data(df, batch_size=1, max_len=5)
    batches = list(ds)
    assert len(batches) == 2
    x, y = batches[0]
    assert x.shape == (1, 5, len(preprocessing.VALID_AMINOACIDS), 1)
    assert y.shape == (1, 7)

# Test for filter_amp
def test_filter_amp_basic():
    assert preprocessing.filter_amp('ACDEFGH') is True
    assert preprocessing.filter_amp('A'*3) is False  # too short
    assert preprocessing.filter_amp('ACDEFGH'*10) is False  # too long
    assert preprocessing.filter_amp('AAAAAAA') is False  # not enough unique
    assert preprocessing.filter_amp('ACDXYZ') is False  # invalid chars

# Test for replace_bzj
def test_replace_bzj():
    # B, Z, J replaced by valid alternatives
    for _ in range(10):
        out = preprocessing.replace_bzj('ABZJ')
        assert set(out).issubset(set(preprocessing.VALID_AMINOACIDS))
        assert len(out) == 4
    # No change if not present
    assert preprocessing.replace_bzj('ACDE') == 'ACDE'

# Test for filter_amp_df
def test_filter_amp_df():
    df = pd.DataFrame({'sequence': ['ACDEFGH', 'AAAAAAA', 'ACDXYZ', 'ACDEFGH']})
    filtered = preprocessing.filter_amp_df(df)
    # Only one valid and unique sequence should remain
    assert filtered.shape[0] == 1
    assert filtered.iloc[0]['sequence'] == 'ACDEFGH' 