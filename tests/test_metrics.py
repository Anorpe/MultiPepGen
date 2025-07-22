import pytest
import numpy as np
import pandas as pd
from multipepgen.validation import metrics

def test_ks_2samp_test():
    assert metrics.ks_2samp_test([1,2,3], [1,2,3])
    # Usar datos claramente distintos para asegurar que el test rechace la hipótesis nula
    assert not metrics.ks_2samp_test([1,1,1,1,1], [100,100,100,100,100])
    res, stat = metrics.ks_2samp_test([1,2,3], [1,2,3], return_stat=True)
    assert isinstance(res, (bool, np.bool_))
    assert isinstance(stat, float)

def test_ks_2samp_score():
    df1 = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    df2 = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    score, passed = metrics.ks_2samp_score(df1, df2)
    assert score == 1.0
    assert set(passed) == {'a', 'b'}

def test_repeat_score():
    df = pd.DataFrame({'sequence': ['AAA', 'AAA', 'AAC', 'AAG']})
    score = metrics.repeat_score(df)
    assert 0.0 <= score <= 1.0

def test_intersect_score():
    df1 = pd.DataFrame({'sequence': ['AAA', 'AAC', 'AAG']})
    df2 = pd.DataFrame({'sequence': ['AAA', 'AAT', 'AAG']})
    score = metrics.intersect_score(df1, df2)
    assert 0.0 <= score <= 1.0

def test_sequence_matcher_max_ratio():
    seq = 'AAA'
    seq_list = ['AAA', 'AAC', 'AAG']
    ratio = metrics.sequence_matcher_max_ratio(seq, seq_list)
    assert 0.0 <= ratio <= 1.0

def test_sequence_matcher_avg_ratio():
    seq = 'AAA'
    seq_list = ['AAA', 'AAC', 'AAG']
    ratio = metrics.sequence_matcher_avg_ratio(seq, seq_list)
    assert 0.0 <= ratio <= 1.0

def test_sequences_matcher_avg_ratio():
    seqs = ['AAA', 'AAC', 'AAG']
    avg, ratios = metrics.sequences_matcher_avg_ratio(seqs)
    assert 0.0 <= avg <= 1.0
    assert len(ratios) == len(seqs)
    avg2, ratios2 = metrics.sequences_matcher_avg_ratio(seqs, ['AAA', 'AAT'])
    assert 0.0 <= avg2 <= 1.0
    assert len(ratios2) == len(seqs)

def test_sequences_matcher_max_ratio():
    seqs = ['AAA', 'AAC', 'AAG']
    avg = metrics.sequences_matcher_max_ratio(seqs)
    assert 0.0 <= avg <= 1.0
    avg2 = metrics.sequences_matcher_max_ratio(seqs, ['AAA', 'AAT'])
    assert 0.0 <= avg2 <= 1.0

def test_sequence_align_max_ratio():
    seq = 'AAA'
    seq_list = ['AAA', 'AAC', 'AAG']
    ratio = metrics.sequence_align_max_ratio(seq, seq_list)
    assert ratio >= 0.0

def test_sequence_align_avg_ratio():
    seq = 'AAA'
    seq_list = ['AAA', 'AAC', 'AAG']
    ratio = metrics.sequence_align_avg_ratio(seq, seq_list)
    assert ratio >= 0.0

def test_sequences_align_avg_ratio():
    seqs = ['AAA', 'AAC', 'AAG']
    avg, ratios = metrics.sequences_align_avg_ratio(seqs)
    assert avg >= 0.0
    assert len(ratios) == len(seqs)
    avg2, ratios2 = metrics.sequences_align_avg_ratio(seqs, ['AAA', 'AAT'])
    assert avg2 >= 0.0
    assert len(ratios2) == len(seqs)

def test_sequences_align_max_ratio():
    seqs = ['AAA', 'AAC', 'AAG']
    avg = metrics.sequences_align_max_ratio(seqs)
    assert avg >= 0.0
    avg2 = metrics.sequences_align_max_ratio(seqs, ['AAA', 'AAT'])
    assert avg2 >= 0.0

def test_frechet_distance():
    a = np.random.rand(10, 5)
    b = np.random.rand(10, 5)
    dist = metrics.frechet_distance(a, b)
    assert isinstance(dist, float)
    assert dist >= 0.0 