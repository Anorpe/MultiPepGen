import pytest
import pandas as pd
import numpy as np
import warnings
from multipepgen.utils import descriptors

def test_get_GAAC():
    seq = 'ACDEFGHIKLMNPQRSTVWY'
    result = descriptors.get_GAAC(seq)
    assert isinstance(result, dict)
    assert all(k.startswith('GAAC_') for k in result)
    assert abs(sum(result.values()) - 1.0) < 1e-6

def test_get_GDPC():
    seq = 'ACDEFGHIKLMNPQRSTVWY'
    result = descriptors.get_GDPC(seq)
    assert isinstance(result, dict)
    assert all(k.startswith('GDPC_') for k in result)
    assert abs(sum(result.values()) - 1.0) < 1e-6 or sum(result.values()) == 0.0

def test_get_GTPC():
    seq = 'ACDEFGHIKLMNPQRSTVWY'
    result = descriptors.get_GTPC(seq)
    assert isinstance(result, dict)
    assert all(k.startswith('GTPC_') for k in result)
    # If sequence is too short, all values should be 0
    short_result = descriptors.get_GTPC('AC')
    assert all(v == 0 for v in short_result.values())

def test_get_features():
    seq = 'ACDEFGHIKLMNPQRSTVWY'
    features = descriptors.get_features(seq)
    assert isinstance(features, dict)
    assert 'sequence' in features
    assert 'length' in features
    assert 'molecular_weight' in features

def test_get_features_df():
    df = pd.DataFrame({'sequence': ['ACDEFGHIKLMNPQRSTVWY', 'GGGGGGGGGG']})
    features_df = descriptors.get_features_df(df)
    assert isinstance(features_df, pd.DataFrame)
    assert 'sequence' in features_df.columns
    assert 'length' in features_df.columns
    # Test error if 'sequence' column missing
    with pytest.raises(ValueError):
        descriptors.get_features_df(pd.DataFrame({'seq': ['ACD']}))
    # Test warning if sequence is not string
    df2 = pd.DataFrame({'sequence': [123, 'ACD']})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        descriptors.get_features_df(df2)
        assert any("Sequence must be a string" in str(warn.message) for warn in w)

def test_CalculateKSCTriad():
    seq = 'ACDEFGHIKLMNPQRSTVWY'
    # Setup for C-Triad
    AAGroup = {
        'g1': 'AGV',
        'g2': 'ILFP',
        'g3': 'YMTS',
        'g4': 'HNQW',
        'g5': 'RK',
        'g6': 'DE',
        'g7': 'C'
    }
    myGroups = sorted(AAGroup.keys())
    AADict = {}
    for g in myGroups:
        for aa in AAGroup[g]:
            AADict[aa] = g
    feats = [f1 + '.'+ f2 + '.' + f3 for f1 in myGroups for f2 in myGroups for f3 in myGroups]
    res = descriptors.CalculateKSCTriad(seq, 0, feats, AADict)
    assert isinstance(res, list)
    assert len(res) == len(feats)

def test_get_grouped_aa_features():
    features = {'sequence': 'ACDEFGHIKLMNPQRSTVWY'}
    out = descriptors.get_grouped_aa_features(features.copy())
    assert any(k.startswith('GAAC_') for k in out)
    assert any(k.startswith('GDPC_') for k in out)
    assert any(k.startswith('GTPC_') for k in out)

def test_get_global_features():
    features = {'sequence': 'ACDEFGHIKLMNPQRSTVWY'}
    out = descriptors.get_global_features(features.copy())
    assert 'molecular_weight' in out
    assert 'charge' in out
    assert 'isoelectric_point' in out
    assert 'gravy' in out

def test_get_ctriad_features():
    features = {'sequence': 'ACDEFGHIKLMNPQRSTVWY'}
    out = descriptors.get_ctriad_features(features.copy())
    assert any(k.startswith('CTriad_') for k in out)
    # Test warning for short sequence
    features_short = {'sequence': 'AC'}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out2 = descriptors.get_ctriad_features(features_short.copy())
        assert out2 == features_short
        assert any("Sequence too short for C-Triad features" in str(warn.message) for warn in w) 