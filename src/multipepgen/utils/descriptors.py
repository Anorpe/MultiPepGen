import collections
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import GlobalDescriptor
from collections import Counter
import warnings

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def get_GAAC(seq):
    """
    Calculate the Grouped Amino Acid Composition (GAAC) features for a peptide sequence.

    Parameters
    ----------
    seq : str
        Amino acid sequence.

    Returns
    -------
    dict
        Dictionary with GAAC feature names as keys and their normalized frequencies as values.

    Example
    -------
    >>> get_GAAC('ACDEFGHIKLMNPQRSTVWY')
    {'GAAC_aliphatic': ..., 'GAAC_aromatic': ..., ...}
    """
    group = {
        'GAAC_aliphatic': 'GAVLMI',
        'GAAC_aromatic': 'FYW',
        'GAAC_positivecharge': 'KRH',
        'GAAC_negativecharge': 'DE',
        'GAAC_uncharged': 'STCPNQ'
    }
    fts = {}
    groupKey = group.keys()
    count = Counter(seq)
    myDict = {}
    for key in groupKey:
        for aa in group[key]:
            myDict[key] = myDict.get(key, 0) + count[aa]
    for key in groupKey:
        fts[key]=(myDict[key]/len(seq))
    return fts

def get_GDPC(seq):
    """
    Calculate the Grouped Dipeptide Composition (GDPC) features for a peptide sequence.

    Parameters
    ----------
    seq : str
        Amino acid sequence.

    Returns
    -------
    dict
        Dictionary with GDPC feature names as keys and their normalized frequencies as values.

    Example
    -------
    >>> get_GDPC('ACDEFGHIKLMNPQRSTVWY')
    {'GDPC_aliphatic.aliphatic': ..., 'GDPC_aliphatic.aromatic': ..., ...}
    """
    group = {
        'aliphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'positivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharged': 'STCPNQ'
    }
    groupKey = group.keys()
    baseNum = len(groupKey)
    dipeptide = [g1 + '.' + g2 for g1 in groupKey for g2 in groupKey]
    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key
    fts = {}
    myDict = {}
    for t in dipeptide:
        myDict[t] = 0
    sum = 0
    for j in range(len(seq) - 2 + 1):
        myDict[index[seq[j]] + '.' + index[seq[j + 1]]] = myDict[index[seq[j]] + '.' + index[
            seq[j + 1]]] + 1
        sum = sum + 1
    if sum == 0:
        for t in dipeptide:
            fts['GDPC_'+t]=0
    else:
        for t in dipeptide:
            fts['GDPC_'+t]=(myDict[t] / sum)
    return fts

def get_GTPC(seq):
    """
    Calculate the Grouped Tripeptide Composition (GTPC) features for a peptide sequence.

    Parameters
    ----------
    seq : str
        Amino acid sequence.

    Returns
    -------
    dict
        Dictionary with GTPC feature names as keys and their normalized frequencies as values.

    Example
    -------
    >>> get_GTPC('ACDEFGHIKLMNPQRSTVWY')
    {'GTPC_aliphatic.aliphatic.aliphatic': ..., ...}
    """
    if len(seq) < 3:
        warnings.warn(f"Sequence too short for GTPC features: '{seq}'")
        return {f'GTPC_{k1}.{k2}.{k3}': 0 for k1 in ['aliphatic','aromatic','positivecharge','negativecharge','uncharged'] for k2 in ['aliphatic','aromatic','positivecharge','negativecharge','uncharged'] for k3 in ['aliphatic','aromatic','positivecharge','negativecharge','uncharged']}
    group = {
        'aliphatic': 'GAVLMI',
        'aromatic': 'FYW',
        'positivecharge': 'KRH',
        'negativecharge': 'DE',
        'uncharged': 'STCPNQ'
    }
    groupKey = group.keys()
    baseNum = len(groupKey)
    triple = [g1+'.'+g2+'.'+g3 for g1 in groupKey for g2 in groupKey for g3 in groupKey]
    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key
    fts = {}
    myDict = {}
    for t in triple:
        myDict[t] = 0
    sum = 0
    for j in range(len(seq) - 3 + 1):
        myDict[index[seq[j]]+'.'+index[seq[j+1]]+'.'+index[seq[j+2]]] = myDict[index[seq[j]]+'.'+index[seq[j+1]]+'.'+index[seq[j+2]]] + 1
        sum = sum +1
    if sum == 0:
        for t in triple:
            fts['GTPC_'+t]=0
    else:
        for t in triple:
            fts['GTPC_'+t]=(myDict[t]/sum)
    return fts

def CalculateKSCTriad(sequence, gap, features, AADict):
    """
    Calculate the C-Triad features for a sequence with a given gap.
    """
    res = []
    for g in range(gap+1):
        myDict = {f: 0 for f in features}
        for i in range(len(sequence)):
            if i+gap+1 < len(sequence) and i+2*gap+2<len(sequence):
                fea = AADict[sequence[i]] + '.' + AADict[sequence[i+gap+1]]+'.'+AADict[sequence[i+2*gap+2]]
                myDict[fea] = myDict[fea] + 1
        maxValue, minValue = max(myDict.values()), min(myDict.values())
        for f in features:
            if maxValue == 0:
                res.append(0)  # Avoid division by zero
            else:
                res.append((myDict[f] - minValue) / maxValue)
    return res

def get_grouped_aa_features(features):
    """
    Add grouped amino acid composition features (GAAC, GDPC, GTPC) to a feature dictionary.

    Parameters
    ----------
    features : dict
        Feature dictionary with at least the key 'sequence'.

    Returns
    -------
    dict
        Updated feature dictionary including grouped amino acid features.
    """
    gaac = get_GAAC(features['sequence'])
    gdpc = get_GDPC(features['sequence'])
    gdtp = get_GTPC(features['sequence'])
    features.update(gaac)
    features.update(gdpc)
    features.update(gdtp)
    return features

def get_global_features(features):
    """
    Add global physicochemical features to a feature dictionary using modlamp and Bio.SeqUtils.

    Parameters
    ----------
    features : dict
        Feature dictionary with at least the key 'sequence'.

    Returns
    -------
    dict
        Updated feature dictionary including global features.
    """
    seq = features['sequence']
    desc = GlobalDescriptor(seq)
    desc.calculate_MW(amide=False)
    features['molecular_weight'] = desc.descriptor[0][0]
    desc.calculate_charge(ph=7.0)
    features['charge'] = desc.descriptor[0][0]
    desc.charge_density(ph = 7.0)
    features['charge_density'] = desc.descriptor[0][0]
    biop_analysis = ProteinAnalysis(seq)
    features['isoelectric_point'] = biop_analysis.isoelectric_point()
    features['gravy'] = biop_analysis.gravy()
    desc.instability_index()
    features['instability_index'] = desc.descriptor[0][0]
    desc.aromaticity()
    features['aromaticity'] = desc.descriptor[0][0]
    desc.aliphatic_index()
    features['aliphatic_index'] = desc.descriptor[0][0]
    desc.boman_index()
    features['boman_index'] = desc.descriptor[0][0]
    desc.hydrophobic_ratio()
    features['hydrophobic_ratio'] = desc.descriptor[0][0]
    return features

def get_ctriad_features(features):
    """
    Add C-Triad features to a feature dictionary.

    Parameters
    ----------
    features : dict
        Feature dictionary with at least the key 'sequence'.

    Returns
    -------
    dict
        Updated feature dictionary including C-Triad features.
    """
    seq = features['sequence']
    if len(seq) < 3:
        warnings.warn(f"Sequence too short for C-Triad features: '{seq}'")
        return features
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
    desc = CalculateKSCTriad(seq, 0, feats, AADict)
    feats = ['CTriad_'+feat for feat in feats]
    ctriad_dic = {feats[i]:desc[i] for i in range(len(desc))}
    features.update(ctriad_dic)
    return features

def get_features(seq):
    """
    Compute all features for a given amino acid sequence.

    Parameters
    ----------
    seq : str
        Amino acid sequence.

    Returns
    -------
    collections.OrderedDict
        Feature dictionary with all computed features.

    Example
    -------
    >>> get_features('ACDEFGHIKLMNPQRSTVWY')
    OrderedDict([...])
    """
    features = collections.OrderedDict()
    features['sequence'] = seq
    features['length'] = len(seq)
    features = get_global_features(features)
    features = get_grouped_aa_features(features)
    features = get_ctriad_features(features)
    return features

def get_features_df(sequences_df, show_progress: bool = False):
    """
    Compute features for all sequences in a DataFrame.

    Parameters
    ----------
    sequences_df : pandas.DataFrame
        DataFrame with a 'sequence' column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with computed features for each sequence.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'sequence': ['ACDEFGHIKLMNPQRSTVWY']})
    >>> get_features_df(df)
    ...
    """
    if 'sequence' not in sequences_df.columns:
        raise ValueError("Input DataFrame must contain a 'sequence' column.")
    features_list = []
    iterator = sequences_df.iterrows()
    if show_progress and TQDM_AVAILABLE:
        iterator = tqdm(iterator, total=len(sequences_df), desc="Extracting features")
    for _, row in iterator:
        seq = row['sequence']
        if not isinstance(seq, str):
            warnings.warn(f"Sequence must be a string, got {type(seq)}. Skipping.")
            continue
        try:
            features = get_features(seq)
            features_list.append(features)
        except Exception as e:
            warnings.warn(f"Error processing sequence '{seq}': {e}. Skipping.")
    return pd.DataFrame(features_list)

if __name__ == "__main__":
    import pandas as pd
    # Minimal test
    data = pd.DataFrame({
        'sequence': [
            'ARNDCEQGHILKMFPSTWYV',
            'ACDEFGHIKLMNPQRSTVWY',
            'MFPSTWYVARNDCEQGHILK',
            'GGGGGGGGGGGGGGGGGGGG',
        ]
    })
    print("Input DataFrame:")
    print(data)
    print("\nCalculating descriptors...")
    features_df = get_features_df(data, show_progress=True)
    print("\nDescriptors DataFrame:")
    print(features_df.head())