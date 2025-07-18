import collections
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from modlamp.descriptors import GlobalDescriptor
from collections import Counter



def get_GAAC(seq):
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
            #code.append(0)
    else:
        for t in dipeptide:

            fts['GDPC_'+t]=(myDict[t] / sum)

    return fts

def get_GTPC(seq):
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

def get_grouped_aa_features(features):
    gaac = get_GAAC(features['sequence'])
    gdpc = get_GDPC(features['sequence'])
    gdtp = get_GTPC(features['sequence'])
    features.update(gaac)
    features.update(gdpc)
    features.update(gdtp)
    return features

def get_global_features(features):
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
    def CalculateKSCTriad(sequence, gap, features, AADict):
        res = []
        for g in range(gap+1):
            myDict = {}
            for f in features:
                myDict[f] = 0

            for i in range(len(sequence)):
                if i+gap+1 < len(sequence) and i+2*gap+2<len(sequence):
                    fea = AADict[sequence[i]] + '.' + AADict[sequence[i+gap+1]]+'.'+AADict[sequence[i+2*gap+2]]
                    myDict[fea] = myDict[fea] + 1

            maxValue, minValue = max(myDict.values()), min(myDict.values())
            for f in features:
                if maxValue == 0:
                    res.append(0)  # Evitar división por cero
                else:
                    res.append((myDict[f] - minValue) / maxValue)

        return res


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
    seq = features['sequence']
    desc = CalculateKSCTriad(seq, 0, feats, AADict)
    feats = ['CTriad_'+feat for feat in feats]
    ctriad_dic = {feats[i]:desc[i] for i in range(len(desc))}
    features.update(ctriad_dic)
    return features

def get_features(seq):
    """This function receives a seqIO sequence container as inpunt and returns a feature
    dictionary."""
    features = collections.OrderedDict()
    features['sequence'] = seq
    features['length'] = len(seq)
    features = get_global_features(features)
    features = get_grouped_aa_features(features)
    features = get_ctriad_features(features)

    return features



def get_features_df(sequences_df):
  # Calculamos los descriptores
  features_list = []

  for index, peptido in sequences_df.iterrows():
        features = get_features(peptido['sequence'])
        features_list.append(features)
  

  data_features = pd.DataFrame(features_list)
  return data_features

if __name__ == "__main__":
    import pandas as pd
    # DataFrame de ejemplo con secuencias
    data = pd.DataFrame({
        'sequence': [
            'ARNDCEQGHILKMFPSTWYV',
            'ACDEFGHIKLMNPQRSTVWY',
            'MFPSTWYVARNDCEQGHILK',
            'GGGGGGGGGGGGGGGGGGGG',  # Secuencia repetitiva
        ]
    })
    print("DataFrame de entrada:")
    print(data)
    print("\nCalculando descriptores...")
    features_df = get_features_df(data)
    print("\nDataFrame de descriptores:")
    print(features_df.head())