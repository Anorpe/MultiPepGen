import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import ks_2samp
from difflib import SequenceMatcher
from Bio import Align
from typing import List, Optional, Tuple, Union

from multipepgen.utils.preprocessing import filter_amp_df
from multipepgen.utils.descriptors import get_features_df
from multipepgen.utils.logger import logger

def _ensure_list(x):
    """
    Convert pandas Series or DataFrame column to list if needed.

    Parameters
    ----------
    x : pandas.Series, pandas.DataFrame, or list
        Input to be converted to list if necessary.

    Returns
    -------
    list
        The input as a list.

    Example
    -------
    >>> import pandas as pd
    >>> _ensure_list(pd.Series([1,2,3]))
    [1, 2, 3]
    >>> _ensure_list([1,2,3])
    [1, 2, 3]
    """
    if isinstance(x, pd.Series):
        return x.tolist()
    elif isinstance(x, pd.DataFrame):
        # If DataFrame, try to use the first column
        return x.iloc[:, 0].tolist()
    return x

# =========================
# 1. Diversity Metrics
# =========================
def ks_2samp_test(reference_values, sample_values, alpha: float = 0.05, return_stat: bool = False) -> Union[bool, Tuple[bool, float]]:
    """
    Performs the Kolmogorov-Smirnov (KS) test for two samples to compare if both come from the same distribution.

    Parameters
    ----------
    reference_values : array-like
        Reference data sample.
    sample_values : array-like
        Sample data to compare.
    alpha : float, optional
        Significance level for the test (default 0.05).
    return_stat : bool, optional
        If True, also returns the KS statistic in addition to the boolean result.

    Returns
    -------
    bool or tuple of (bool, float)
        True if the null hypothesis is not rejected (samples may come from the same distribution), False otherwise.
        If return_stat is True, also returns the KS statistic (float).

    Example
    -------
    >>> ks_2samp_test([1,2,3], [1,2,3])
    True
    """
    stat, p = ks_2samp(reference_values, sample_values)
    result = p > alpha
    if return_stat:
        return result, float(stat)
    return result


def ks_2samp_score(reference_descriptors, sample_descriptors, alpha: float = 0.05) -> tuple[float, list]:
    """
    Calculates the percentage and the list of descriptors that pass the Kolmogorov-Smirnov (ks_2samp) test.

    Parameters
    ----------
    reference_descriptors : pandas.DataFrame
        DataFrame with descriptors of the reference dataset.
    sample_descriptors : pandas.DataFrame
        DataFrame with descriptors of the sample dataset.
    alpha : float, optional
        Significance level for the KS test (default 0.05).

    Returns
    -------
    score : float
        Percentage of columns/descriptors that do not reject the null hypothesis (similar distributions).
    passed_columns : list
        List of column names that pass the KS test.

    Raises
    ------
    ValueError
        If either input is not a pandas DataFrame.

    Example
    -------
    >>> import pandas as pd
    >>> df1 = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    >>> df2 = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    >>> ks_2samp_score(df1, df2)
    (1.0, ['a', 'b'])
    """
    if not isinstance(reference_descriptors, pd.DataFrame) or not isinstance(sample_descriptors, pd.DataFrame):
        raise ValueError("Both reference_descriptors and sample_descriptors must be pandas DataFrames.")
    passed_columns = []
    total_columns = len(reference_descriptors.columns)
    for column in reference_descriptors.columns:
        result = ks_2samp_test(reference_descriptors[column], sample_descriptors[column], alpha=alpha)
        if result:
            passed_columns.append(column)
    score = len(passed_columns) / total_columns if total_columns > 0 else 0.0
    return score, passed_columns


def repeat_score(sequence_df: pd.DataFrame) -> float:
    """
    Calculates the percentage of duplicate sequences in a DataFrame.

    Parameters
    ----------
    sequence_df : pandas.DataFrame
        DataFrame that must contain a 'sequence' column.

    Returns
    -------
    float
        Proportion of duplicate sequences with respect to the total.

    Raises
    ------
    ValueError
        If the input is not a DataFrame or does not contain a 'sequence' column.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'sequence': ['AAA', 'BBB', 'AAA']})
    >>> repeat_score(df)
    0.333...
    """
    if not isinstance(sequence_df, pd.DataFrame) or 'sequence' not in sequence_df.columns:
        raise ValueError("Input must be a DataFrame with a 'sequence' column.")
    len_ini = sequence_df.shape[0]
    data_unique = sequence_df.drop_duplicates(subset=["sequence"])
    len_uniques = data_unique.shape[0]
    return (len_ini - len_uniques) / len_ini if len_ini > 0 else 0.0


def intersect_score(reference_df: pd.DataFrame, query_df: pd.DataFrame) -> float:
    """
    Calculates the percentage of sequences in query_df that are present in reference_df.

    Parameters
    ----------
    reference_df : pandas.DataFrame
        Reference DataFrame with 'sequence' column.
    query_df : pandas.DataFrame
        Query DataFrame with 'sequence' column.

    Returns
    -------
    float
        Proportion of sequences in query_df that are in reference_df.

    Raises
    ------
    ValueError
        If either input is not a DataFrame or does not contain a 'sequence' column.

    Example
    -------
    >>> import pandas as pd
    >>> ref = pd.DataFrame({'sequence': ['AAA', 'BBB']})
    >>> qry = pd.DataFrame({'sequence': ['AAA', 'CCC']})
    >>> intersect_score(ref, qry)
    0.5
    """
    if not (isinstance(reference_df, pd.DataFrame) and isinstance(query_df, pd.DataFrame)):
        raise ValueError("Both reference_df and query_df must be DataFrames.")
    if 'sequence' not in reference_df.columns or 'sequence' not in query_df.columns:
        raise ValueError("Both DataFrames must have a 'sequence' column.")
    ref_set = set(list(reference_df["sequence"]))
    qry_set = set(list(query_df["sequence"]))
    intersection = ref_set & qry_set
    return len(intersection) / len(qry_set) if len(qry_set) > 0 else 0.0


def sequence_matcher_max_ratio(sequence: str, sequence_list: list[str]) -> float:
    """
    Calculates the maximum quick_ratio similarity between a sequence and a list of sequences using SequenceMatcher.

    Parameters
    ----------
    sequence : str
        Sequence to compare.
    sequence_list : list of str
        List of sequences to compare against.

    Returns
    -------
    float
        Maximum quick_ratio similarity found.

    Example
    -------
    >>> sequence_matcher_max_ratio('AAA', ['AAB', 'AAC'])
    0.666...
    """
    if not isinstance(sequence_list, list):
        sequence_list = list(sequence_list)
    if len(sequence_list) == 0:
        return 0.0
    ratio = 0
    for sequence2 in sequence_list:
        ratio = max(SequenceMatcher(None, sequence, sequence2).quick_ratio(), ratio)
    return ratio


def sequence_matcher_avg_ratio(sequence: str, sequence_list: list[str]) -> float:
    """
    Calculates the average quick_ratio similarity between a sequence and a list of sequences using SequenceMatcher.

    Parameters
    ----------
    sequence : str
        Sequence to compare.
    sequence_list : list of str
        List of sequences to compare against.

    Returns
    -------
    float
        Average quick_ratio similarity.

    Example
    -------
    >>> sequence_matcher_avg_ratio('AAA', ['AAB', 'AAC'])
    0.666...
    """
    if not isinstance(sequence_list, list):
        sequence_list = list(sequence_list)
    if len(sequence_list) == 0:
        return 0.0
    ratio_sum = 0
    for sequence2 in sequence_list:
        ratio_sum += SequenceMatcher(None, sequence, sequence2).quick_ratio()
    return ratio_sum / len(sequence_list)


def sequences_matcher_avg_ratio(sequence_list: list[str], reference_list: Optional[List[str]] = None) -> Tuple[float, List[float]]:
    """
    Calculates the average of the average similarities between sequences in sequence_list and reference_list (or among themselves if reference_list is None).

    Parameters
    ----------
    sequence_list : list of str or pandas.Series
        List of sequences to compare.
    reference_list : list of str or pandas.Series, optional
        Reference list. If None, each sequence in sequence_list is compared with the rest of sequence_list.

    Returns
    -------
    float
        Average similarity.
    list of float
        List of individual similarities.

    Example
    -------
    >>> sequences_matcher_avg_ratio(['AAA', 'AAB', 'AAC'])
    (0.666..., [...])
    """
    sequence_list = _ensure_list(sequence_list)
    if reference_list is not None:
        reference_list = _ensure_list(reference_list)
    if reference_list is None:
        ratio_sum = 0
        ratios = []
        for sequence in sequence_list:
            data_aux = list(sequence_list)
            data_aux.remove(sequence)
            aux = sequence_matcher_avg_ratio(sequence, data_aux)
            ratio_sum += aux
            ratios.append(aux)
        return ratio_sum / len(sequence_list) if len(sequence_list) > 0 else 0.0, ratios
    else:
        ratio_sum = 0
        ratios = []
        for sequence1 in sequence_list:
            aux = sequence_matcher_avg_ratio(sequence1, reference_list)
            ratio_sum += aux
            ratios.append(aux)
        return ratio_sum / len(sequence_list) if len(sequence_list) > 0 else 0.0, ratios


def sequences_matcher_max_ratio(sequence_list: list[str], reference_list: Optional[List[str]] = None) -> float:
    """
    Calculates the average of the maximum similarities between sequences in sequence_list and reference_list (or among themselves if reference_list is None).

    Parameters
    ----------
    sequence_list : list of str or pandas.Series
        List of sequences to compare.
    reference_list : list of str or pandas.Series, optional
        Reference list. If None, each sequence in sequence_list is compared with the rest of sequence_list.

    Returns
    -------
    float
        Average of the maximum similarity found for each sequence.

    Example
    -------
    >>> sequences_matcher_max_ratio(['AAA', 'AAB', 'AAC'])
    0.833...
    """
    sequence_list = _ensure_list(sequence_list)
    if reference_list is not None:
        reference_list = _ensure_list(reference_list)
    if reference_list is None:
        ratio_sum = 0
        for sequence in sequence_list:
            data_aux = list(sequence_list)
            data_aux.remove(sequence)
            ratio_sum += sequence_matcher_max_ratio(sequence, data_aux)
        return ratio_sum / len(sequence_list) if len(sequence_list) > 0 else 0.0
    else:
        ratio_sum = 0
        for sequence1 in sequence_list:
            ratio_sum += sequence_matcher_max_ratio(sequence1, reference_list)
        return ratio_sum / len(sequence_list) if len(sequence_list) > 0 else 0.0

aligner = Align.PairwiseAligner()

def sequence_align_max_ratio(sequence: str, sequence_list: list[str]) -> float:
    """
    Calculates the maximum alignment score between a sequence and a list of sequences using PairwiseAligner.

    Parameters
    ----------
    sequence : str
        Sequence to compare.
    sequence_list : list of str
        List of sequences to compare against.

    Returns
    -------
    float
        Maximum alignment score found.

    Example
    -------
    >>> sequence_align_max_ratio('AAA', ['AAB', 'AAC'])
    0.666...
    """
    if not isinstance(sequence_list, list):
        sequence_list = list(sequence_list)
    if len(sequence_list) == 0:
        return 0.0
    ratio = 0
    for sequence2 in sequence_list:
        ratio = max(aligner.align(sequence, sequence2).score, ratio)
    return ratio


def sequence_align_avg_ratio(sequence: str, sequence_list: list[str]) -> float:
    """
    Calculates the average alignment score between a sequence and a list of sequences using PairwiseAligner.

    Parameters
    ----------
    sequence : str
        Sequence to compare.
    sequence_list : list of str
        List of sequences to compare against.

    Returns
    -------
    float
        Average alignment score.

    Example
    -------
    >>> sequence_align_avg_ratio('AAA', ['AAB', 'AAC'])
    0.666...
    """
    if not isinstance(sequence_list, list):
        sequence_list = list(sequence_list)
    if len(sequence_list) == 0:
        return 0.0
    ratio_sum = 0
    for sequence2 in sequence_list:
        ratio_sum += aligner.align(sequence, sequence2).score
    return ratio_sum / len(sequence_list)


def sequences_align_avg_ratio(sequence_list: list[str], reference_list: Optional[List[str]] = None) -> Tuple[float, List[float]]:
    """
    Calculates the average of the average alignment scores between sequences in sequence_list and reference_list (or among themselves if reference_list is None).

    Parameters
    ----------
    sequence_list : list of str or pandas.Series
        List of sequences to compare.
    reference_list : list of str or pandas.Series, optional
        Reference list. If None, each sequence in sequence_list is compared with the rest of sequence_list.

    Returns
    -------
    float
        Average of the average alignment scores.
    list of float
        List of individual average alignment scores.

    Example
    -------
    >>> sequences_align_avg_ratio(['AAA', 'AAB', 'AAC'])
    (0.666..., [...])
    """
    sequence_list = _ensure_list(sequence_list)
    if reference_list is not None:
        reference_list = _ensure_list(reference_list)
    if reference_list is None:
        ratio_sum = 0
        ratios = []
        for sequence in sequence_list:
            data_aux = list(sequence_list)
            data_aux.remove(sequence)
            aux = sequence_align_avg_ratio(sequence, data_aux)
            ratio_sum += aux
            ratios.append(aux)
        return ratio_sum / len(sequence_list) if len(sequence_list) > 0 else 0.0, ratios
    else:
        ratio_sum = 0
        ratios = []
        for sequence1 in sequence_list:
            aux = sequence_align_avg_ratio(sequence1, reference_list)
            ratio_sum += aux
            ratios.append(aux)
        return ratio_sum / len(sequence_list) if len(sequence_list) > 0 else 0.0, ratios


def sequences_align_max_ratio(sequence_list: list[str], reference_list: Optional[List[str]] = None) -> float:
    """
    Calculates the average of the maximum alignment scores between sequences in sequence_list and reference_list (or among themselves if reference_list is None).

    Parameters
    ----------
    sequence_list : list of str or pandas.Series
        List of sequences to compare.
    reference_list : list of str or pandas.Series, optional
        Reference list. If None, each sequence in sequence_list is compared with the rest of sequence_list.

    Returns
    -------
    float
        Average of the maximum alignment scores.

    Example
    -------
    >>> sequences_align_max_ratio(['AAA', 'AAB', 'AAC'])
    0.833...
    """
    sequence_list = _ensure_list(sequence_list)
    if reference_list is not None:
        reference_list = _ensure_list(reference_list)
    if reference_list is None:
        ratio_sum = 0
        for sequence in sequence_list:
            data_aux = list(sequence_list)
            data_aux.remove(sequence)
            ratio_sum += sequence_align_max_ratio(sequence, data_aux)
        return ratio_sum / len(sequence_list) if len(sequence_list) > 0 else 0.0
    else:
        ratio_sum = 0
        for sequence1 in sequence_list:
            ratio_sum += sequence_align_max_ratio(sequence1, reference_list)
        return ratio_sum / len(sequence_list) if len(sequence_list) > 0 else 0.0

# =========================
# 5. Quality Metrics
# =========================
def frechet_distance(descriptors1: np.ndarray, descriptors2: np.ndarray, eps: float = 1e-6, axis: int = 0) -> float:
    """
    Calculates the Frechet Distance (FID) between two sets of descriptors.

    Parameters
    ----------
    descriptors1 : numpy.ndarray
        Descriptor matrix of the first set.
    descriptors2 : numpy.ndarray
        Descriptor matrix of the second set.
    eps : float, optional
        Small value for numerical stability.
    axis : int, optional
        Axis over which to calculate the mean and covariance.

    Returns
    -------
    float
        Frechet distance between the two sets.

    Raises
    ------
    ValueError
        If the covariance matrix is not positive semi-definite or has imaginary components.

    Example
    -------
    >>> import numpy as np
    >>> a = np.random.rand(10, 5)
    >>> b = np.random.rand(10, 5)
    >>> frechet_distance(a, b)
    0.0  # (example value)
    """
    mu1 = np.mean(descriptors1, axis=axis)
    sigma1 = np.cov(descriptors1, rowvar=not bool(axis))
    mu2 = np.mean(descriptors2, axis=axis)
    sigma2 = np.cov(descriptors2, rowvar=not bool(axis))
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    # Handle if linalg.sqrtm returns a tuple (matrix, info)
    if isinstance(covmean, tuple):
        covmean = covmean[0]
    if not np.isfinite(covmean).all():
        msg = f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if isinstance(covmean, tuple):
            covmean = covmean[0]
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# =========================
# 6. Prediction and Validation (requires external dependencies)
# =========================
def prediction_score(data_seq_descrip, scaler_path='models/MinMaxScaler.pkl', xgboost_path='models/xgboost_train.pkl', selected_features=None) -> tuple[float, np.ndarray]:
    """
    Predicts the probability of belonging to the positive class using a previously trained XGBoost model.

    Parameters
    ----------
    data_seq_descrip : pandas.DataFrame
        DataFrame with the sequence descriptors.
    scaler_path : str, optional
        Path to the scaler model file (default 'models/MinMaxScaler.pkl').
    xgboost_path : str, optional
        Path to the XGBoost model file (default 'models/xgboost_train.pkl').
    selected_features : list of str
        List of feature names to use for prediction.

    Returns
    -------
    float
        Mean of the predicted probabilities.
    numpy.ndarray
        Vector of predicted probabilities for each sample.

    Raises
    ------
    ImportError
        If joblib is not installed.
    RuntimeError
        If there is an error loading the model files.
    ValueError
        If selected_features is not provided.

    Example
    -------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'f1': [0.1, 0.2], 'f2': [0.3, 0.4]})
    >>> prediction_score(df, scaler_path='scaler.pkl', xgboost_path='model.pkl', selected_features=['f1', 'f2'])
    (0.5, array([0.5, 0.5]))  # (example values)
    """
    try:
        import joblib
    except ImportError:
        raise ImportError("joblib is required for loading models. Please install it.")
    try:
        scaler = joblib.load(scaler_path)
        xgboost = joblib.load(xgboost_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model files: {e}")
    if selected_features is None:
        raise ValueError("selected_features must be provided.")
    data_seq_descrip = data_seq_descrip[selected_features]
    data_seq_descrip_scaler = scaler.transform(data_seq_descrip)
    predictions = xgboost.predict_proba(data_seq_descrip_scaler)
    predictions_proba = np.array([x[1] for x in predictions])
    return predictions_proba.mean(), predictions_proba


def validation_scores(data, data_seq) -> Tuple[dict, dict]:
    """
    Calculates a set of validation and quality metrics for a set of generated sequences.

    Parameters
    ----------
    data : pandas.DataFrame
        Reference DataFrame (original).
    data_seq : pandas.DataFrame
        DataFrame of generated sequences.

    Returns
    -------
    scores : dict
        Dictionary with validation and quality metrics.
    scores_df : dict
        Dictionary with detailed results per sequence.

    Raises
    ------
    ValueError
        If either input is not a DataFrame or does not contain a 'sequence' column.

    Example
    -------
    >>> import pandas as pd
    >>> df_real = pd.DataFrame({'sequence': ['AAA', 'BBB']})
    >>> df_gen = pd.DataFrame({'sequence': ['AAA', 'CCC']})
    >>> scores, scores_df = validation_scores(df_real, df_gen)
    """
    if not (isinstance(data, pd.DataFrame) and isinstance(data_seq, pd.DataFrame)):
        raise ValueError("Both data and data_seq must be pandas DataFrames.")
    if 'sequence' not in data.columns or 'sequence' not in data_seq.columns:
        raise ValueError("Both DataFrames must have a 'sequence' column.")
    scores = {}
    generated_sequences = data_seq['sequence'].tolist()
    len_data_seq = data_seq.shape[0]
    # Diversity metrics
    scores["repeat"] = repeat_score(data_seq)
    scores["intersect"] = intersect_score(data, data_seq)
    scores["sequence_matcher"], sequence_matchers = sequences_matcher_avg_ratio(_ensure_list(data_seq['sequence']), _ensure_list(data['sequence']))
    scores["sequence_matcher_self"], sequence_matchers_self = sequences_matcher_avg_ratio(_ensure_list(data_seq['sequence']))
    # Filtering valid sequences (define filter_amp_df)
    data_seq_valid = filter_amp_df(data_seq)
    scores["valid_sequences"] = data_seq_valid.shape[0] / len_data_seq if len_data_seq > 0 else 0.0
    scores["align"], aligns = sequences_align_avg_ratio(_ensure_list(data_seq_valid['sequence']), _ensure_list(data['sequence']))
    scores["align_self"], aligns_self = sequences_align_avg_ratio(_ensure_list(data_seq_valid['sequence']))
    # Quality metrics
    data_seq_descrip = get_features_df(data_seq_valid)
    data_descrip = get_features_df(data)
    scores["ks_2samp"], ks_2samp_columns = ks_2samp_score(data_descrip, data_seq_descrip)
    scores_df = {
        "sequence": generated_sequences,
        #"predictions": predictions,
        "aligns": aligns,
        "aligns_self": aligns_self,
        "sequence_matchers": sequence_matchers,
        "sequence_matchers_self": sequence_matchers_self,
        "ks_2samp_columns": ks_2samp_columns
    }
    return scores, scores_df

if __name__ == "__main__":
    import pandas as pd
    # Reference DataFrame (original)
    data = pd.DataFrame({
        'sequence': [
            'ARNDCEQGHILKMFPSTWYV',
            'ACDEFGHIKLMNPQRSTVWY',
            'MFPSTWYVARNDCEQGHILK',
            'GGGGGGGGGGGGGGGGGGGG',
        ]
    })
    # DataFrame of generated sequences
    data_seq = pd.DataFrame({
        'sequence': [
            'ARNDCEQGHILKMFPSTWYV',
            'ACDEFGHIKLMNPQRSTVWY',
            'VVVVVVVVVVVVVVVVVVVV',
            'YYYYYYYYYYYYYYYYYYYYY',
        ]
    })
    logger.info("Reference DataFrame:")
    logger.info(data.to_string())
    logger.info("\nGenerated sequences DataFrame:")
    logger.info(data_seq.to_string())
    logger.info("\nCalculating validation metrics...")
    scores, scores_df = validation_scores(data, data_seq)
    logger.info("\nValidation results:")
    print(scores)