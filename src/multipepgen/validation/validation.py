from __future__ import absolute_import, division, print_function
# =========================
# Imports estándar y de terceros
# =========================
import warnings
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.stats import ks_2samp
from difflib import SequenceMatcher
from Bio import Align
from typing import List, Optional, Tuple, Union
# NOTA: joblib, amp_filter_df, get_features_df, preprocessing_df, selected_features_xgboost, selection_robust deben estar definidos/importados

# =========================
# 1. Pruebas de Hipótesis y Estadísticas
# =========================
def ks_2samp_test(
    data1,
    data2,
    alpha: float = 0.05,
    return_stat: bool = False
) -> Union[bool, Tuple[bool, float]]:
    """
    Realiza la prueba de Kolmogorov-Smirnov (KS) de dos muestras para comparar si ambas provienen de la misma distribución.

    Parámetros
    ----------
    data1 : array-like
        Primera muestra de datos.
    data2 : array-like
        Segunda muestra de datos.
    alpha : float, opcional
        Nivel de significancia para la prueba (por defecto 0.05).
    return_stat : bool, opcional
        Si es True, retorna también el estadístico KS además del resultado booleano.

    Retorna
    -------
    bool o (bool, float)
        True si no se rechaza la hipótesis nula (las muestras pueden provenir de la misma distribución), False en caso contrario.
        Si return_stat es True, también retorna el estadístico KS.
    """
    stat, p = ks_2samp(data1, data2)
    resultado = p > alpha
    if return_stat:
        return resultado, stat
    return resultado


def ks_2samp_score(descriptors1, descriptors2, alpha: float = 0.05) -> tuple[float, list]:
    """
    Calcula el porcentaje y la lista de descriptores que cumplen la prueba de Kolmogorov-Smirnov (ks_2samp).

    Parámetros
    ----------
    descriptors1 : pd.DataFrame
        DataFrame con los descriptores del primer conjunto de datos.
    descriptors2 : pd.DataFrame
        DataFrame con los descriptores del segundo conjunto de datos.
    alpha : float, opcional
        Nivel de significancia para la prueba KS (por defecto 0.05).

    Retorna
    -------
    score : float
        Porcentaje de columnas/descriptores que no rechazan la hipótesis nula (distribuciones similares).
    passed_columns : list
        Lista de nombres de columnas que cumplen la prueba KS.
    """
    passed_columns = []
    total_columns = len(descriptors1.columns)
    for column in descriptors1.columns:
        resultado = ks_2samp_test(descriptors1[column], descriptors2[column], alpha=alpha)
        if resultado:
            passed_columns.append(column)
    score = len(passed_columns) / total_columns if total_columns > 0 else 0.0
    return score, passed_columns

# =========================
# 2. Métricas de Diversidad
# =========================
def repeat_score(data: pd.DataFrame) -> float:
    """
    Calcula el porcentaje de secuencias duplicadas en un DataFrame.

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame que debe contener una columna 'sequence' con las secuencias a analizar.

    Retorna
    -------
    float
        Proporción de secuencias duplicadas respecto al total.
    """
    len_ini = data.shape[0]
    data_unique = data.drop_duplicates(subset=["sequence"])
    len_uniques = data_unique.shape[0]
    return (len_ini - len_uniques) / len_ini


def intersect_score(data_base: pd.DataFrame, data_sample: pd.DataFrame) -> float:
    """
    Calcula el porcentaje de secuencias en data_sample que están presentes en data_base.

    Parámetros
    ----------
    data_base : pd.DataFrame
        DataFrame base con columna 'sequence'.
    data_sample : pd.DataFrame
        DataFrame de muestra con columna 'sequence'.

    Retorna
    -------
    float
        Proporción de secuencias de data_sample que están en data_base.
    """
    data_base_list = set(list(data_base["sequence"]))
    data_sample_list = set(list(data_sample["sequence"]))
    intersection = data_base_list & data_sample_list
    return len(intersection) / len(data_sample_list)

# =========================
# 3. Métricas de Similitud de Secuencias (SequenceMatcher)
# =========================
def sequence_matcher_max_ratio(sequence: str, data2: list[str]) -> float:
    """
    Calcula la máxima similitud rápida (quick_ratio) entre una secuencia y una lista de secuencias usando SequenceMatcher.

    Parámetros
    ----------
    sequence : str
        Secuencia a comparar.
    data2 : list of str
        Lista de secuencias contra las que se compara.

    Retorna
    -------
    float
        Máximo valor de similitud rápida encontrado.
    """
    ratio = 0
    for sequence2 in data2:
        ratio = max(SequenceMatcher(None, sequence, sequence2).quick_ratio(), ratio)
    return ratio


def sequence_matcher_avg_ratio(sequence: str, data2: list[str]) -> float:
    """
    Calcula el promedio de similitud rápida (quick_ratio) entre una secuencia y una lista de secuencias usando SequenceMatcher.

    Parámetros
    ----------
    sequence : str
        Secuencia a comparar.
    data2 : list of str
        Lista de secuencias contra las que se compara.

    Retorna
    -------
    float
        Promedio de similitud rápida.
    """
    ratio_sum = 0
    for sequence2 in data2:
        ratio_sum += SequenceMatcher(None, sequence, sequence2).quick_ratio()
    return ratio_sum / len(data2)


def sequences_matcher_avg_ratio(data1: list[str], data2: Optional[List[str]] = None) -> Tuple[float, List[float]]:
    """
    Calcula el promedio de las similitudes promedio entre secuencias de data1 y data2 (o entre sí mismas si data2 es None).

    Parámetros
    ----------
    data1 : list of str
        Lista de secuencias a comparar.
    data2 : list of str, opcional
        Lista de referencia. Si es None, se compara cada secuencia de data1 con el resto de data1.

    Retorna
    -------
    float
        Promedio de similitud.
    list of float
        Lista de similitudes individuales.
    """
    if data2 is None:
        ratio_sum = 0
        ratios = []
        data = list(data1)
        for sequence in data1:
            data_aux = list(data1.copy())
            data_aux.remove(sequence)
            aux = sequence_matcher_avg_ratio(sequence, data_aux)
            ratio_sum += aux
            ratios.append(aux)
        return ratio_sum / len(data1), ratios
    else:
        ratio_sum = 0
        ratios = []
        for sequence1 in data1:
            aux = sequence_matcher_avg_ratio(sequence1, data2)
            ratio_sum += aux
            ratios.append(aux)
        return ratio_sum / len(data1), ratios


def sequences_matcher_max_ratio(data1: list[str], data2: Optional[List[str]] = None) -> float:
    """
    Calcula el promedio de las máximas similitudes entre secuencias de data1 y data2 (o entre sí mismas si data2 es None).

    Parámetros
    ----------
    data1 : list of str
        Lista de secuencias a comparar.
    data2 : list of str, opcional
        Lista de referencia. Si es None, se compara cada secuencia de data1 con el resto de data1.

    Retorna
    -------
    float
        Promedio de la máxima similitud encontrada para cada secuencia.
    """
    if data2 is None:
        ratio_sum = 0
        for sequence in data1:
            data_aux = list(data1.copy())
            data_aux.remove(sequence)
            ratio_sum += sequence_matcher_max_ratio(sequence, data_aux)
        return ratio_sum / len(data1)
    else:
        ratio_sum = 0
        for sequence1 in data1:
            ratio_sum += sequence_matcher_max_ratio(sequence1, data2)
        return ratio_sum / len(data1)

# =========================
# 4. Métricas de Similitud de Secuencias (Alineamiento)
# =========================
aligner = Align.PairwiseAligner()

def sequence_align_max_ratio(sequence: str, data2: list[str]) -> float:
    """
    Calcula la máxima puntuación de alineamiento entre una secuencia y una lista de secuencias usando PairwiseAligner.

    Parámetros
    ----------
    sequence : str
        Secuencia a comparar.
    data2 : list of str
        Lista de secuencias contra las que se compara.

    Retorna
    -------
    float
        Máxima puntuación de alineamiento encontrada.
    """
    ratio = 0
    for sequence2 in data2:
        ratio = max(aligner.align(sequence, sequence2).score, ratio)
    return ratio


def sequence_align_avg_ratio(sequence: str, data2: list[str]) -> float:
    """
    Calcula el promedio de puntuaciones de alineamiento entre una secuencia y una lista de secuencias usando PairwiseAligner.

    Parámetros
    ----------
    sequence : str
        Secuencia a comparar.
    data2 : list of str
        Lista de secuencias contra las que se compara.

    Retorna
    -------
    float
        Promedio de puntuaciones de alineamiento.
    """
    ratio_sum = 0
    for sequence2 in data2:
        ratio_sum += aligner.align(sequence, sequence2).score
    return ratio_sum / len(data2)


def sequences_align_avg_ratio(data1: list[str], data2: Optional[List[str]] = None) -> Tuple[float, List[float]]:
    """
    Calcula el promedio de las puntuaciones promedio de alineamiento entre secuencias de data1 y data2 (o entre sí mismas si data2 es None).

    Parámetros
    ----------
    data1 : list of str
        Lista de secuencias a comparar.
    data2 : list of str, opcional
        Lista de referencia. Si es None, se compara cada secuencia de data1 con el resto de data1.

    Retorna
    -------
    float
        Promedio de puntuaciones de alineamiento.
    list of float
        Lista de puntuaciones individuales.
    """
    if data2 is None:
        ratio_sum = 0
        ratios = []
        data = list(data1)
        for sequence in data1:
            data_aux = list(data1.copy())
            data_aux.remove(sequence)
            aux = sequence_align_avg_ratio(sequence, data_aux)
            ratio_sum += aux
            ratios.append(aux)
        return ratio_sum / len(data1), ratios
    else:
        ratio_sum = 0
        ratios = []
        for sequence1 in data1:
            aux = sequence_align_avg_ratio(sequence1, data2)
            ratio_sum += aux
            ratios.append(aux)
        return ratio_sum / len(data1), ratios


def sequences_align_max_ratio(data1: list[str], data2: Optional[List[str]] = None) -> float:
    """
    Calcula el promedio de las máximas puntuaciones de alineamiento entre secuencias de data1 y data2 (o entre sí mismas si data2 es None).

    Parámetros
    ----------
    data1 : list of str
        Lista de secuencias a comparar.
    data2 : list of str, opcional
        Lista de referencia. Si es None, se compara cada secuencia de data1 con el resto de data1.

    Retorna
    -------
    float
        Promedio de la máxima puntuación de alineamiento encontrada para cada secuencia.
    """
    if data2 is None:
        ratio_sum = 0
        for sequence in data1:
            data_aux = list(data1.copy())
            data_aux.remove(sequence)
            ratio_sum += sequence_align_max_ratio(sequence, data_aux)
        return ratio_sum / len(data1)
    else:
        ratio_sum = 0
        for sequence1 in data1:
            ratio_sum += sequence_align_max_ratio(sequence1, data2)
        return ratio_sum / len(data1)

# =========================
# 5. Frechet Distance (FID)
# =========================
def frechet_distance(descriptores1: np.ndarray, descriptores2: np.ndarray, eps: float = 1e-6, axis: int = 0) -> float:
    """
    Calcula la Frechet Distance (FID) entre dos conjuntos de descriptores.

    Parámetros
    ----------
    descriptores1 : np.ndarray
        Matriz de descriptores del primer conjunto.
    descriptores2 : np.ndarray
        Matriz de descriptores del segundo conjunto.
    eps : float, opcional
        Pequeño valor para estabilidad numérica.
    axis : int, opcional
        Eje sobre el que calcular la media y covarianza.

    Retorna
    -------
    float
        Distancia de Frechet entre los dos conjuntos.
    """
    mu1 = np.mean(descriptores1, axis=axis)
    sigma1 = np.cov(descriptores1, rowvar=not bool(axis))
    mu2 = np.mean(descriptores2, axis=axis)
    sigma2 = np.cov(descriptores2, rowvar=not bool(axis))
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

# =========================
# 6. Predicción y Validación (requiere dependencias externas)
# =========================
# NOTA: joblib, selected_features_xgboost, amp_filter_df, get_features_df, preprocessing_df, selection_robust deben estar definidos/importados

def prediction_score(data_seq_descrip) -> tuple[float, np.ndarray]:
    """
    Predice la probabilidad de pertenencia a la clase positiva usando un modelo XGBoost previamente entrenado.

    Parámetros
    ----------
    data_seq_descrip : pd.DataFrame
        DataFrame con los descriptores de las secuencias.

    Retorna
    -------
    float
        Promedio de las probabilidades predichas.
    np.ndarray
        Vector de probabilidades predichas para cada muestra.
    """
    scaler = joblib.load('models/MinMaxScaler.pkl')
    xgboost = joblib.load('models/xgboost_train.pkl')
    predictions = []
    data_seq_descrip = data_seq_descrip[selected_features_xgboost]
    data_seq_descrip_scaler = scaler.transform(data_seq_descrip)
    predictions = xgboost.predict_proba(data_seq_descrip_scaler)
    predictions_proba = np.array([x[1] for x in predictions])
    return predictions_proba.mean(), predictions_proba


def validation_score(data, data_robust, data_minmax, data_seq):
    """
    Calcula un conjunto de métricas de validación y calidad para un conjunto de secuencias generadas.

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame de referencia (original).
    data_robust : pd.DataFrame
        DataFrame de descriptores robustos de referencia.
    data_minmax : pd.DataFrame
        DataFrame de descriptores minmax de referencia.
    data_seq : pd.DataFrame
        DataFrame de secuencias generadas.

    Retorna
    -------
    scores : dict
        Diccionario con métricas de validación y calidad.
    scores_df : dict
        Diccionario con resultados detallados por secuencia.
    """
    scores = {}
    generated_sequences = data_seq['sequence'].tolist()
    len_data_seq = data_seq.shape[0]
    # Diversidad
    scores["repeat"] = repeat_score(data_seq)
    scores["intersect"] = intersect_score(data, data_seq)
    scores["sequence_matcher"], sequence_matchers = sequences_matcher_avg_ratio(data_seq['sequence'], data['sequence'])
    scores["sequence_matcher_self"], sequence_matchers_self = sequences_matcher_avg_ratio(data_seq['sequence'])
    # Filtrado de secuencias válidas (definir amp_filter_df)
    data_seq = amp_filter_df(data_seq)
    scores["valid_sequences"] = data_seq.shape[0] / len_data_seq
    scores["align"], aligns = sequences_align_avg_ratio(data_seq['sequence'], data['sequence'])
    scores["align_self"], aligns_self = sequences_align_avg_ratio(data_seq['sequence'])
    # Calidad
    data_seq_descrip = get_features_df(data_seq)
    minmax = joblib.load('models/minmax_generator.pkl')
    data_seq_descrip_minmax = preprocessing_df(data_seq_descrip, minmax)
    robust = joblib.load('models/robust_generator.pkl')
    data_seq_descrip_robust = preprocessing_df(data_seq_descrip, robust)
    scores["ks_2samp"], ks_2samp_columns = ks_2samp_score(data_robust[selection_robust], data_seq_descrip_robust[selection_robust])
    scores["fretech"] = frechet_distance(
        data_minmax.drop(["sequence"], axis='columns').sample(data_seq.shape[0]),
        data_seq_descrip_minmax.drop(["sequence"], axis='columns')
    )
    scores["prediction"], predictions = prediction_score(data_seq_descrip)
    scores_df = {
        "sequence": generated_sequences,
        "predictions": predictions,
        "aligns": aligns,
        "aligns_self": aligns_self,
        "sequence_matchers": sequence_matchers,
        "sequence_matchers_self": sequence_matchers_self,
        "ks_2samp_columns": ks_2samp_columns
    }
    return scores, scores_df