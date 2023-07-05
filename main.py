from pgmpy.models import BayesianNetwork
from pgmpy.estimators import (MaximumLikelihoodEstimator, ParameterEstimator, BayesianEstimator)
from pgmpy.sampling import BayesianModelSampling
from pgmpy.metrics import log_likelihood_score
from pgmpy.inference import VariableElimination
from random import random
import pandas as pd
from math import floor
from typing import List
from data_preparation import (Columns, session)


def full_print(df: pd.DataFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3,):
        print(df)


def half(df: pd.DataFrame) -> int:
    return floor(len(df) / 2) + 1


def precision(y_pred: List[int], y_real: List[int]) -> int:
    tp = sum(map(lambda t: 1 if t[0] == 1 and t[1] == 1 else 0, zip(y_pred, y_real)))
    fp = sum(map(lambda t: 1 if t[0] == 1 and t[1] == 0 else 0, zip(y_pred, y_real)))
    return tp / (tp + fp)


def accuracy(y_pred: List[int], y_real: List[int]) -> int:
    t = sum(map(lambda t: 1 if t[0] == t[1] else 0, zip(y_pred, y_real)))
    return t / len(y_real)


def tp_fp_tn_fn(y_pred: List[int], y_real: List[int]) -> tuple[int, int, int, int]:
    tp = sum(map(lambda t: 1 if t[0] == 1 and t[1] == 1 else 0, zip(y_pred, y_real)))
    fp = sum(map(lambda t: 1 if t[0] == 1 and t[1] == 0 else 0, zip(y_pred, y_real)))
    tn = sum(map(lambda t: 1 if t[0] == 0 and t[1] == 0 else 0, zip(y_pred, y_real)))
    fn = sum(map(lambda t: 1 if t[0] == 0 and t[1] == 1 else 0, zip(y_pred, y_real)))
    return (tp, fp, tn, fn)


def print_metrics(prediction, validation, model, data):
    y_pred = list(prediction[Columns.KRANKHEIT_LUNGE_BRONCHIEN.value])
    y_real = list(validation[Columns.KRANKHEIT_LUNGE_BRONCHIEN.value])
    # print("precision: ", precision(y_pred, y_real))
    # print("accuracy: ", accuracy(y_pred, y_real))
    print("log likelihood score: ", log_likelihood_score(model=model, data=data))
    tp, fp, tn, fn = tp_fp_tn_fn(y_pred, y_real)
    print(f"{tp = }")
    print(f"{fp = }")
    print(f"{tn = }")
    print(f"{fn = }")


def get_lubro(model):
    cpd_lubro = model.get_cpds()[4]
    len_lubro = pd.DataFrame(cpd_lubro.get_values()[0])
    print(cpd_lubro)
    print(len_lubro[len_lubro[0] != 0.5])


def compare_prediction_and_validation(prediction: pd.DataFrame, validation: pd.DataFrame):
    validation.index = prediction.index
    validation["prediction"] = prediction[Columns.KRANKHEIT_LUNGE_BRONCHIEN.value]
    validation["tp"] = validation.apply(lambda x: 1 if x.loc["prediction"] == 1 and x.loc[Columns.KRANKHEIT_LUNGE_BRONCHIEN.value] == 1 else 0, axis=1)
    validation["fp"] = validation.apply(lambda x: 1 if x.loc["prediction"] == 1 and x.loc[Columns.KRANKHEIT_LUNGE_BRONCHIEN.value] == 0 else 0, axis=1)
    validation["tn"] = validation.apply(lambda x: 1 if x.loc["prediction"] == 0 and x.loc[Columns.KRANKHEIT_LUNGE_BRONCHIEN.value] == 0 else 0, axis=1)
    validation["fn"] = validation.apply(lambda x: 1 if x.loc["prediction"] == 0 and x.loc[Columns.KRANKHEIT_LUNGE_BRONCHIEN.value] == 1 else 0, axis=1)
    mask = (validation["kehle"] != 1) & (validation["fn"] != 0)
    # full_print(validation[mask].drop([Columns.KRANKHEIT_LUNGE_BRONCHIEN.value, "prediction"], axis=1))
    print(validation["kehle"].corr(validation[Columns.KRANKHEIT_LUNGE_BRONCHIEN.value]))


def get_used_columns(graph: List[tuple[Columns, Columns]]):
    used_columns = []
    for edge in graph:
        node1, node2 = edge
        used_columns.append(node1)
        used_columns.append(node2)
    return [u.value for u in used_columns]


def get_train_validation(graph: List[tuple[Columns, Columns]]):
    used_columns = get_used_columns(graph)
    train = pd.read_csv("./dataset/prepared/train.csv")
    train = train.drop(columns=[c for c in train.columns if c not in used_columns])
    validation = pd.read_csv("./dataset/prepared/validation.csv")
    validation = validation.drop(columns=[c for c in validation.columns if c not in used_columns])
    return train, validation


def build_model(graph):
    model = BayesianNetwork([(c1.value, c2.value) for c1, c2 in graph])
    train, validation = get_train_validation(graph)
    model.fit(train, estimator=BayesianEstimator)
    return model, train, validation


def print_model_info(model, train, validation):
    prediction = model.predict(validation.drop(Columns.KRANKHEIT_LUNGE_BRONCHIEN.value, axis=1))
    [print(c) for c in model.get_cpds()]
    print_metrics(prediction, validation, model, train)
    # compare_prediction_and_validation(prediction, validation)


def get_original_graph():
    """
    log likelihood score:  -5413.999050408858
    tp = 37
    fp = 17
    tn = 206
    fn = 128
    """
    return [
            (Columns.HAT_HAEUFIG_HUSTEN, Columns.HAT_KEHLKOPFENTZUENDUNG),
            (Columns.HAT_HAEUFIG_SCHNUPFEN, Columns.HAT_KEHLKOPFENTZUENDUNG),
            (Columns.ALLERGIE_ATEMWEGE, Columns.HAT_HAEUFIG_HUSTEN),
            (Columns.ALLERGIE_ATEMWEGE, Columns.HAT_HAEUFIG_SCHNUPFEN),
            (Columns.HAT_KEHLKOPFENTZUENDUNG, Columns.KRANKHEIT_LUNGE_BRONCHIEN),
            (Columns.UMWELTBELASTUNG, Columns.KRANKHEIT_LUNGE_BRONCHIEN),
            (Columns.IST_EIN_ELTERNTEIL_RAUCHER, Columns.KRANKHEIT_LUNGE_BRONCHIEN),
            (Columns.BILDUNGSSTAND_ELTERN, Columns.KRANKHEIT_LUNGE_BRONCHIEN),
            (Columns.BMI, Columns.KRANKHEIT_LUNGE_BRONCHIEN),
            ]


graph = get_original_graph()
model, train, validation = build_model(graph)
print_model_info(model, train, validation)
