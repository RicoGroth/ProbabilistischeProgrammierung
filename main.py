from pgmpy.models import BayesianNetwork
from pgmpy.estimators import (MaximumLikelihoodEstimator, ParameterEstimator, BayesianEstimator)
from pgmpy.sampling import BayesianModelSampling
from pgmpy.metrics import log_likelihood_score
from random import random
import pandas as pd
from math import floor
from typing import List


def full_print(df: pd.DataFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.precision', 3,):
        print(df)


model = BayesianNetwork(
    [
        # ("aller", "huste"),
        # ("aller", "schnu"),
        # ("huste", "kehle"),
        # ("schnu", "kehle"),
        # ("kehle", "huste"),
        # ("kehle", "schnu"),
        ("huste", "lubro"),
        # ("kehle", "lubro"),
        # ("zone", "lubro"),
        # ("alter", "lubro"),
        # ("alter_cat", "lubro"),
        # ("rauel", "lubro"),
        # ("rauel_bool", "lubro"),
        # ("rauva", "lubro"),
        # ("raumu", "lubro"),
        # ("sozio", "lubro"),
        # ("bmi", "lubro"),
        # ("bmiex", "lubro"),
    ]
)


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
    y_pred = list(prediction["lubro"])
    y_real = list(validation["lubro"])
    print("precision: ", precision(y_pred, y_real))
    print("accuracy: ", accuracy(y_pred, y_real))
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
    validation["prediction"] = prediction["lubro"]
    validation["tp"] = validation.apply(lambda x: 1 if x.loc["prediction"] == 1 and x.loc["lubro"] == 1 else 0, axis=1)
    validation["fp"] = validation.apply(lambda x: 1 if x.loc["prediction"] == 1 and x.loc["lubro"] == 0 else 0, axis=1)
    validation["tn"] = validation.apply(lambda x: 1 if x.loc["prediction"] == 0 and x.loc["lubro"] == 0 else 0, axis=1)
    validation["fn"] = validation.apply(lambda x: 1 if x.loc["prediction"] == 0 and x.loc["lubro"] == 1 else 0, axis=1)
    mask = (validation["kehle"] != 1) & (validation["fn"] != 0)
    # full_print(validation[mask].drop(["lubro", "prediction"], axis=1))
    print(validation["kehle"].corr(validation["lubro"]))


df = pd.read_csv("./dataset/train-prepared.csv", delim_whitespace=True)
validation = pd.read_csv("./dataset/validation-prepared.csv", delim_whitespace=True)
model.fit(df, estimator=BayesianEstimator)
[print(c) for c in model.get_cpds()]
prediction = model.predict(validation.drop("lubro", axis=1))
print_metrics(prediction, validation, model, df)
# compare_prediction_and_validation(prediction, validation)
