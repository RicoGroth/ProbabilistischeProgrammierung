from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.metrics import log_likelihood_score
import pandas as pd
from typing import List
from data_preparation import Columns

Edge = tuple[Columns, Columns]
EdgeData = tuple[str, str]
Graph = List[Edge]
GraphData = List[EdgeData]


def get_train_validation(graph: GraphData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    def drop_unused_columns(df: pd.DataFrame, used_columns: List[Columns]) -> pd.DataFrame:
        return df.drop(columns=[c for c in df.columns if c not in used_columns])

    used_columns = []
    for edge in graph:
        node1, node2 = edge
        used_columns.append(node1)
        used_columns.append(node2)

    train = pd.read_csv("./dataset/prepared/train.csv")
    train_prepared = drop_unused_columns(train, used_columns)
    validation = pd.read_csv("./dataset/prepared/validation.csv")
    validation = drop_unused_columns(validation, used_columns)
    return train, train_prepared, validation


def build_graph(*edges: Edge) -> GraphData:
    return [(e1.value, e2.value) for e1, e2 in edges]


if __name__ == "__main__":
    graph = build_graph(
                (Columns.HAT_HAEUFIG_HUSTEN, Columns.KRANKHEIT_LUNGE_BRONCHIEN),
                )
    train, train_prepared, validation = get_train_validation(graph)
    model = BayesianNetwork(graph)
    model.fit(train, estimator=MaximumLikelihoodEstimator)
    log_likelihood = log_likelihood_score(model=model, data=train_prepared)
    print(f'log likelihood: {log_likelihood}')
