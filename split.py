import pandas as pd
from sklearn.model_selection import train_test_split
import pgmpy as pg

TRAIN_PORTION = .6
ORIGINAL_VALIDATION_PORTION = .25
TEST_PORTION = .15
VALIDATION_PORTION = 1/(TRAIN_PORTION + ORIGINAL_VALIDATION_PORTION) * ORIGINAL_VALIDATION_PORTION

edges = [
        ('gebja', 
        ]

dataset = pd.read_csv("./dataset/atemwege.asc", delimiter=' ')
Y = dataset['lubro']
X = dataset.drop('lubro', axis=1)
X_trainvalidation, _, Y_trainvalidation, _ = train_test_split(X, Y, test_size=TEST_PORTION, random_state=42)
X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_trainvalidation,
        Y_trainvalidation,
        test_size=VALIDATION_PORTION,
        random_state=42
        )

