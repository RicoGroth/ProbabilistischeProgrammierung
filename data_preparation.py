from __future__ import annotations
from os import path
import pandas as pd
from sklearn.model_selection import train_test_split
from enum import Enum
from requests import Response
import requests_threads
from re import findall as regex_findall
from typing import (List, Union, Callable)

session = requests_threads.AsyncSession(n=20)


class Columns(Enum):
    # Originale Spalten des Datensatz'
    KRANKHEIT_LUNGE_BRONCHIEN = 'lubro'
    UMWELTBELASTUNG = 'zone'
    ALLERGIE_ATEMWEGE = 'aller'
    HAT_KEHLKOPFENTZUENDUNG = 'kehle'
    IST_MUTTER_RAUCHERIN = 'raumu'
    IST_VATER_RAUCHER = 'rauva'
    BILDUNGSSTAND_ELTERN = 'sozio'
    HAT_HAEUFIG_SCHNUPFEN = 'schnu'
    HAT_HAEUFIG_HUSTEN = 'huste'
    GEBURTSMONAT = 'gebmo'
    GEBURTSJAHR = 'gebja'
    GEBURTSTAG = 'gebtg'
    KOERPERGROESSE = 'gross'
    KOERPERGEWICHT = 'gewi'
    GESCHLECHT = 'sex'
    UNTERSUCHUNGSMONAT = 'untmo'
    UNTERSUCHUNGSJAHR = 'untja'
    UNTERSUCHUNGSTAG = 'unttg'
    FVC = 'fvc'                     # Forcierte Vitalkapazitaet
    FEF50 = 'fef50'                 # Ausatemstrom nach 50%-iger Ausatmung
    FEF75 = 'fef75'                 # Ausatemstrom nach 75%-iger Ausatmung
    PEF = 'pef'                     # Maximaler Ausatemstrom
    BEOBACHTUNGSNUMMER = 'nr'

    # Beigefuegte Spalten
    BMI = 'sbmi'
    ALTER = 'alter'
    ALTER_KATEGORISIERT = 'altka'
    IST_EIN_ELTERNTEIL_RAUCHER = 'rauel'
    ANZAHL_RAUCHENDE_ELTERNTEILE = 'rauan'

    @staticmethod
    def all() -> List[Columns]:
        return [c for c in Columns]


class Sex(Enum):
    FEMALE = 1
    MALE = 0

    @staticmethod
    def from_int(i: int) -> Sex:
        if i == 1:
            return Sex.MALE
        elif i == 2:
            return Sex.FEMALE
        else:
            raise Exception("Data cell doesn't have a valid encoding. Encodings are: 1 - male, 2 - female")


class BMI(Enum):
    ANOREXIA = 0
    UNDERWEIGHT = 1
    LIGHT_UNDERWEIGHT = 2
    NORMAL = 3
    LIGHT_OVERWEIGHT = 4
    OVERWEIGHT = 5
    ADIPOSITAS = 6

    @staticmethod
    def from_int(i: int) -> BMI:
        if i < 10:
            return BMI.ANOREXIA
        elif i in range(10, 20):
            return BMI.UNDERWEIGHT
        elif i in range(20, 30):
            return BMI.LIGHT_UNDERWEIGHT
        elif i in range(30, 40):
            return BMI.NORMAL
        elif i in range(40, 50):
            return BMI.LIGHT_OVERWEIGHT
        elif i in range(50, 60):
            return BMI.OVERWEIGHT
        else:
            return BMI.ADIPOSITAS


def split_train_validation_test(
        df: pd.DataFrame,
        target_name: str,
        train_size: float,
        validation_size: float,
        test_size: float,
        seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_size + validation_size + test_size - 1.0) < 0.00001,\
        "train, validation and test size don't add up to 1.0"

    X_trainvalidation, X_test, Y_trainvalidation, Y_test = train_test_split(
            df.drop(columns=[target_name]),
            df[target_name],
            test_size=test_size,
            random_state=seed)

    X_train, X_validation, Y_train, Y_validation = train_test_split(
            X_trainvalidation,
            Y_trainvalidation,
            test_size=validation_size / (train_size + validation_size),
            random_state=seed
            )

    X_train[target_name] = Y_train
    X_validation[target_name] = Y_validation
    X_test[target_name] = Y_test
    return X_train, X_validation, X_test


def with_age(df: pd.DataFrame, categorization: Union[Callable, None] = None) -> pd.DataFrame:
    df[Columns.ALTER.value] = df[Columns.UNTERSUCHUNGSJAHR.value] - (df[Columns.GEBURTSJAHR.value] + 1900)
    if categorization is not None and Columns.ALTER_KATEGORISIERT.value not in df.columns:
        df[Columns.ALTER_KATEGORISIERT.value] = df.apply(categorization, axis=1)
    return df


def with_number_of_smoking_parents(df: pd.DataFrame) -> pd.DataFrame:
    df[Columns.ANZAHL_RAUCHENDE_ELTERNTEILE.value] = df[Columns.IST_MUTTER_RAUCHERIN.value] + df[Columns.IST_VATER_RAUCHER.value]
    return df


def with_at_least_one_smoking_parent(df: pd.DataFrame) -> pd.DataFrame:
    def does_one_parent_smoke(x):
        return 1 if x[Columns.IST_VATER_RAUCHER.value] > 0\
                or x[Columns.IST_MUTTER_RAUCHERIN.value] > 0\
                else 0
    df[Columns.IST_EIN_ELTERNTEIL_RAUCHER.value] = df.apply(does_one_parent_smoke, axis=1)
    return df


async def fetch_bmi_for_all_rows(df: pd.DataFrame, fallback: Union[BMI, None] = None, hard_fail: bool = False) -> pd.DataFrame:
    df = with_age(df)
    urls = [f'https://de.smartbmicalculator.com/ergebnis.html?hc={h}&wk={w}&us={Sex.from_int(s).value}&ua={a}&unit=0'
            for a, w, s, h in
            zip(df[Columns.ALTER.value], df[Columns.KOERPERGEWICHT.value], df[Columns.GESCHLECHT.value], df[Columns.KOERPERGROESSE.value])]
    print('Start fetching BMI', flush=True)
    promises = [session.get(url) for url in urls]
    responses: List[Response] = [await p for p in promises]
    print('Stop fetching BMI', flush=True)
    bmi_list: List[BMI] = []
    for index, r in enumerate(responses):
        matches = regex_findall("SBMI = .{0,2}/.{0,2}", r.content.decode("utf-8"))
        if len(matches) == 0:
            message = f'No BMI was calculated for entry {index}.'
            if hard_fail:
                raise Exception(message)
            else:
                print(message, flush=True)
                bmi_list.append(fallback)
        else:
            bmi_list.append(BMI.from_int(int(matches[0].split()[2].split("/")[0])))
    df[Columns.BMI.value] = [bmi.value for bmi in bmi_list]
    return df


async def prepare_dataframe(df: pd.DataFrame,
                            allow_fetching_outside_data: bool,
                            columns_to_keep: List[Columns],
                            ) -> pd.DataFrame:
    df = with_age(df)
    df = with_number_of_smoking_parents(df)
    df = with_at_least_one_smoking_parent(df)
    if Columns.BMI in columns_to_keep\
            and Columns.BMI.value not in df.columns\
            and allow_fetching_outside_data:
        df = await fetch_bmi_for_all_rows(df, fallback=BMI.NORMAL, hard_fail=False)
    all_columns = Columns.all()
    if any([c not in all_columns for c in columns_to_keep]):
        columns = [c.value for c in columns_to_keep]
        invalid_columns = [c for c in columns if c not in df.columns]
        [print(f'Skipping {c} since it is not a column in the dataframe') for c in invalid_columns]
        valid_columns = [c for c in columns if c in df.columns]
        df = df.drop(columns=valid_columns)
    return df


async def write_prepared_datasets(train_name: str, validation_name: str, test_name: str, write_path: str):
    original_dataset_path = path.join(
        path.abspath(path.curdir),
        'dataset',
        'immutable',
        'atemwege.asc'
        )
    original_dataset = pd.read_csv(original_dataset_path, delim_whitespace=True)
    prepared = await prepare_dataframe(original_dataset, allow_fetching_outside_data=True, columns_to_keep=Columns.all())
    train, validation, test = split_train_validation_test(
            prepared,
            target_name=Columns.KRANKHEIT_LUNGE_BRONCHIEN.value,
            train_size=.6,
            validation_size=.25,
            test_size=.15,
            seed=42
            )
    train_path = path.join(write_path, train_name)
    train.to_csv(path_or_buf=train_path)
    print(f'Training data written to {train_path}')

    validation_path = path.join(write_path, validation_name)
    validation.to_csv(path_or_buf=validation_path)
    print(f'Validation data written to {validation_path}')

    test_path = path.join(write_path, test_name)
    test.to_csv(path_or_buf=test_path)
    print(f'Test data written to {test_path}')


async def main():
    train_name = 'train.csv'
    validation_name = 'validation.csv'
    test_name = 'test.csv'
    prepared_data_path = path.join(path.abspath(path.curdir), 'dataset', 'prepared')
    all_files_exist = all([path.exists(path.join(prepared_data_path, f))
                           for f in [train_name, validation_name, test_name]
                           ])
    if not all_files_exist:
        await write_prepared_datasets(train_name, validation_name, test_name, prepared_data_path)


if __name__ == "__main__":
    session.run(main)
