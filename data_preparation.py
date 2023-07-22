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
    LUNGENVOLUMEN = 'lunvo'
    BEOBACHTUNGSNUMMER = 'nr'

    # Beigefuegte Spalten
    BMI = 'bmi'
    BMI_KATEGORISIERT_AUSREISSER = 'bmic1'
    BMI_KATEGORISIERT_REDUKTION = 'bmic2'
    BMI_BERECHNET = 'bmi_b'
    ALTER = 'alter'
    ALTER_KATEGORISIERT = 'altka'
    IST_EIN_ELTERNTEIL_RAUCHER = 'rauel'
    ANZAHL_RAUCHENDE_ELTERNTEILE = 'rauan'
    FVC_KATEGORISIERT = 'fvckat'
    FEF50_KATEGORISIERT = 'fef50kat'
    FEF75_KATEGORISIERT= 'fef75kat'
    PEF_KATEGORISIERT = 'pefkat'

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


async def fetch_bmi_for_all_rows(df_path: str, sep: str, fallback: Union[BMI, None] = None, hard_fail: bool = False) -> pd.DataFrame:
    df = pd.read_csv(df_path, sep=sep)
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


def with_calculated_bmi(df: pd.DataFrame) -> pd.DataFrame:
    def t(x):
        gew = x.loc[Columns.KOERPERGEWICHT.value]
        gro = x.loc[Columns.KOERPERGROESSE.value] / 100
        return gew / (gro ** 2)
    df[Columns.BMI_BERECHNET.value] = df.apply(t, axis=1)
    return df


def with_bmi_categorized(df: pd.DataFrame, categorization, column: Columns) -> pd.DataFrame:
    df[column.value] = df.apply(lambda x: categorization(x[Columns.BMI.value]), axis=1)
    return df


def with_ternary_categorization(df: pd.DataFrame, column: Columns, new_column: Columns) -> pd.DataFrame:
    from math import floor
    max = df[column.value].max()
    first_boundary = floor(max / 3) + 1
    second_boundary = floor(max / 3 * 2) + 1

    def t(x):
        if x < first_boundary:
            return 0
        elif x >= first_boundary and x < second_boundary:
            return 1
        else:
            return 2
    df[new_column.value] = df.apply(lambda x: t(x[column.value]), axis=1)
    return df


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = with_age(df)
    df = with_number_of_smoking_parents(df)
    df = with_at_least_one_smoking_parent(df)
    df = with_calculated_bmi(df)
    df = with_bmi_categorized(df, lambda x: 1 if x > 3 or x < 3 else 0, Columns.BMI_KATEGORISIERT_AUSREISSER)
    df = with_ternary_categorization(df, Columns.PEF, Columns.PEF_KATEGORISIERT)
    df = with_ternary_categorization(df, Columns.FEF50, Columns.FEF50_KATEGORISIERT)
    df = with_ternary_categorization(df, Columns.FEF75, Columns.FEF75_KATEGORISIERT)
    df = with_ternary_categorization(df, Columns.FVC, Columns.FVC_KATEGORISIERT)

    def t(x):
        if x < 3:
            return 0
        elif x == 3:
            return 1
        else:
            return 2
    df = with_bmi_categorized(df, lambda x: 1 if x > 3 or x < 3 else 0, Columns.BMI_KATEGORISIERT_AUSREISSER)
    df = with_bmi_categorized(df, t, Columns.BMI_KATEGORISIERT_REDUKTION)
    return df


def write_split_datasets(
        complete_data_set: pd.DataFrame,
        train_name: str,
        validation_name: str,
        test_name: str,
        write_path: str):
    train, validation, test = split_train_validation_test(
            complete_data_set,
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


async def get_original_dataset():
    immutable_path = path.join(path.abspath(path.curdir), 'dataset', 'immutable')
    original_dataset_path = path.join(immutable_path, 'atemwege.asc')
    prepared_original_dataset_path = path.join(immutable_path, 'atemwege-prepared.csv')
    if not path.exists(prepared_original_dataset_path):
        print(f'{prepared_original_dataset_path} doesn\'t exist.')
        df = await fetch_bmi_for_all_rows(original_dataset_path, sep=' ', fallback=BMI.NORMAL, hard_fail=False)
        df.to_csv(path_or_buf=prepared_original_dataset_path)
    return pd.read_csv(prepared_original_dataset_path)


async def main():
    prepared_data_path = path.join(path.abspath(path.curdir), 'dataset', 'prepared')
    x = await get_original_dataset()
    prepared = prepare_dataframe(x)
    write_split_datasets(
            prepared,
            path.join(prepared_data_path, 'train.csv'),
            path.join(prepared_data_path, 'validation.csv'),
            path.join(prepared_data_path, 'test.csv'),
            prepared_data_path
            )


if __name__ == "__main__":
    session.run(main)
