# https://de.smartbmicalculator.com/ergebnis.html?unit=0&hc=180&wk=94&us=0&ua=60&dp=1
# ua -> age
# hc -> height (cm)
# us -> sex (1 female, 0 male)
# wk -> weight (kg)
import pandas as pd
from requests import Response
import requests_threads
from re import findall as regex_findall
from enum import Enum
import math


session = requests_threads.AsyncSession(n=10)


def full_print(df: pd.DataFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


class Sex(Enum):
    FEMALE = 1
    MALE = 0


def from_int_to_sex(sex_cell: int) -> Sex:
    if sex_cell == 1:
        return Sex.MALE
    elif sex_cell == 2:
        return Sex.FEMALE
    else:
        raise Exception("Data cell doesn't have a valid encoding. Encodings are: 1 - male, 2 - female")


class BMI_CATEGORY(Enum):
    ANOREXIA = 0
    UNDERWEIGHT = 1
    LIGHT_UNDERWEIGHT = 2
    NORMAL = 3
    LIGHT_OVERWEIGHT = 4
    OVERWEIGHT = 5
    ADIPOSITAS = 6


def create_request_url(age: int, weightInKilograms: int, sex: Sex, heightInCentimeters: int) -> str:
    schema = "https"
    host = "de.smartbmicalculator.com"
    path = "ergebnis.html"
    params = {
            "unit": 0,  # 0 == kilogram
            "hc": heightInCentimeters,
            "wk": weightInKilograms,
            "us": sex.value,
            "ua": age,
            }
    joined_params = "&".join([f"{x[0]}={x[1]}" for x in params.items()])
    return f"{schema}://{host}/{path}?{joined_params}"


def categorize_sbmi(sbmi: int) -> BMI_CATEGORY:
    if sbmi < 10:
        return BMI_CATEGORY.ANOREXIA
    elif sbmi in range(10, 20):
        return BMI_CATEGORY.UNDERWEIGHT
    elif sbmi in range(20, 30):
        return BMI_CATEGORY.LIGHT_UNDERWEIGHT
    elif sbmi in range(30, 40):
        return BMI_CATEGORY.NORMAL
    elif sbmi in range(40, 50):
        return BMI_CATEGORY.LIGHT_OVERWEIGHT
    elif sbmi in range(50, 60):
        return BMI_CATEGORY.OVERWEIGHT
    else:
        return BMI_CATEGORY.ADIPOSITAS


def from_response_to_bmi_category(response: Response) -> BMI_CATEGORY:
    html = response.content.decode("utf-8")
    matches = regex_findall("SBMI = .{0,2}/.{0,2}", html)
    if len(matches) == 0:
        raise Exception("No SBMI was calculated.")
    sbmi_parsed = int(matches[0].split()[2].split("/")[0])
    return categorize_sbmi(sbmi_parsed)


# work in progress
def log_pef_prediction(age, heightInCentimeters, sex: Sex):
    male_prediction = 0.367 * math.log(age) - 0.012 * age - 58.8 / heightInCentimeters + 5.63
    female_prediction = 0.544
    return male_prediction if sex == sex.MALE else female_prediction


def send_request(url: str, verbose=False):
    if verbose:
        print(f"Sending request: {url}")
    return session.get(url)


async def resolve_promise(promise, verbose=False):
    if verbose:
        print(f"Resolving promise {promise}")
    return await promise


def get_bmi_category(response: Response):
    try:
        return from_response_to_bmi_category(response)
    except Exception as exception:
        print(exception)
        return BMI_CATEGORY.NORMAL


async def get_bmi_column(df: pd.DataFrame):
    urls = [create_request_url(age, weightInKilograms, from_int_to_sex(sex), heightInCentimeters)
            for age, weightInKilograms, sex, heightInCentimeters in
            zip(df["untja"] - (df["gebja"] + 1900), df["gewi"], df["sex"], df["gross"])]
    promises = [send_request(url, verbose=True) for url in urls]
    responses = [await resolve_promise(promise, verbose=True) for promise in promises]
    return [get_bmi_category(response) for response in responses]


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["nr",
                     "gebmo", "gebtg", "gebja", "untmo", "unttg", "untja",
                     "pef", "fvc", "fef50", "fef75",
                     "gross", "gewi",
                     ])


async def main():
    data = pd.read_csv("dataset/atemwege.asc", delim_whitespace=True)
    data["sbmi"] = [sbmi.value for sbmi in await get_bmi_column(data)]
    data = drop_unnecessary_columns(data)
    data.to_csv(path_or_buf="./dataset/atemwege-prepared.csv", sep=" ")

if __name__ == "__main__":
    session.run(main)
