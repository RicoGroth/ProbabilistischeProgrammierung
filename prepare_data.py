# https://de.smartbmicalculator.com/ergebnis.html?unit=0&hc=180&wk=94&us=0&ua=60&dp=1
# ua -> age
# hc -> height (cm)
# us -> sex (1 female, 0 male)
# wk -> weight (kg)
import pandas as pd
from requests_threads import AsyncSession
from requests import Response
import re
from enum import Enum
import math

session = AsyncSession(n=50)


def full_print(df: pd.DataFrame):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)


class Sex(Enum):
    FEMALE = 1,
    MALE = 0


def from_int_to_sex(sex_cell: int) -> Sex:
    if sex_cell == 1:
        return Sex.MALE
    elif sex_cell == 2:
        return Sex.FEMALE
    else:
        raise Exception("Data cell doesn't have a valid encoding. Encodings are: 1 - male, 2 - female")


class BMI_CATEGORY(Enum):
    ANOREXIA = 0,
    UNDERWEIGHT = 1,
    LIGHT_UNDERWEIGHT = 2,
    NORMAL = 3,
    LIGHT_OVERWEIGHT = 4,
    OVERWEIGHT = 5,
    ADIPOSITAS = 6


def create_request_url(age: int, weightInKilograms: int, sex: Sex, heightInCentimeters: int) -> str:
    schema = "https"
    host = "de.smartbmicalculator.com"
    path = "ergebnis.html"
    params = {
            "unit": 0,  # kilogram
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
    html = (await response).content.decode("utf-8")
    matches = re.findall("SBMI = .{0,2}/.{0,2}", html)
    if len(matches) == 0:
        raise Exception("No SBMI was calculated.")
    sbmi_parsed = int(matches[0].split()[2].split("/")[0])
    return categorize_sbmi(sbmi_parsed)


def sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = df.columns.tolist()
    return df[columns[0:(len(columns) - 2)]
              + [columns[len(columns) - 1]]
              + [columns[len(columns) - 2]]]


def log_pef_prediction(age, heightInCentimeters, sex: Sex):
    male_prediction = 0.367 * math.log(age) - 0.012 * age - 58.8 / heightInCentimeters + 5.63
    female_prediction = 0.544
    return male_prediction if sex == sex.MALE else female_prediction


def main():
    data = pd.read_csv("dataset/atemwege.asc", delim_whitespace=True)
    data["gebja"] = data["gebja"] + 1900
    data["alter"] = data["untja"] - data["gebja"]
    data = data.drop(columns=["gebmo", "gebtg", "gebja", "untmo", "unttg", "untja"])
    data = data.drop(columns=["pef", "fvc", "fef50", "fef75"])
    data = data.drop(columns=["gross", "gewi"])
    print(data.columns)


if __name__ == "__main__":
    main()
