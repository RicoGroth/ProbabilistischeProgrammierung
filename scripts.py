from __future__ import annotations
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from requests import Response
import requests_threads
from re import findall as regex_findall
from enum import Enum
import math


DATASET_PATH = os.path.join(os.path.curdir, "dataset")
ORIGINAL_DATASET_PATH = os.path.join(DATASET_PATH, "immutable", "atemwege.asc")
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, "train.csv")
VALIDATION_DATASET_PATH = os.path.join(DATASET_PATH, "validation.csv")
TEST_DATASET_PATH = os.path.join(DATASET_PATH, "test.csv")


session = requests_threads.AsyncSession(n=20)


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


class BMI_CATEGORY(Enum):
    ANOREXIA = 0
    UNDERWEIGHT = 1
    LIGHT_UNDERWEIGHT = 2
    NORMAL = 3
    LIGHT_OVERWEIGHT = 4
    OVERWEIGHT = 5
    ADIPOSITAS = 6

    @staticmethod
    def from_int(i: int) -> BMI_CATEGORY:
        if i < 10:
            return BMI_CATEGORY.ANOREXIA
        elif i in range(10, 20):
            return BMI_CATEGORY.UNDERWEIGHT
        elif i in range(20, 30):
            return BMI_CATEGORY.LIGHT_UNDERWEIGHT
        elif i in range(30, 40):
            return BMI_CATEGORY.NORMAL
        elif i in range(40, 50):
            return BMI_CATEGORY.LIGHT_OVERWEIGHT
        elif i in range(50, 60):
            return BMI_CATEGORY.OVERWEIGHT
        else:
            return BMI_CATEGORY.ADIPOSITAS


# work in progress
def pef_prediction(age, heightInCentimeters, sex: Sex):
    male_prediction = math.exp(0.367 * math.log(age) - 0.012 * age - 58.8 / heightInCentimeters + 5.63)
    female_prediction = math.exp(0.544)  # TODO
    return male_prediction if sex == sex.MALE else female_prediction


class DataFrameManipulator:
    """
    A class to manipulate the atemwege data in a pandas dataframe.
    Operations can be done in any order, only if you call
    get_dataframe() or write_to_disk() do columns get dropped or finally
    changed
    """

    def __init__(self, df: pd.DataFrame, hard_fail=True, verbosity_level=0):
        """
        Initial configuration of the Manipulator.

        df                  the dataframe to manipulate
        hard_fail           if set to True, all operations will crash the
                            program if something unexpected happens like
                            a failed fetch
        verbosity_level     a value between 0 and 2, 0 being the lowest
        """
        self.df = df
        self.columns = df.columns
        self.hard_fail = hard_fail
        self.verbosity_level = verbosity_level
        self.__columns_to_drop = []

    def __column_name_guard(self, column_name: str):
        message_on_fail = f"{column_name} is not available in the dataframe."
        if self.hard_fail:
            assert column_name in self.df, message_on_fail
        elif column_name not in self.df:
            print(message_on_fail)

    def __vprint(self, thing, verbosity_level_threshold):
        if self.verbosity_level >= verbosity_level_threshold:
            print(thing)

    def without(self, columns: list):
        """
        Drops the specified columns.

        columns             columns to drop

        The columns are buffered and finally dropped later, so this
        does not have an immediate effect.
        """
        for c in columns:
            self.__vprint(f"Dropping column \"{c}\"", 1)
        self.__columns_to_drop += filter(lambda x: x is not None,
                                        [c if c in self.df
                                         else self.__column_name_guard(c)
                                         for c in columns])
        return self

    async def with_bmi(self, fallback: BMI_CATEGORY = BMI_CATEGORY.NORMAL):
        """
        Fetches the SBMI category from an "API" (wink).

        fallback                the category to assign if no value was
                                fetched

        Basically, it sends a request to de.smartbmicalculator.com, parses
        the HTML response and looks for a SBMI value, which can then be
        categorized.
        """
        async def get_bmi_column(df: pd.DataFrame):
            def from_response_to_bmi_category(response: Response) \
                    -> BMI_CATEGORY:
                html = response.content.decode("utf-8")
                matches = regex_findall("SBMI = .{0,2}/.{0,2}", html)
                if len(matches) == 0:
                    raise Exception("No SBMI was calculated.")
                sbmi_parsed = int(matches[0].split()[2].split("/")[0])
                return BMI_CATEGORY.from_int(sbmi_parsed)

            async def resolve_promise(promise):
                self.__vprint(f"Resolving promise {promise}", 2)
                return await promise

            def send_request(url: str):
                self.__vprint(f"Sending request: {url}", 2)
                return session.get(url)

            def get_bmi_category(response: Response, fallback=fallback):
                if self.hard_fail:
                    return from_response_to_bmi_category(response)
                else:
                    try:
                        return from_response_to_bmi_category(response)
                    except Exception as exception:
                        self.__vprint(exception, 1)
                        return BMI_CATEGORY.NORMAL

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

            age_column = df["untja"] - (df["gebja"] + 1900)
            urls = [create_request_url(age, weight, Sex.from_int(sex), height)
                    for age, weight, sex, height in
                    zip(age_column, df["gewi"], df["sex"], df["gross"])]
            self.__vprint("Start fetching bmi data", 1)
            promises = [send_request(url) for url in urls]
            responses = [await resolve_promise(p) for p in promises]
            self.__vprint("Fetched bmi data", 1)
            return [get_bmi_category(response) for response in responses]

        for column in ["gewi", "gross", "sex", "gebja", "untja"]:
            self.__column_name_guard(column)
        self.df["bmi"] = [sbmi.value for sbmi in await get_bmi_column(self.df)]
        return self

    def get_dataframe(self) -> pd.DataFrame:
        return self.df.drop(columns=self.__columns_to_drop, axis=1)

    def write_to_disk(self, path=os.path.join(DATASET_PATH, "train-prepared.csv")):
        """
        Writes the manipulated dataframe to the specified or default path.

        path                path of the file to write
        """
        self.get_dataframe().to_csv(path_or_buf=path, sep=" ")
        self.__vprint(f"Manipulated dataframe was written to {path}", 1)


def read_complete_dataset(path: str = ORIGINAL_DATASET_PATH) \
        -> pd.DataFrame:
    """
    Reads the complete provided dataset as is into a pandas dataframe.

    The resulting dataframe represents the starting point for the model.
    Thus, no splitting into test, train and validation set or any other
    preparation is done.
    """
    return pd.read_csv(path, delim_whitespace=True)


def split_train_validation_test(
        df: pd.DataFrame,
        target_name: str = "lubro",
        train_portion: float = .6,
        validation_portion: float = .25,
        test_portion: float = .15,
        train_set_write_path: str = TRAIN_DATASET_PATH,
        validation_set_write_path: str = VALIDATION_DATASET_PATH,
        test_set_write_path: str = TEST_DATASET_PATH,
        seed: int = 42):
    """
    Prepares the dataset by splitting it into train, validation and test set,
    and writing the test set to disk as .csv-file

    df                      dataframe to prepare
    target_name             name of the column that represents the target
                            vector
    train_portion           number in [0, 1] that determines the size of the
                            train set
    validation_portion      number in [0, 1] that determines the size of the
                            validation set
    test_portion            number in [0, 1] that determines the size of the
                            test set
    test_set_write_path     path where to write the resulting test data set to
    seed                    number that determines the split, needs to be the
                            same for the same training since a different seed
                            results in a different test set

    Return:
    X_train, Y_train, X_validation, Y_validation
    """
    portions = train_portion + validation_portion + test_portion
    assert abs(portions - 1.0) < 0.00001, "train, validation and test portion don't add up to 1.0"
    validation_portion = 1/(train_portion + validation_portion) * validation_portion
    Y = df[target_name]
    X = df.drop(target_name, axis=1)

    X_trainvalidation, X_test, Y_trainvalidation, Y_test = train_test_split(
            X,
            Y,
            test_size=test_portion,
            random_state=seed)

    X_train, X_validation, Y_train, Y_validation = train_test_split(
            X_trainvalidation,
            Y_trainvalidation,
            test_size=validation_portion,
            random_state=seed
            )

    X_train[target_name] = Y_train
    X_validation[target_name] = Y_validation
    X_test[target_name] = Y_test
    separator = " "

    X_train.to_csv(path_or_buf=train_set_write_path, sep=separator)
    print(f"train dataset written to {train_set_write_path}")

    X_validation.to_csv(path_or_buf=validation_set_write_path, sep=separator)
    print(f"validation dataset written to {validation_set_write_path}")

    X_test.to_csv(path_or_buf=test_set_write_path, sep=separator)
    print(f"test dataset written to {test_set_write_path}")

    X_train = X_train.drop(columns=[target_name])
    X_validation = X_validation.drop(columns=[target_name])
    return X_train, Y_train, X_validation, Y_validation


def just_split_dataset():
    """
    Basically a shortcut function to split the dataset without any
    prior configuration. Gets the job done.
    """
    split_train_validation_test(read_complete_dataset())


async def _main():
    if not os.path.exists(TRAIN_DATASET_PATH):
        just_split_dataset()
    dataset = pd.read_csv(TRAIN_DATASET_PATH, delim_whitespace=True)
    manipulator = DataFrameManipulator(dataset, verbosity_level=1, hard_fail=False)
    manipulator.without(["nr"])
    manipulator.without(["fef50", "fef75", "pef", "fvc"])
    manipulator.without(["gebja", "gebtg", "gebmo", "untja", "unttg", "untmo"])
    await manipulator.with_bmi()
    manipulator.write_to_disk()
    print(manipulator.get_dataframe())


def main():
    session.run(_main)


if __name__ == "__main__":
    main()
