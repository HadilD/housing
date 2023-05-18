import os
import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
import hashlib
from typing import Callable

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url: str = HOUSING_URL,
                       housing_path: str = HOUSING_PATH) -> None:
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)

    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path: str = HOUSING_PATH) -> pd.DataFrame:
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data: pd.DataFrame, test_ratio: float) -> (
pd.DataFrame, pd.DataFrame):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier: int, test_ratio: float, hash: Callable):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data: pd.DataFrame, test_ratio: float,
                           id_column: str, hash: Callable = hashlib.md5) -> (
                           pd.DataFrame, pd.DataFrame):
    ids = data[id_column]
    in_test_set_mask = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set_mask], data.loc[in_test_set_mask]
