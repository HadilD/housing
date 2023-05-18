import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from src.util import load_housing_data

housing = load_housing_data()

# info: median income is important for predicting median housing prices therefore it's
# important to ensure that the test set is representative of all income categories


# create a discrete income category since median income is continuous
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


# stratified sampling based on the income category 
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# remove income category 
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)


# create a clean train set & separate predictors from labels
train_set = strat_train_set.drop("median_house_value", axis=1)
train_set_labels = strat_train_set["median_house_value"].copy()



