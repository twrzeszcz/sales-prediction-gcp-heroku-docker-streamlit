import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class custom_imputer(BaseEstimator, TransformerMixin):
    def __init_(self):
        self.mode_to_replace = None
        self.mean_to_replace = None

    def fit(self, X, y=None):
        X_new = X.copy()
        self.mode_to_replace = X_new['Outlet_Size'].mode()[0]
        self.mean_to_replace = X_new['Item_Weight'].mean()

        return self

    def transform(self, X, y=None):
        X_new = X.copy()
        X_new['Outlet_Size'].fillna(self.mode_to_replace, inplace=True)
        X_new['Item_Weight'].fillna(self.mean_to_replace, inplace=True)

        return X_new

def replace_add_drop_cat(X):
    X_new = X.copy()
    X_new['Item_Fat_Content'].replace({'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}, inplace=True)

    X_new['New_Item_Type'] = X_new['Item_Identifier'].apply(lambda x: x[:2])
    X_new['New_Item_Type'].replace({'FD': 'Food', 'NC': 'Non-Consumable', 'DR': 'Drinks'}, inplace=True)
    X_new.loc[X_new['New_Item_Type'] == 'Non-Consumable', 'Item_Fat_Content'] = 'Non-Edible'

    X_new.drop('Item_Type', axis=1, inplace=True)

    return X_new

class replace_add_drop_num(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.median_to_replace = None
        self.max_year = None
        self.item_visib_avg = None
        self.avg = None
        self.unique_name = None
        self.mapper = None

    def fit(self, X, y=None):
        # only on the training set
        X_new = X.copy()

        # replace 0 with nan and then median
        X_new.loc[X_new['Item_Visibility'] == 0, 'Item_Visibility'] = np.nan
        self.median_to_replace = X_new['Item_Visibility'].median()
        X_new['Item_Visibility'].fillna(self.median_to_replace, inplace=True)

        # get the max year
        self.max_year = X_new['Outlet_Establishment_Year'].max()

        # get the list of average item visibilities for each identifier and
        # create dictionary to map this value to the corresponding identifier
        self.avg = [X_new[X_new['Item_Identifier'] == item]['Item_Visibility'].mean() for item in
                    X_new['Item_Identifier'].unique()]
        self.unique_name = X_new['Item_Identifier'].unique()
        self.mapper = dict(zip(self.unique_name, self.avg))

        return self

    def transform(self, X, y=None):
        X_new = X.copy()

        X_new['Item_Visibility'].replace({0: self.median_to_replace}, inplace=True)

        X_new['Outlet_Establishment_Year'] = self.max_year - X_new['Outlet_Establishment_Year']

        X_new['item_visible_avg'] = X_new['Item_Visibility']
        for name, item in self.mapper.items():
            X_new.loc[X_new['Item_Identifier'] == name, 'item_visible_avg'] /= item

        X_new.drop('Item_Identifier', axis=1, inplace=True)

        return X_new