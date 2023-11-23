import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import uvicorn
from scipy.stats import mstats
from typing import Literal
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin
from category_encoders import OrdinalEncoder
from datetime import datetime
from pycaret.internal.preprocess.transformers import TransformerWrapper, ExtractDateTimeFeatures
from fastapi import FastAPI
from pydantic import BaseModel

# Load All Data Cleaned
data = pd.read_csv('D:\\david\\OneDrive\\Documents\\Future\\Project\\Application\\Education\\Purwadhika\\Capstone\\ApartmentData\\Notebook\\data_daegu_apartment_preparation_cleaned.csv')

# Define the features and target variable
features = data.drop('SalePrice', axis=1)
target = data['SalePrice']

# Define the features for each encoding type
min_max_features = ['N_Parkinglot(Basement)', 'N_FacilitiesInApt', 'Size(sqf)']
categorical_features_onehot = ['SubwayStation']
ordinal_features = ['TimeToSubway', 'HallwayType', 'N_FacilitiesNearBy(PublicOffice)', 'N_FacilitiesNearBy(ETC)', 'N_SchoolNearBy(University)']
outlier_features = ['Size(sqf)']
facilities_features = ['N_FacilitiesNearBy(PublicOffice)', 'N_FacilitiesNearBy(ETC)', 'N_SchoolNearBy(University)']
columns_to_drop = ['remainder__Size(sqf)_bins', 'remainder__N_FacilitiesInApt_bins', 
                   'remainder__N_FacilitiesNearBy(PublicOffice)_bins', 
                   'remainder__N_SchoolNearBy(University)_bins']

# import winsorize
from scipy.stats.mstats import winsorize

class HandleOutlier(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        X['Size(sqf)'] = winsorize(X['Size(sqf)'], limits=(0.01, 0.01))
        return X

    def set_output(self, transform: Literal['default', 'pandas']):
        return super().set_output(transform=transform)

class AgeBinner(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        X['Binned_AgeProperty'] = pd.cut(X['AgeProperty'], bins=[0, 20, 35, 50], labels=[3, 2, 1])
        return X

    def set_output(self, transform: Literal['default', 'pandas']):
        return super().set_output(transform=transform)
    
class AgeTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        X['AgeProperty'] = 2016 - X['YearBuilt']
        return X

    def set_output(self, transform: Literal['default', 'pandas']):
        return super().set_output(transform=transform)

from sklearn.base import BaseEstimator, TransformerMixin

class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(self.columns_to_drop, axis=1)
    
# Create transformers for each encoding type
min_max_transformer = MinMaxScaler(feature_range=(0, 1))

# Create OneHotEncoder for categorical features
categorical_transformer_onehot = OneHotEncoder(sparse=False)

mappings = [
    {'col': 'TimeToSubway',
    'mapping': {
        'no_bus_stop_nearby': 0, 
        '0-5min': 4, 
        '5min~10min': 2, 
        '10min~15min': 2,
        '15min~20min': 1,
        }
    },
    {'col': 'HallwayType',
    'mapping': {
        'terraced': 3, 
        'mixed': 2, 
        'corridor': 1, 
        }
    },
    {'col': 'N_FacilitiesNearBy(PublicOffice)',
    'mapping': {
        0 : 4, 
        1 : 3, 
        2 : 2,
        5 : 1, 
        }
    },
    {'col': 'N_SchoolNearBy(University)',
    'mapping': {
        0 : 6, 
        1 : 5, 
        2 : 4,
        3 : 3,
        4 : 2,
        5 : 1, 
        }
    },
    {'col': 'N_FacilitiesNearBy(ETC)',
    'mapping': {
        0 : 4, 
        1 : 3, 
        2 : 2,
        5 : 1, 
        }
    },
]

ordinal_transformer = OrdinalEncoder(mapping=mappings)

# Create a column transformer to apply different transformers to different columns
preprocessor = [
    # Handle Outlier
    ('Outlier', TransformerWrapper(
        include=outlier_features,
        transformer=HandleOutlier())),

    # Create New Features Age
    ('Age', TransformerWrapper(
        include=['YearBuilt'],
        transformer=AgeTransformer())),
    
    # Binning Features Size(sqf) and SalePrice
    ('Binnning', TransformerWrapper(
        include=['AgeProperty'],
        transformer=AgeBinner())),
    
    # Transform Some Data
    ('Transform', ColumnTransformer(
    transformers=[
        ('minmax', min_max_transformer, min_max_features),
        ('onehot', categorical_transformer_onehot, categorical_features_onehot),
        ('ordinal', ordinal_transformer, ordinal_features),
    ], remainder='passthrough').set_output(transform='pandas')),

    # Drop Columns
    ('Drop', ColumnDropper(columns_to_drop)),
]

# Make a preprocessor pipeline
pipeline = Pipeline(steps=preprocessor)