"""Fold-fitted one-hot sensitivity for four reviewer-named nominal predictors.

Codes are preserved as labels, not recoded semantically. Unknown validation levels
produce all-zero indicators. Fit vocabulary uses training rows only.
"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
NOMINAL_COLUMNS=['COUNTRY','EMPLOYMENT','MARITAL_STATUS','GENDER']

def categorical_model(estimator):
    transform=ColumnTransformer([('nominal',OneHotEncoder(handle_unknown='ignore',sparse_output=False),NOMINAL_COLUMNS)],remainder='passthrough',sparse_threshold=0)
    return Pipeline([('encoding',transform),('model',estimator)])
