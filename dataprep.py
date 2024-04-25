import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def remove_outliers(df):
    """Remove outliers"""
    return df


def impute_missing_values(df):
    """Impute missing values using x strategy"""
    return df


def generate_features(df):
    """Engineer new features and add them to the dataframe. Delete unused features. """
    return df

def clean_all(df):
    """All data cleaning steps in one function. (Easy use for other files such as recommendation)"""
    df = remove_outliers(df)
    df = impute_missing_values(df)
    df = generate_features(df)
    return df

if __name__ == '__main__':
    # Import dataset
    df = None
    # Clean dataset
    df = clean_all(df)

