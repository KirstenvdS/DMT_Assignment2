import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
import dask.dataframe as dd
import hvplot.dask


def plot_features(df):
    """Just some plots of feature values, distribution, outliers,  aso"""
    # Size of dataframe
    print(df.shape)
    print(df.info)
    print(df.dtypes)

    # Make boxplots for variables with range of numeric variables, see Assignment2 pdf, Table 1
    df_num = df.select_dtypes(include=['float64'])
    # Use hvplot for plotting because dataframe is too large!
    boxplots = df_num.hvplot.box()
    #hvplot.show(boxplots)
    return


def plot_missing_values(df):
    """Plot missing values for each feature"""
    return

def plot_correlations(df):
    """Plot correlations and other relationships between variables"""
    return

if __name__ == '__main__':
    # Execute once: Import dataset
    start = time()
    df = dd.read_csv("training_set_VU_DM.csv")
    end = time()
    print(f"Loading csv takes {end-start}.") #22 secs with pandas, 0.02 s with dask
    # Execute once: store to hdf5 for faster imports
    #df.to_hdf("store.h5", key="table", append=True)

    # Now always import from hdf5
    #start= time()
    #store = dd.read_hdf("store.h5", "table")
    #end = time()
    #print(f"Loading hdf5 takes {end - start}. ") # 13 seconds, much faster than csv
    #df = dd.DataFrame(store)
    # Plot features
    plot_features(df)
    plot_missing_values(df)
    plot_correlations(df)
    end2 = time()
    print(f"Total runtime: {end2- start}") # 69 seconds with pandas

