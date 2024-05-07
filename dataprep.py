import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
import dask.dataframe as dd


def remove_outliers(df):
    """Remove outliers"""
    return df


def impute_missing_values(df):
    """Impute missing values using x strategy"""
    return df


def generate_features(df):
    """Engineer new features and add them to the dataframe. Delete unused features. """
    N = len(df.index)

    # Stay type
    cond_single_night = (df.srch_saturday_night_bool == False) & (df.srch_length_of_stay == 1)
    cond_saturday_night = (df.srch_saturday_night_bool == True) & (df.srch_length_of_stay == 1)
    cond_weekend = (df.srch_saturday_night_bool == True) & (df.srch_length_of_stay == 2)
    cond_business_trip = (df.srch_saturday_night_bool == False) & (df.srch_length_of_stay > 1)
    cond_long_weekend = (df.srch_saturday_night_bool== True) & (df.srch_length_of_stay > 2)
    cond_long_stay = df.srch_length_of_stay > 5 # saturday night must be included
    df["stay_type"] = np.select([cond_long_weekend, cond_weekend, cond_saturday_night, cond_long_stay,
                                 cond_single_night, cond_business_trip],
                                ["extended_weekend", "weekend", "saturday_night", "long_stay",
                                 "weekday_single_night", "business_trip"], pd.NA)
    print("Stay type: ", df["stay_type"].value_counts(dropna=False)/N)



    # Travel type based on destination and origin
    cond_domestic = df.prop_country_id == df.visitor_location_country_id
    df["travel_type"] = np.select([cond_domestic], ["domestic"], "international")
    print("Travel type: ", df["travel_type"].value_counts(dropna=False)/N)


    # Customer group
    cond_solo = (df.srch_adults_count == 1) & (df.srch_children_count == 0)
    cond_solo_parent_family = (df.srch_adults_count == 1) & (df.srch_children_count > 0)
    cond_couple = (df.srch_adults_count == 2) & (df.srch_children_count == 0)
    cond_nuclear_family = (df.srch_adults_count == 2) & (df.srch_children_count > 0)
    cond_group = (df.srch_adults_count > 2)
    df["customer_group"] = np.select([cond_solo, cond_solo_parent_family, cond_couple, cond_nuclear_family,
                                              cond_group], ['solo', 'solo_parent_family', 'couple', 'nuclear_family',
                                                            'group'], pd.NA)
    print("Customer groups: ", df["customer_group"].value_counts(dropna=False)/N)

    # Customer booking lead time
    df.hist(column="srch_booking_window", bins=30)
    plt.show()
    cutoff = 14
    cond_late = df.srch_booking_window <= cutoff
    cond_early_planner = df.srch_booking_window > cutoff
    df["customer_type"] = np.select([cond_late, cond_early_planner], ["late", "early_planner"], pd.NA)
    print("Customer planning type: ", df["customer_type"].value_counts(dropna=False)/N)

    # Customer's day of travel type, proxy for trip plans
    cond_weekend = df["srch_saturday_night_bool"] == True
    df["day_of_travel_type"] = np.select([cond_weekend], ["weekend"], "weekday")
    print("Day of travel type: ", df["day_of_travel_type"].value_counts(dropna=False)/N)

    return df

def clean_all(df):
    """All data cleaning steps in one function. (Easy use for other files such as recommendation)"""
    df = remove_outliers(df)
    df = impute_missing_values(df)
    df = generate_features(df)
    return df

if __name__ == '__main__':
    # Import dataset
    start = time()
    df = dd.read_csv("training_set_VU_DM.csv")
    df = df.compute() # convert to pandas because no significant performance difference for further calculations
    generate_features(df)
    end = time()
    print(f"Total runtime: {end - start}")  # secs


