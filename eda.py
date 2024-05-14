import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
import dask.dataframe as dd
import hvplot.dask


def plot_features(df):
    """Just some plots of feature values, distribution, outliers,  aso"""
    df = df.compute() # convert to pandas
    N = len(df.index)

    # Size of dataframe
    print(df.shape)
    print(df.info)
    print(df.dtypes)

    # Boxplots for ratings distribution
    selected_cols = ["visitor_hist_starrating", "prop_starrating", "prop_review_score",
                     "srch_adults_count",
                     "srch_children_count", "srch_room_count"]
    df_ratings = df[selected_cols]
    df_ratings.boxplot()
    plt.title("Distribution of some variables")
    plt.ylabel("Value")
    xticks_pos = range(1, len(selected_cols) +1)
    plt.xticks(ticks=xticks_pos, labels=selected_cols, rotation=90)
    plt.tight_layout()
    plt.savefig("boxplots_ratings_srch_attributes.png")
    plt.show()

    # Summarize boolean data (number of counts for "yes"): hotel chain, weekend, random positioning, clicked, booked
    print("Part of major hotel chain: \n", df["prop_brand_bool"].value_counts()/N)
    print("Sale price promotion displayed: \n", df["promotion_flag"].value_counts()/N)
    print("Saturday night stay: \n", df["srch_saturday_night_bool"].value_counts()/N)
    print("Random positioning: \n", df["random_bool"].value_counts()/N)
    print("Competitor 1 has availability: \n", df["comp1_inv"].value_counts()/N)
    print("Competitor 2 has availability: \n",df["comp2_inv"].value_counts()/N)
    print("Competitor 3 has availability: \n",df["comp3_inv"].value_counts()/N)
    print("Competitor 4 has availability: \n",df["comp4_inv"].value_counts()/N)
    print("Competitor 5 has availability: \n",df["comp5_inv"].value_counts()/N)
    print("Competitor 6 has availability: \n",df["comp6_inv"].value_counts()/N)
    print("Competitor 7 has availability: \n", df["comp7_inv"].value_counts()/N)
    print("Competitor 8 has availability: \n", df["comp8_inv"].value_counts()/N)
    print("Percentage of clicks in total: \n", df["click_bool"].sum()/N)
    print("Percentage of bookings in total: \n", df["booking_bool"].sum()/N)
    print("Number of bookings in total: \n", df["booking_bool"].sum())

    # Number of clicks per search, number of bookings per search

    # Average position of purchased offers

    #

    # Number of properties, countries, searches, destinations, ... ("id" variables)
    print("Number of searches: ", df["srch_id"].nunique())
    print("Number of hotels/properties: ", df["prop_id"].nunique())
    print("Number of Expedia sites: ", df["site_id"].nunique())
    print("Number of property countries: ", df["prop_country_id"].nunique())
    print("Number of search destinations: ", df["srch_destination_id"].nunique())
    print("Number of visitor country locations: ", df["visitor_location_country_id"].nunique())

    return


def plot_missing_values(df):
    """Plot missing values for each feature"""
    df_pd = df.compute() # convert to pandas
    start = time()
    missing_values_pd = 100 * df_pd.isnull().sum()/len(df.index)
    end = time()
    print(f"Missing value computation takes {end-start} with Pandas") # 14.5 seconds
    print("pandas NA dataframe head: ", missing_values_pd.head())

    plt.figure().set_figwidth(12)
    missing_values_pd.plot.bar()
    plt.title("Number of missing values for each attribute")
    plt.xlabel("Variable")
    plt.ylabel("% missing")
    plt.axhline(100, color="grey")
    plt.tight_layout()
    plt.savefig("missing_values_all_attributes.png")
    plt.show()
    return

def plot_correlations(df):
    """Plot correlations and other relationships between variables"""
    return

if __name__ == '__main__':
    # Import dataset
    start = time()
    df = dd.read_csv("training_set_VU_DM.csv")
    end = time()
    print(f"Loading csv takes {end-start}.") #22 secs with pandas, 0.02 s with dask

    # Plot features
    plot_features(df) # runtime ~ 45 seconds
    plot_missing_values(df) # runtime ~ 20 seconds
    plot_correlations(df)
    end2 = time()
    print(f"Total runtime: {end2- start}") # 69 seconds with pandas

