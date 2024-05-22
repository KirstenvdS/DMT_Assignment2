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
    #print(df.shape)
    #print(df.info)
    #print(df.dtypes)

    # Boxplots for ratings distribution
    selected_cols = ["srch_booking_window", "srch_length_of_stay",
                     "srch_adults_count",
                     "srch_children_count", "srch_room_count"]
    df_ratings = df[selected_cols]
    plt.boxplot(df_ratings, labels=selected_cols)
    plt.title("Distribution of search attributes")
    plt.ylabel("Value")
    xticks_pos = range(1, len(selected_cols) +1)
    plt.xticks(ticks=xticks_pos, labels=selected_cols, rotation=45)
    ax = plt.gca()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig("boxplots_srch_attributes.png")
    plt.show()

    # Summarize boolean data (number of counts for "yes"): hotel chain, weekend, random positioning, clicked, booked
    print("Part of major hotel chain: \n", df["prop_brand_bool"].value_counts()/N)
    print("Sale price promotion displayed: \n", df["promotion_flag"].value_counts()/N)
    print("Saturday night stay: \n", df["srch_saturday_night_bool"].value_counts()/N)
    print("Random positioning: \n", df["random_bool"].value_counts()/N)
    #print("Competitor 1 has availability: \n", df["comp1_inv"].value_counts()/N)
    #print("Competitor 2 has availability: \n",df["comp2_inv"].value_counts()/N)
    #print("Competitor 3 has availability: \n",df["comp3_inv"].value_counts()/N)
    #print("Competitor 4 has availability: \n",df["comp4_inv"].value_counts()/N)
    #print("Competitor 5 has availability: \n",df["comp5_inv"].value_counts()/N)
    #print("Competitor 6 has availability: \n",df["comp6_inv"].value_counts()/N)
    #print("Competitor 7 has availability: \n", df["comp7_inv"].value_counts()/N)
    #print("Competitor 8 has availability: \n", df["comp8_inv"].value_counts()/N)
    print("Percentage of clicks in total: \n", df["click_bool"].sum()/N)
    print("Percentage of bookings in total: \n", df["booking_bool"].sum()/N)
    print("Number of bookings in total: \n", df["booking_bool"].sum())

    # Hotel properties
    print("Number of hotels: ", len(df["prop_id"].unique()))
    print("Number of hotel countries: ", len(df["prop_country_id"].unique()))
    print("avg hotel star rating : ", np.mean(df["prop_starrating"]))
    print("std hotel star rating : ", np.std(df["prop_starrating"]))
    print("avg hotel review score : ", np.mean(df["prop_review_score"]))
    print("std hotel review score : ", np.std(df["prop_review_score"]))
    print("avg hotel price : ", np.mean(df["price_usd"]))
    print("std hotel price : ", np.std(df["price_usd"]))

    # Number of clicks per search, number of bookings per search
    clicks_per_search = df.groupby("srch_id")["click_bool"].sum()
    bookings_per_search = df.groupby("srch_id")["booking_bool"].sum()
    print("Avg. number of clicks per search: ", np.mean(clicks_per_search))
    print("std. number of clicks per search: ", np.std(clicks_per_search))
    print("Avg. number of bookings per search: ", np.mean(bookings_per_search))
    print("std. number of bookings per search: ", np.std(bookings_per_search))

    # Search properties
    print("Avg. number of adults: ", np.mean(df["srch_adults_count"]))
    print("std. number of adults: ", np.std(df["srch_adults_count"]))
    print("avg. number of children: ", np.std(df["srch_children_count"]))
    print("std. number of children: ", np.std(df["srch_children_count"]))
    print("avg. number of rooms: ", np.mean(df["srch_room_count"]))
    print("std. number of rooms: ", np.std(df["srch_room_count"]))
    print("avg. length of stay: ", np.mean(df["srch_length_of_stay"]))
    print("std. length of stay: ", np.std(df["srch_length_of_stay"]))
    print("avg. days booked ahead: ", np.mean(df["srch_booking_window"]))
    print("std. days booked ahead: ", np.std(df["srch_booking_window"]))

    # Average position of purchased offers
    positions_of_bookings = df.loc[df["booking_bool"] == True, "position"]
    print("Avg. position of purchased hotels: ", np.mean(positions_of_bookings))
    print("std. position of purchased hotels: ", np.std(positions_of_bookings))
    print("Min position of purchased hotels: ", np.min(positions_of_bookings))
    print("Max position of purchased hotels: ", np.max(positions_of_bookings))

    # Number results per searches
    searches = df.groupby("srch_id").size()
    print("Avg. Number of results per search: ", np.mean(searches))
    print("std. Number of results per search: ", np.std(searches))
    print("Min of Number of results per search: ", np.min(searches))
    print("Max of Number of results per search: ", np.max(searches))

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

if __name__ == '__main__':
    # Import dataset
    start = time()
    df = dd.read_csv("training_set_VU_DM.csv")
    end = time()
    print(f"Loading csv takes {end-start}.") #22 secs with pandas, 0.02 s with dask

    # Plot features
    plot_features(df) # runtime ~ 45 seconds
    plot_missing_values(df) # runtime ~ 20 seconds
    end2 = time()
    print(f"Total runtime: {end2- start}") # 69 seconds with pandas

