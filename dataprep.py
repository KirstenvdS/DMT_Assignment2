import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
import dask.dataframe as dd
from miceforest import ImputationKernel
# from sklearnex import patch_sklearn
# patch_sklearn()
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.metrics import silhouette_score






def remove_outliers(df):
    """Remove outliers"""
    #df.boxplot(column="price_usd") # outliers > 1e4
    #plt.show()
    #df.boxplot(column="prop_location_score1") #no outliers
    #plt.show()
    #df.boxplot(column="prop_location_score2") #no outliers
    #plt.show()
    #df.boxplot(column="prop_log_historical_price") # no outliers
    #plt.show()
    #df.boxplot(column="srch_length_of_stay") # 4 outliers > 30 nights
    #plt.show()
    #df.boxplot(column="srch_booking_window") # no outliers
    #plt.show()
    #df.boxplot(column="srch_adults_count")
    #plt.show()
    #df.boxplot(column="srch_children_count")
    #plt.show()
    #df.boxplot(column="srch_room_count")
    #plt.show()
    #df.boxplot(column="srch_query_affinity_score") # maybe exclude all values < 250
    #plt.show()
    #df.boxplot(column="orig_destination_distance")
    #plt.show()
    #df.boxplot(column="visitor_hist_adr_usd")
    #plt.show()
    #df.boxplot(column="visitor_hist_starrating")
    #plt.show()
    #df.boxplot(column="prop_starrating")
    #plt.show()
    #df.boxplot(column="prop_review_score")
    #plt.show()
    #df.boxplot(column="position")
    #plt.show()

    # Discard rows with outlier prices
    noutliers = len(df.loc[df["price_usd"] >= 1e4, ].index)
    print(f"Remove {noutliers} outliers with price >= 10,000.")
    df = df.loc[df["price_usd"] < 1e4, ]

    df.boxplot(column="gross_bookings_usd") # outliers > 10,000
    plt.show()

    # Discard rows with outlier length of stay
    noutliers = len(df.loc[df["srch_length_of_stay"] > 30,].index)
    print(f"Remove {noutliers} outliers with length of stay > 30.")
    df = df.loc[df["srch_length_of_stay"] <= 30,]

    return df


def impute_missing_values(df):
    """Impute missing values using x strategy"""
    N = len(df.index)

    # Impute numerical base features with missing = 0
    nmissing = sum(df["prop_location_score2"].isna())
    print(f"Replace {nmissing} missing values in prop_location_score_2. ")
    df.loc[df["prop_location_score2"].isna(), "prop_location_score2"] = 0
    nmissing = sum(df["prop_review_score"].isna())
    print(f"Replace {nmissing} missing values in prop_review_score. ")
    df.loc[df["prop_review_score"].isna(), "prop_review_score"] = 0
    return df


def generate_features(df):
    """Engineer new features and add them to the dataframe. Delete unused features. """
    N = len(df.index)

    # Customer previous spends
    cond_high_spends = df.visitor_hist_adr_usd > 160 # value obtained after looking at histogram
    cond_low_spends = df.visitor_hist_adr_usd <= 160
    df["customer_past_spends"] = np.select([cond_high_spends, cond_low_spends], ["high", "low"], pd.NA)
    #print("Customer past spends: ", df["customer_past_spends"].value_counts(dropna=False)/N)

    # Customer previous starrating
    cond_high_stars = df.visitor_hist_starrating > 3.5 # value obtained after looking at histogram
    cond_low_stars = df.visitor_hist_starrating <= 3.5
    df["customer_past_starrating"] = np.select([cond_high_stars, cond_low_stars], ["high", "low"], pd.NA)
    #print("Customer past starrating: ", df["customer_past_starrating"].value_counts(dropna=False)/N)

    # Stay type
    cond_single_night = (df.srch_saturday_night_bool == False) & (df.srch_length_of_stay == 1)
    cond_saturday_night = (df.srch_saturday_night_bool == True) & (df.srch_length_of_stay == 1)
    cond_weekend = (df.srch_saturday_night_bool == True) & (df.srch_length_of_stay == 2)
    cond_business_trip = (df.srch_saturday_night_bool == False) & (df.srch_length_of_stay > 1) \
                         & (df.srch_length_of_stay <= 5)
    cond_long_weekend = (df.srch_saturday_night_bool== True) & (df.srch_length_of_stay > 2) \
                        & (df.srch_length_of_stay <= 5)
    cond_long_stay = df.srch_length_of_stay > 5
    df["stay_type"] = np.select([cond_long_weekend, cond_weekend, cond_saturday_night, cond_long_stay,
                                 cond_single_night, cond_business_trip],
                                ["extended_weekend", "weekend", "saturday_night", "long_stay",
                                 "weekday_single_night", "business_trip"], pd.NA)
    #print("Stay type: ", df["stay_type"].value_counts(dropna=False)/N)



    # Travel type based on destination and origin
    cond_domestic = df.prop_country_id == df.visitor_location_country_id
    df["travel_type"] = np.select([cond_domestic], ["domestic"], "international")
    #print("Travel type: ", df["travel_type"].value_counts(dropna=False)/N)


    # Customer group
    cond_solo = (df.srch_adults_count == 1) & (df.srch_children_count == 0)
    cond_solo_parent_family = (df.srch_adults_count == 1) & (df.srch_children_count > 0)
    cond_couple = (df.srch_adults_count == 2) & (df.srch_children_count == 0)
    cond_nuclear_family = (df.srch_adults_count == 2) & (df.srch_children_count > 0)
    cond_group = (df.srch_adults_count > 2)
    df["customer_group"] = np.select([cond_solo, cond_solo_parent_family, cond_couple, cond_nuclear_family,
                                              cond_group], ['solo', 'solo_parent_family', 'couple', 'nuclear_family',
                                                            'group'], pd.NA)
    #print("Customer groups: ", df["customer_group"].value_counts(dropna=False)/N)

    # Customer booking lead time
    cutoff = 14 # value obtained from histogram of length of srch_booking_window
    cond_late = df.srch_booking_window <= cutoff
    cond_early_planner = df.srch_booking_window > cutoff
    df["customer_type"] = np.select([cond_late, cond_early_planner], ["late", "early_planner"], pd.NA)
    #print("Customer planning type: ", df["customer_type"].value_counts(dropna=False)/N)

    # Customer's day of travel type, proxy for trip plans
    cond_weekend = df["srch_saturday_night_bool"] == True
    df["day_of_travel_type"] = np.select([cond_weekend], ["weekend"], "weekday")
    #print("Day of travel type: ", df["day_of_travel_type"].value_counts(dropna=False)/N)

    return df

def clean_all(df):
    """All data cleaning steps in one function. (Easy use for other files such as recommendation)"""
    df = remove_outliers(df)
    df = impute_missing_values(df)
    df = generate_features(df)
    return df

def tune_k(df):
    # Initialize result data structures
    num_clusters = range(2,15,1)
    sil_scores = np.zeros(len(num_clusters))
    sses = np.zeros(len(num_clusters))
    print(f"Test for cluster sizes: {num_clusters}")

    # Subset dataframe
    subdf = df[["stay_type", "travel_type", "customer_group", "customer_type",
                "day_of_travel_type", "customer_past_spends", "customer_past_starrating"]]
    subdf_dummies = pd.get_dummies(subdf)
    subdf_norm = preprocessing.normalize(subdf_dummies)
    #subdf_dummies = subdf_dummies.sample(n= 1000) # comment this line if you want to run the clustering on the entire dataset
    #n_clusters = 10 #took 10 from article, but we should do our own hyperparameter tuning I think

    # Elbow method for k: calculate silhouette score + SSE for each number of clusters
    for i,n_clusters in enumerate(num_clusters):
        kmeans = KMeans(n_clusters, init='k-means++', random_state=0,
                    n_init='auto', max_iter=300, # n_init == 1 if init = k-means++
                    algorithm='elkan') # x 'elkan' is faster than 'llyod'
        print(f"Running k-means with {n_clusters} clusters... ")
        start = time()
        kmeans.fit(subdf_norm)
        end1 = time()
        print(f"K-means execution took {end1-start} seconds.") # 5 seconds
        sil_scores[i] = silhouette_score(subdf_norm, kmeans.labels_, metric='euclidean', sample_size=10000) # 100,000: 2 minutes
        end2 = time()
        sses[i] = kmeans.inertia_
        print(f"Silhouette score computation took {end2-end1} seconds.")
        print("Silhouette score: ", sil_scores[i])
        print(f"SSE: {sses[i]}")

    # Plot results
    plt.figure(figsize=(15,5))
    plt.subplot(121)
    plt.plot(num_clusters, sses, "x-")
    plt.title("Elbow method for optimal k")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("SSE (sum of squared errors)")
    plt.subplot(122)
    plt.plot(num_clusters, sil_scores, "x-")
    plt.title("Silhouette scores")
    plt.xlabel("k (number of clusters)")
    plt.ylabel("Silhouette score")
    plt.savefig("k-hyperparameter-tuning-results-normed-data-ss100,000.png")
    plt.show()

def add_customer_profile(df, k = 7):
    n_clusters = k
    subdf = df[["stay_type", "travel_type", "customer_group", "customer_type", "day_of_travel_type"]]
    subdf_dummies = pd.get_dummies(subdf)
    subdf_norm = preprocessing.normalize(subdf_dummies)
    kmeans = KMeans(n_clusters, init='k-means++', random_state=0,
                    n_init='auto', max_iter=300,  # n_init == 1 if init = k-means++
                    algorithm='elkan')  # x 'elkan' is faster than 'llyod'
    kmeans.fit(subdf_norm)
    df["customer_segment"] = kmeans.labels_
    return df

if __name__ == '__main__':
    # Import dataset
    start = time()
    df = dd.read_csv("training_set_VU_DM.csv")
    df = df.compute() # convert to pandas because no significant performance difference for further calculations

    # Generate customer information
    df = impute_missing_values(df)
    df = remove_outliers(df)
    df = generate_features(df)
    # Clustering
    #tune_k(df) # result: 7 is suitable number of clusters
    df = add_customer_profile(df, 7)
    print("New variable customer segment: ", df["customer_segment"].value_counts(dropna=False)/len(df.index))
    end = time()
    print(f"Total runtime: {end - start}")  # secs


