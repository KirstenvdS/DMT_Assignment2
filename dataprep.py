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
from sklearn.model_selection import train_test_split





def remove_outliers(df):
    """Remove outliers"""
    return df


def impute_missing_values(df):
    """Impute missing values using x strategy"""
    N = len(df.index)

    # Impute customer previous star ratings & previous spends
    df_sub = df[["visitor_location_country_id", "srch_destination_id", "srch_length_of_stay", "srch_booking_window",
                 "srch_adults_count", "srch_children_count", "srch_room_count", "srch_saturday_night_bool",
                 "orig_destination_distance", "customer_past_spends", "customer_past_starrating", "customer_type",
                 "travel_type", "stay_type", "customer_group", "day_of_travel_type"]]
    cat_features = ["visitor_location_country_id", "srch_destination_id",
                    "srch_saturday_night_bool", "customer_type", "customer_past_spends", "customer_past_starrating",
                    "travel_type", "stay_type", "customer_group", "day_of_travel_type"]

    for c in cat_features:
        df_sub.loc[:, c] = pd.Categorical(df_sub[c])

    print(df_sub.dtypes)

    # TODO: imputation
    #print("Customer past spends after imputation: ", df_sub_imp["customer_past_spends"].value_counts(dropna=False)/N)
    #print("Customer past starrating after imputation: ", df_sub_imp["customer_past_starrating"].value_counts(dropna=False)/N)

    return df


def generate_features(df):
    """Engineer new features and add them to the dataframe. Delete unused features. """
    N = len(df.index)

    # Customer previous spends
    cond_high_spends = df.visitor_hist_adr_usd > 160 # value obtained after looking at histogram
    cond_low_spends = df.visitor_hist_adr_usd <= 160
    df["customer_past_spends"] = np.select([cond_high_spends, cond_low_spends], ["high", "low"], pd.NA)
    print("Customer past spends: ", df["customer_past_spends"].value_counts(dropna=False)/N)

    # Customer previous starrating
    cond_high_stars = df.visitor_hist_starrating > 3.5 # value obtained after looking at histogram
    cond_low_stars = df.visitor_hist_starrating <= 3.5
    df["customer_past_starrating"] = np.select([cond_high_stars, cond_low_stars], ["high", "low"], pd.NA)
    print("Customer past starrating: ", df["customer_past_starrating"].value_counts(dropna=False)/N)

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
    cutoff = 14 # value obtained from histogram of length of srch_booking_window
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

def clustering(df):
    # Initialize result data structures
    num_clusters = range(2,15,1)
    sil_scores = np.zeros(len(num_clusters))
    sses = np.zeros(len(num_clusters))
    print(f"Test for cluster sizes: {num_clusters}")

    # Subset dataframe
    subdf = df[["stay_type", "travel_type", "customer_group", "customer_type", "day_of_travel_type"]]
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
        sil_scores[i] = silhouette_score(subdf_norm, kmeans.labels_, metric='euclidean', sample_size=100000) # 100,000: 2 minutes
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



if __name__ == '__main__':
    # Import dataset
    start = time()
    df = dd.read_csv("training_set_VU_DM.csv")
    df = df.compute() # convert to pandas because no significant performance difference for further calculations
    df = generate_features(df)
    #impute_missing_values(df)
    clustering(df)
    end = time()
    print(f"Total runtime: {end - start}")  # secs


