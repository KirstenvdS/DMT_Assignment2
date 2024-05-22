import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
from dataprep import generate_features, add_customer_profile, remove_outliers, impute_missing_values
from wtp import estimate_prior_wtp, predictors, customer_segments


def calculate_wtp(beta_m, beta_price):
    return np.abs(beta_m/beta_price)

def get_topsis_weights(coefs):
    """Convert coefficients to WTP and to topsis weights."""
    n_predictors = coefs.shape[0]
    n_segments = coefs.shape[1]
    topsis_weights = np.zeros((n_predictors -1, n_segments)) # without intercept (index 0 in coefs)
    assert n_segments == len(customer_segments), "Indexing Error!"
    price_ind = predictors.index("price_usd")

    for i in range(len(customer_segments)):
        # Calculate WTP
        beta_price = coefs[price_ind,i]
        wtps = np.zeros(n_predictors)
        wtp_sum = 0
        for m in range(n_predictors):
            beta_m = coefs[m,i]
            wtps[m] = calculate_wtp(beta_m, beta_price)
            assert wtps[m] > 0, "Value error in WTP, should be > 0. "
            wtp_sum += wtps[m]
        #print(f"Wtps segment_ind {i}: {np.round(wtps,3)}")
        # Sort ascending, overwrite topsis weights
        for m in range(n_predictors-1):
            topsis_weights[m,i] = wtps[m+1]/wtp_sum

    assert all(np.sum(topsis_weights,axis=1)) <= 1, "Sum of topsis weights must be smaller than 1"
    print(f"Topsis weights, all segments, sorted according to wtps: {np.round(topsis_weights,3)}.")
    return topsis_weights

def normalize(df):
    """Normalize matrix for topsis"""
    for col in df.columns:
        sqrt_ss = np.sqrt(np.sum(df[col] ** 2))
        if sqrt_ss > 0:
            df.loc[:, col] = df.loc[:,col] / sqrt_ss
        else :
            df.loc[:,col] = 0
    return df

def ideal_best_worst_distance(df, coefs, segment_ind):
    """TOPSIS calculate ideal best and ideal worst euclidean distance"""
    p_sln = df.max().values
    n_sln = df.min().values
    assert (np.size(coefs,0) -1) == len(p_sln) == len(n_sln), "Assertion Error: length of predictors does not match."

    tmp_worst = np.zeros((len(df.index), len(df.columns)))
    tmp_best = np.zeros((len(df.index), len(df.columns)))
    for m, col in enumerate(df.columns):
        # If negative impact, swap
        if coefs[m+1, segment_ind] < 0:
            p_sln[m], n_sln[m] = n_sln[m], p_sln[m]
        tmp_worst[:, m] = (df.loc[:,col] - n_sln[m])**2
        tmp_best[:,m] = (df.loc[:,col] - p_sln[m])**2

    df["d_worst"] = np.sqrt(np.sum(tmp_worst, axis=1))
    df["d_best"] = np.sqrt(np.sum(tmp_best, axis=1))
    return df

def topsis(df,W, coefs):
    """TOPSIS implementation"""
    n_predictors = coefs.shape[0]
    n_segments = coefs.shape[1]
    searches = df["srch_id"].unique()

    print(f"There are {len(searches)} different searches.")
    assert len(customer_segments) == n_segments, "Error: Segment lengths not matching."
    assert np.size(W,0) == len(predictors) == (n_predictors -1), "Error: Number of predictors not matching."

    for i, search_id in enumerate(searches):
        if i % 1000 == 0:
            print(f"Search {i} of {len(searches)}: {100* np.round(i/len(searches),2)} % complete")

        # Extract attribute matrix with hotel attributes (predictors)
        M = df.loc[df["srch_id"] == search_id, predictors]

        # Cast all columns to float
        M = M.astype(float)

        # Skip trivial case
        if len(M.index) == 1:
            M["topsis_score"] = np.infty
            M["rank"] = 1
            #print(f"Srch id {search_id} has only one alternative, skip Topsis. ")
            continue

        # Get segment for this search id
        #assert len(df.loc[df["srch_id"] == search_id, "customer_segment"].unique()) == 1
        segment = df.loc[df["srch_id"] == search_id, "customer_segment"].iloc[0]
        segment_id = np.where(customer_segments == segment)[0]

        # Normalize M
        M = normalize(M)

        # Calculate weighted matrix by multiplying with TOPSIS weights
        for m, pred in enumerate(predictors):
            # Multiply whole column with weight
            M[pred] = M[pred] * W[m,segment_id]
        #print(f"Weighted, normalized M head: {np.round(M.head(),3)}")

        # Calculate ideal best and ideal worst distances
        M = ideal_best_worst_distance(M, coefs, segment_id)

        # Calculate TOPSIS score: ideal worst distance / (ideal worst d + ideal best d)
        divisors = (M["d_best"] + M["d_worst"])
        # Divide in zero and nonzero divisors (to avoid division by zero because it would set topsis score to NaN)
        nonzero_inds = divisors.index[divisors != 0].tolist()
        zero_inds = divisors.index[divisors == 0].tolist()
        assert (len(nonzero_inds) + len(zero_inds)) == len(divisors)
        if len(zero_inds) >= 1:
            M.loc[zero_inds, "topsis_score"] = np.infty
        if len(nonzero_inds) >= 1:
            M.loc[nonzero_inds, "topsis_score"] = M.loc[nonzero_inds, "d_worst"] / divisors[nonzero_inds]

        # Rank according to Topsis score
        M["rank"] = M["topsis_score"].rank(method='max', ascending=False)

        #assert not M.isna().values.any(), "Assertion Error: Matrix contains NaN after TOPSIS."

        #print("Final Topsis score and rank: ", M.loc[:,["d_worst", "d_best", "topsis_score", "rank"]].head())
        df.loc[M.index, "rank"] = M.loc[:,"rank"]
    return df

def ndcg(df):
    """Calculate NDCG@5 (only possible with training data set)."""
    df_ideal_sort = df[["srch_id", "booking_bool", "click_bool"]].sort_values(by=["srch_id", "booking_bool", "click_bool"], ascending=[True, False, False])
    df_results_sort = df[["srch_id", "booking_bool", "click_bool", "rank"]].sort_values(by=["srch_id", "rank"])
    searches = df["srch_id"].unique()

    # first calculate IDCG
    idcg = 0
    dcg = 0
    for search in searches:
        for i in range(1,6):
            rel_ideal = 0
            rel_results = 0
            if df_ideal_sort.loc[df["srch_id"] == search, "booking_bool"].iloc[i-1] == True:
                rel_ideal = 5
            elif df_ideal_sort.loc[df["srch_id"] == search, "click_bool"].iloc[i-1] == True:
                rel_ideal = 1
            if df_results_sort.loc[df["srch_id"] == search, "booking_bool"].iloc[i-1] == True:
                rel_results = 5
            elif df_ideal_sort.loc[df["srch_id"] == search, "click_bool"].iloc[i-1] == True:
                rel_results = 1
            idcg += (2**(rel_ideal) - 1) / np.log2(i+1)
            dcg += (2**(rel_results) - 1)/ np.log2(i+1)
        ndcg5 = dcg/idcg
    return ndcg5

if __name__ == '__main__':
    # Import dataset
    start = time()
    df = pd.read_csv("training_set_VU_DM.csv")
    # preprocessing
    print(f"Number of rows: {len(df.index)}")
    df = df.iloc[:100000] # first 100,000 rows just for debugging
    df = impute_missing_values(df)
    df = remove_outliers(df)
    df = generate_features(df)

    # Cluster customer data with k-means and 7 clusters
    df = add_customer_profile(df, k=len(customer_segments))

    # Read beta coefficients from txt file (Run wtp.py to generate this file!)
    coefs = np.genfromtxt('coefs.csv', delimiter=',')

    # TOPSIS preparation
    # Formulate TOPSIS weights (between 0.5 and > 0) based on WTP ranking
    W = get_topsis_weights(coefs)

    # TOPSIS
    topsis_start = time()
    df = topsis(df, W, coefs)
    # Get NDCG (only possible if training data set)
    score = ndcg(df)
    print(f"TOPSIS NDCG@5: {score}")

    # Write results to submission file
    results = df[["srch_id", "prop_id", "rank"]]
    results = results.sort_values(by=["srch_id", "rank"])
    results[["srch_id", "prop_id"]].to_csv("submission_training.csv", index=False)
    end = time()

    print('TOPSIS runtime: %.3f seconds' % (end - topsis_start))
    print('Total runtime: %.3f seconds' % (end - start))
