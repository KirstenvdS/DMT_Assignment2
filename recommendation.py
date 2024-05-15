import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from time import time
from dataprep import generate_features, add_customer_profile, remove_outliers, impute_missing_values
from wtp import estimate_prior_wtp

n_customer_segments = 7

def calculate_wtp(beta_m, beta_price):
    return np.abs(beta_m/beta_price)

def get_topsis_weights(coefs):
    """Convert coefficients to WTP and to topsis weights."""
    n_predictors = coefs.shape[0]
    n_segments = coefs.shape[1]
    topsis_weights = np.zeros((n_predictors -1, n_segments)) # without intercept (index 0 in coefs)
    assert n_segments == n_customer_segments, "Indexing Error!"
    price_ind = 5
    for i in range(n_segments):
        # Calculate WTP
        beta_price = coefs[price_ind,i]
        wtps = np.zeros(n_predictors)
        topsis_weights[:,i] = np.linspace(0.5, 1e-4, n_predictors-1)
        for m in range(n_predictors):
            beta_m = coefs[m,i]
            wtps[m] = calculate_wtp(beta_m, beta_price)
            assert wtps[m] > 0, "Value error in WTP, should be > 0. "
        print(f"Wtps segment_ind {i}: {np.round(wtps,3)}")
        # Sort ascending, overwrite topsis weights
        topsis_weights[:,i] = topsis_weights[np.flip(np.argsort(wtps[1:])),i] # Index 0 is for intercept!
    print(f"Topsis weights, all segments, sorted according to wtps: {np.round(topsis_weights,3)}.")
    return topsis_weights

def normalize(df):
    """Normalize matrix for topsis"""
    for col in df.columns:
        sqrt_ss = np.sqrt(np.sum(df[col] ** 2))
        df.loc[:, col] = df.loc[:,col] / sqrt_ss
    print(f"Normalized matrix head: {np.round(df.head(),3)}")
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

def topsis(df,W, coefs, predictors):
    """TOPSIS implementation"""
    n_predictors = coefs.shape[0]
    n_segments = coefs.shape[1]
    segments = df["customer_segment"].unique()
    assert len(segments) == n_customer_segments == n_segments, "Error: Segment lengths not matching."
    assert np.size(W,0) == len(predictors), "Error: Number of predictors not matching."

    for i, segment in enumerate(segments):
        # Extract attribute matrix with hotel attributes (predictors)
        M = df.loc[df["customer_segment"] == segment, predictors]

        # Normalize M
        normalize(M)

        # Calculate weighted matrix by multiplying with TOPSIS weights
        for m, pred in enumerate(predictors):
            # Multiply whole column with weight
            M[pred] = M[pred] * W[m,i]
        print(f"Weighted, normalized M head: {np.round(M.head(),3)}")

        # Calculate ideal best and ideal worst distances
        M = ideal_best_worst_distance(M, coefs, i)

        # Calculate TOPSIS score: ideal worst distance / (ideal worst d + ideal best d)
        M["topsis_score"] = M["d_worst"] / (M["d_best"] + M["d_worst"])

        # Rank according to Topsis score
        M["Rank"] = M["topsis_score"].rank(method='max', ascending=False)

        print("Final Topsis score and rank: ", M.head())


if __name__ == '__main__':
    # Import dataset
    start = time()
    df = pd.read_csv("training_set_VU_DM.csv")
    # preprocessing
    df = df.sample(n=100000) # just for debugging
    df = impute_missing_values(df)
    df = remove_outliers(df)
    df = generate_features(df)

    # Cluster customer data with k-means and 7 clusters
    df = add_customer_profile(df, k=n_customer_segments)

    # Estimate prior WTP weights calculated from each customer segment
    coefs, predictors = estimate_prior_wtp(df)

    # TOPSIS preparation
    # Formulate TOPSIS weights (between 0.5 and > 0) based on WTP ranking
    W = get_topsis_weights(coefs)

    # TOPSIS
    topsis(df, W, coefs, predictors)
    end = time()
    print('TOPSIS total runtime: %.3f seconds' % (end - start))
