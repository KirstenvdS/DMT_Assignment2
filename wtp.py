from dataprep import generate_features, add_customer_profile, remove_outliers, impute_missing_values
from time import time
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import numpy as np
import dask.dataframe as dd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import seaborn as sns

customer_segments = [0,1,2,3,4,5,6]
predictors = ["prop_starrating", "prop_review_score", "prop_location_score1", "prop_location_score2", "price_usd",
                  "promotion_flag", "prop_brand_bool"]

def estimate_prior_wtp(df):
    """Estimating a prior WTP per segment using multinominal logistic regression. See Bayesrec paper section 3.2.2."""
    # Subset dataframe and only regard customers(=searches) who bought something
    booking_srch = df.loc[df["booking_bool"] == True, "srch_id"]
    # Select features according to paper, section 3.2.2
    features = predictors + ["customer_segment", "booking_bool"]
    subdf = df.loc[df["srch_id"].isin(booking_srch), features]

    #print("Percentage of searches with bookings: ", len(subdf.index) / len(df.index))
    #print("Customer segments in reduced dataframe: ", subdf["customer_segment"].value_counts())
    #print("Customer segments (%) in reduced dataframe: ", subdf["customer_segment"].value_counts() / len(subdf.index))

    # Assert no missing values (important for multinomial logit)
    assert not subdf.isna().values.any(), "Assertion Error: Subdf contains NaN"

    # Construct X and y for logit
    coefficients = np.zeros((len(predictors)+1,len(customer_segments)))
    for i, segment in enumerate(customer_segments):
        X = subdf.loc[subdf["customer_segment"] == segment, predictors]
        y = subdf.loc[subdf["customer_segment"] == segment, ["booking_bool"]]
        # Multinominal logit
        model = LogisticRegression(multi_class='auto', solver='lbfgs')
        model.fit(X,y)
        coefficients[0,i] = model.intercept_
        coefficients[1:,i] = model.coef_
        print(f"Segment {segment} (id: {i}), Coefficients: {np.round(coefficients[:,i],3)}")

        # Based on model, estimate distribution for this segment over all data
        all_obs_subdf = df.loc[df["customer_segment"] == segment, predictors]
        booking_proba = model.predict(all_obs_subdf) # underestimates bookings!!
        unique, counts = np.unique(booking_proba, return_counts=True)
        print("booking probabilities: \n", np.asarray((unique, counts)).T)
    return coefficients

def logit(mu):
    return np.log(mu/(1-mu))

def hierarchical_bayes(df, coefs):
    """Hierarchical bayes model"""

    segment_ind = 0
    segment = 1
    customer_attributes = ["visitor_location_country_id", "visitor_hist_adr_usd",
                            "visitor_hist_starrating", "srch_adults_count", "srch_children_count"]
    all_obs_subdf = df.loc[df["customer_segment"] == segment, predictors].astype(float)
    Y = df.loc[df["customer_segment"] == segment, "booking_bool"].astype(float)
    W = df.loc[df["customer_segment"] == segment, customer_attributes].astype(float)
    intercept = coefs[0, segment_ind]x
    hb_model = pm.Model()
    with hb_model:
        # Priors are semi-informative based on multinominal logit
        alpha = pm.Normal(name='alpha', mu=intercept, sigma=1e-4)
        B=1 #?
        beta = pm.Normal('beta', mu=coefs[1:, segment_ind], sigma=B, shape=len(predictors))
        print(f"beta: {beta[:]}, all_obs_subdf = {all_obs_subdf.head()}")
        D=1 #?
        b = pm.Normal("b", mu=0, sigma=D, shape=len(customer_attributes))
        sigma_e = 1 #?
        eta = pm.Normal("eta", mu = 0, sigma = sigma_e)

        # Expected value of the outcome
        mu = alpha + (beta[:] * all_obs_subdf).sum(axis=1) +(b * W).sum(axis=1) + eta
        print(f"Calculated mu: {mu[0]}")
        Y_obs = pm.Bernoulli('Y_obs', p=logit(mu), observed=Y)

        step = pm.Metropolis()
        # sample with 4 independent Markov chains
        trace = pm.sample(draws=10000, chains=4, step=step, return_inferencedata=True)
    pm.traceplot(trace)
    pm.plot_posterior(trace)
    return df


if __name__ == '__main__':
    start = time()
    df = dd.read_csv("training_set_VU_DM.csv")
    df = df.compute()  # convert to pandas because no significant performance difference for further calculations
    # preprocessing
    df = df.iloc[:100000] # just for debugging
    df = impute_missing_values(df)
    df = remove_outliers(df)
    df = generate_features(df)
    df = add_customer_profile(df, k=7)

    # Estimate prior WTP
    #coefs = estimate_prior_wtp(df)
    # Write results to csv, only calculate once
    #np.savetxt("coefs.csv", coefs, delimiter=",")
    coefs = np.genfromtxt('coefs.csv', delimiter=',')

    # Hierarchical Bayes for WTP
    hierarchical_bayes(df, coefs)
    end = time()
    print('WTP total runtime: %.3f seconds' % (end-start))
