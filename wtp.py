from dataprep import generate_features, add_customer_profile, remove_outliers, impute_missing_values
from time import time
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import dask.dataframe as dd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
import seaborn as sns

def estimate_prior_wtp(df):
    """Estimating a prior WTP per segment using multinominal logistic regression. See Bayesrec paper section 3.2.2."""
    # Subset dataframe and only regard customers(=searches) who bought something
    booking_srch = df.loc[df["booking_bool"] == True, "srch_id"]
    # Select features according to paper, section 3.2.2
    subdf = df.loc[df["srch_id"].isin(booking_srch), ["prop_starrating", "prop_review_score", "prop_location_score1",
                                                  "prop_location_score2", "price_usd", "promotion_flag",
                                                  "prop_brand_bool", "customer_segment", "booking_bool"]]
    print("Percentage of searches with bookings: ", len(subdf.index) / len(df.index))
    print("Customer segments in reduced dataframe: ", subdf["customer_segment"].value_counts())
    print("Customer segments (%) in reduced dataframe: ", subdf["customer_segment"].value_counts() / len(subdf.index))
    # Assert no missing values (important for multinomial logit)
    assert not subdf.isna().values.any(), "Assertion Error: Subdf contains NaN"

    # Construct X and y for logit
    customer_segments = subdf["customer_segment"].unique()
    predictors = ["prop_starrating", "prop_review_score", "prop_location_score1", "prop_location_score2", "price_usd",
               "promotion_flag", "prop_brand_bool"]
    coefficients = np.zeros((len(predictors)+1,len(customer_segments)))
    for i, segment in enumerate(customer_segments):
        X = subdf.loc[subdf["customer_segment"] == segment, predictors]
        y = subdf.loc[subdf["customer_segment"] == segment, ["booking_bool"]]
        # Multinominal logit
        model = LogisticRegression(multi_class='auto', solver='lbfgs')
        model.fit(X,y)
        coefficients[0,i] = model.intercept_
        coefficients[1:,i] = model.coef_
        print(f"Segment {segment} Coefficients: {np.round(coefficients[:,i],3)}")

        # Based on model, estimate distribution for this segment over all data
        #all_obs_subdf = df.loc[df["customer_segment"] == segment, predictors]
        booking_proba = model.predict(X) # underestimates bookings!!
        unique, counts = np.unique(booking_proba, return_counts=True)
        print("booking probabilities: \n", np.asarray((unique, counts)).T)
        sns.kdeplot(model.classes_).set_title(f'Booking probability distribution (Segment {segment})')
        plt.savefig(f"booking_proba_segment_{segment}.png")
        plt.show()

def hierarchical_bayes(df):



if __name__ == '__main__':
    start = time()
    df = dd.read_csv("training_set_VU_DM.csv")
    df = df.compute()  # convert to pandas because no significant performance difference for further calculations
    # preprocessing
    #df = df.sample(n=100000) # just for debugging
    df = impute_missing_values(df)
    df = remove_outliers(df)
    df = generate_features(df)
    df = add_customer_profile(df, k=7)

    # Estimate prior WTP
    estimate_prior_wtp(df)
    end = time()
    print('WTP total runtime: %.3f seconds' % (end-start))
