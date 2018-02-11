#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma (CIH745)
    Data Science Co-op
    Summer 2017
****************************************

Module: Data Impute
Purpose: Splices original datasets into multiple subsets using settings & transformations found in config &
         common/dataselect

"""
from numpy import random
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def insert_random_var(seed, new_var, dataframe):
    """Function to insert random variable into Pandas DataFrame.
    """
    random.seed(seed)
    dataframe[new_var] = random.random_sample(dataframe.shape[0])

def oneHotEncode(df, columns_to_encode, encode_nan=True, le_dict = {}):
    if encode_nan:
        for col in columns_to_encode:
            parameter = "Nan"
            df[col].fillna(parameter, inplace=True)
            df[col] = df[col].apply(str)
    if le_dict:
        encoding_cols = le_dict.keys()
        if set(encoding_cols) != set(columns_to_encode):
            print("Label Encoder dict does not have all encoding cols needed.")
            return False,False
        train = False
    else:
        train = True

    for feature in columns_to_encode:
        if train:
            le = LabelEncoder()
            le_dict[feature] = le
            df[feature] = le_dict[feature].fit_transform(df[feature])
        else:
            df[feature] = le_dict[feature].transform(df[feature])

    return le_dict

#function used to select subset of another list, elements in order of first list
def ordered_subset(preserve_order_list, cut_list):
    d = {k:v for v,k in enumerate(preserve_order_list)}
    new = set(preserve_order_list).intersection(cut_list)
    new = list(new)
    new.sort(key=d.get)
    return new

# Imputation functions:
# All of the following operations are in place
def impute_categories(df, category_cols):
    the_cols = df.columns.tolist()
    category_cols = ordered_subset(the_cols, category_cols)
    labelencode_dict = oneHotEncode(df, category_cols)
    return labelencode_dict

def apply_le_dict(df, category_cols, le_dict):
    oneHotEncode(df, category_cols, True, le_dict)

def get_means(df, impute_to_mean):
    the_cols = df.columns.tolist()
    impute_to_mean = ordered_subset(the_cols, impute_to_mean)
    df_means = df[impute_to_mean].mean()
    mean_dict = dict(zip(df_means.index.tolist(), df_means.values.tolist()))
    return mean_dict

def impute_na_rows(df, remove_na_rows):
    the_cols = df.columns.tolist()
    remove_na_rows = ordered_subset(the_cols, remove_na_rows)
    df.dropna(subset=remove_na_rows, inplace=True)

def impute_blacklist(df, blacklist):
    df.drop([col for col in blacklist if col in df], axis=1, inplace=True)

def add_random_variables(df, y, random_variables):
    y_max = df[y].max()
    y_min = df[y].min()
    y_range = y_max - y_min
    for r_var in random_variables:
        r_state = abs(hash(r_var)) % 10 ** 4 #create a seed based on the r.v. name
        insert_random_var(r_state, r_var, df)

def impute_conditions(df, conditions):
    for cond in conditions:
        df.query(cond, inplace=True)
        
def impute_to_value(df, impute_values):
    imputation = pd.Series(impute_values)
    df.fillna(imputation, inplace=True)

def impute_it(raw_df, y, category_cols, impute_to_mean, impute_to_value, remove_na_rows, blacklist, random_variables, conditions):
    print("Shape of data going in: " + str(raw_df.shape))
    raw_df.sort_values(y)
    the_cols = raw_df.columns.tolist()

    ## IMPUTATIONS
    #remove all rows with NA for these columns
    if remove_na_rows:
        remove_na_rows = ordered_subset(the_cols, remove_na_rows)
        raw_df = raw_df.dropna(subset=remove_na_rows)
        the_cols = raw_df.columns.tolist()

    #apply conditions on columns
    if conditions:
        for cond in conditions:
            raw_df = raw_df.query(cond)
        the_cols = raw_df.columns.tolist()

    #remove blacklisted features
    raw_df = raw_df.drop([col for col in blacklist if col in raw_df], axis=1)
    the_cols = raw_df.columns.tolist()

    #insert random variables for importance testing
    y_max = raw_df[y].max()
    y_min = raw_df[y].min()
    y_range = y_max - y_min
    for r_var in random_variables:
        r_state = abs(hash(r_var)) % 10 ** 4
        insert_random_var(r_state, r_var, raw_df)
    the_cols = raw_df.columns.tolist()

    #impute missings to mean
    if impute_to_mean:
        impute_to_mean = ordered_subset(the_cols, impute_to_mean)
        df_means = raw_df[impute_to_mean].mean()
        new_df = raw_df.fillna(df_means)
        #save imputed vals for items in save_imputed_rows to another file
        if save_imputed_rows:
            impute_dir = "./"
            for col in save_imputed_rows:
                curr_df = new_df[raw_df[col].isnull()]
                curr_df.to_csv(impute_dir + '/' + col + '_imputed_data.csv')
        #save the means you have imputed, if you need to see it later
        mean_dict = dict(zip(df_means.index.tolist(), df_means.values.tolist()))
        #set the raw_df to the new_df 
        raw_df = new_df
        del new_df

    #impute missings to values given
    if impute_to_value:
        imputation = pd.Series(impute_to_value)
        raw_df = raw_df.fillna(imputation)

    #encode categorical variables
    if category_cols:
        category_cols = ordered_subset(the_cols, category_cols)
        raw_df, labelencode_dict = oneHotEncode(raw_df, category_cols)

    ## OUTPUT
    #export impute dict
    impute_dict = {**mean_dict, **impute_to_value}

    #reindex & export imputed dataset
    raw_df.index = list(range(len(raw_df)))
    print("Shape of data going out: " + str(raw_df.shape))

    return raw_df, labelencode_dict, impute_dict

