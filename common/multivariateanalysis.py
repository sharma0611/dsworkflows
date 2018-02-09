#/usr/bin/env python3

import csv
import os
import pandas as pd
import numpy as np

#RAW DATAFRAME PROCESSING ANALYSIS

#function used to get information about all cols in a dataframe
def variable_info(df, category_threshold=30):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    missing_vals_df = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    summary_stats = df.describe().transpose()
    types = pd.DataFrame(pd.Series(df.dtypes, dtype="category"), columns=['type'])
    object_cols = df.select_dtypes(include=['object']).columns
    number_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = []
    categorical_df = []
    non_categorical_object_cols = []

    for col in object_cols:
        curr_series = df[col]
        short_series = curr_series.sample(500)
        short_uniques = short_series.unique()
        short_uniques_num = len(short_uniques)
        if short_uniques_num < category_threshold:
            all_uniques = curr_series.unique()
            num_uniques = len(all_uniques)
            if num_uniques < category_threshold:
                d = {'numberofcategories': num_uniques,
                     'allcategories': str(all_uniques.tolist())}
                categorical_df.append(d)
                categorical_cols.append(col)
            else:
                non_categorical_object_cols.append(col)
        else:
            non_categorical_object_cols.append(col)

    categorical_df = pd.DataFrame(categorical_df, index=categorical_cols)
    output = pd.concat([missing_vals_df, types, categorical_df, summary_stats], axis=1)
    output = output.sort_values(["type", "% of Total Values"], ascending=[True, False])
    number_cols = sorted(number_cols, key=lambda col: output.loc[str(col),'% of Total Values'], reverse=True)
    categorical_cols = sorted(categorical_cols, key=lambda col: output.loc[str(col),'% of Total Values'],
                              reverse=True)
    non_categorical_object_cols = sorted(non_categorical_object_cols,
                                         key=lambda col: output.loc[str(col),'% of Total Values'],
                                         reverse=True)
    return output, categorical_cols, non_categorical_object_cols, number_cols


#function to get variable info and post summary analysis
def full_analysis(mydf, exportingdirectory, category_threshold=40):
    if not os.path.isdir(exportingdirectory):
        os.mkdir(exportingdirectory)

    #Variable Data
    var_info_df,categorical_vars,text_vars,number_cols = variable_info(mydf, category_threshold)
    var_info_df.to_csv(exportingdirectory + '/features_info.csv')

    #Summary Data
    with open(exportingdirectory + '/summary_info.csv', 'w') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',')
        mywriter.writerow(["Total Features", len(mydf.columns)])
        features_na = mydf.columns[mydf.isnull().any()].tolist()
        mywriter.writerow(["Features with a NA", len(features_na),
                           str(100 * len(features_na)/mydf.shape[1])+'%',
                           str(features_na)])
        mywriter.writerow(["Total Rows", len(mydf)])
        rows_na = mydf.shape[0] - mydf.dropna().shape[0]
        mywriter.writerow(["Rows with a NA",rows_na, str(100 * rows_na/mydf.shape[0]) + '%'])
        mywriter.writerow([])
        mywriter.writerow(["Dtypes:"])
        all_types = list(set(mydf.dtypes))
        g = mydf.columns.to_series().groupby(mydf.dtypes).groups
        total = 0

        for curr_type in all_types:
            curr_cols = g[curr_type].tolist()
            mywriter.writerow([str(curr_type),len(curr_cols),"",str(curr_cols)])
            total = total + len(curr_cols)
        mywriter.writerow(["TOTAL:", total])
        mywriter.writerow([])
        mywriter.writerow(["Processed:"])
        mywriter.writerow(["Categorical:", len(categorical_vars), len(categorical_vars)/mydf.shape[1],
                           str(categorical_vars)])
        categorical_vars_na = set(categorical_vars).intersection(features_na)
        categorical_vars_na = list(categorical_vars_na)
        mywriter.writerow(["Categorical with NA:", len(categorical_vars_na),
                           len(categorical_vars_na)/mydf.shape[1],
                           str(categorical_vars_na)])
        mywriter.writerow(["Text:", len(text_vars), len(text_vars)/mydf.shape[1],
                           str(text_vars)])
        text_vars_na = set(text_vars).intersection(features_na)
        text_vars_na = list(text_vars_na)
        mywriter.writerow(["Text with NA:", len(text_vars_na), len(text_vars_na)/mydf.shape[1],
                           str(text_vars_na)])
        number_cols_na = set(number_cols).intersection(features_na)
        number_cols_na = list(number_cols_na)
        mywriter.writerow(["Numerical:", len(number_cols), len(number_cols)/mydf.shape[1],
                           str(number_cols)])
        mywriter.writerow(["Numerical with NA:", len(number_cols_na), len(number_cols_na)/mydf.shape[1],
                           str(number_cols_na)])
        total = len(number_cols) + len(categorical_vars) + len(text_vars)
        mywriter.writerow(["TOTAL:", total])

