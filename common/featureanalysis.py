#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma
    sharma0611
****************************************

Module: Feature Analysis
Purpose: Hosts functions that prune the feature set using wrapper & filter analysis

"""

import os
from importlib import import_module, reload
import csv
from time import time
from functools import reduce

from sklearn.model_selection import KFold
from sklearn.feature_selection import f_regression, mutual_info_regression, RFECV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from boruta import BorutaPy
import pandas as pd
from scipy import stats
from scipy.stats import mstats
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
from numpy import array
import numpy.ma as ma
from math import log

from common.multivariateanalysis import var_importance_table
from common.graphing import feature_importance_bargraphs
from config.config import max_drop_dict, fs_step

def get_max_drop(num_ftrs, max_drop_dict):
    max_drop = False
    for tuple_pair, curr_max_drop in max_drop_dict.items():
        lower_limit = tuple_pair[0]
        upper_limit = tuple_pair[1]
        if lower_limit < num_ftrs <= upper_limit:
            max_drop = curr_max_drop
            break
    return max_drop

#function for random importance screening
def grab_top_features(ordered_ftrs, random_vars):
    final_features = []
    for ftr in ordered_ftrs:
        if ftr in random_vars:
            break
        final_features.append(ftr)
    return final_features

#function to get all features in columns of dataframe
#this is used for getting important features from the important_feature dataframe
def grab_all_features(thedf):
    #use chosen fs algos & their results to get all important features
    important_features = []
    for col in thedf.columns.tolist():
        ordered_features = thedf[col].values.tolist()
        important_features = important_features + ordered_features
    important_features = list(set(important_features))
    important_features = [x for x in important_features if not pd.isnull(x)]
    return important_features

#function used to select subset of another list, elements in order of first list
def ordered_subset(preserve_order_list, cut_list):
    d = {k:v for v,k in enumerate(preserve_order_list)}
    new = set(preserve_order_list).intersection(cut_list)
    new = list(new)
    new.sort(key=d.get)
    return new

#function to get FI from models df
def get_model_feature_importances(models_df):
    model_dict = load_all_models(models_df)
    all_imp_dfs = []
    for k, model in model_dict.items():
        importances = model.feature_importances_
        results = models_df.loc[(models_df['ModelNum'] == k)]
        if len(results)==1:
            row = results.iloc[0]
            X = row["FeaturesUsed"]
            X = eval(X)
            model_algo = row["ModelAlgo"]
            special_tag = row["SpecialTag"]
            imp_df = var_importance_table(importances, X, model_algo + '_' + special_tag)
            all_imp_dfs.append(imp_df)

    master_imp_df = reduce(lambda left,right: pd.merge(left,right,how='outer',on='var'), all_imp_dfs)
    return master_imp_df

def get_transform_fn(var):
    transform_mod = import_module('.transforms', package='common')
    transform_mod = reload(transform_mod)
    transform_func = getattr(transform_mod, 'transform_' + var)
    return transform_func

def get_inverse_transform_fn(var):
    transform_mod = import_module('.transforms', package='common')
    transform_mod = reload(transform_mod)
    transform_func = getattr(transform_mod, 'inverse_transform_' + var)
    return transform_func

def fill_mask(mask, vals, default):
    if mask:
        good_vals = sum(x == 0 for x in mask)
    else:
        return vals
    assert good_vals == len(vals) #ensure
    final = []
    i = 0
    for x in mask:
        if x == 0:
            y = vals[i]
            i = i + 1
        else:
            y = default
        final.append(y)
    return final

#function used to perform feature selection, needs many globals only used in featureselection
def feature_selection(output_dir, mm, ds, y, X, use_fs_algos=["all"], curr_round=False):
    if not use_fs_algos:
        use_fs_algos = ["all"]

    #setup test/train arrays
    test_X_arr, train_X_arr = ds.get_test_train_X_arrays(X)
    test_y_arr, train_y_arr = ds.get_test_train_var_arrays(y)
    categorical_bool = ds.get_categorical_bool(X)
    random_variables = ds.get_random_variables()

    print("Shape of Training Features: " + str(train_X_arr.shape))

    #FULL FEATURE SELECTION ALGOS
    algo_dfs = []
    #Mutual Information
    if "mutual_info" in use_fs_algos or "all" in use_fs_algos:
        print("Mutual Info Algo")
        t1 = time()
        mi = mutual_info_regression(train_X_arr, train_y_arr, discrete_features=categorical_bool)
        mi /= np.max(mi)
        mi_df = var_importance_table(mi, X, 'mutual_info')
        t2 = time()
        t = (t2-t1)/60
        print("Mutual Info completed in " + '{0:.2f}'.format(t) + "m.\n")
        algo_dfs.append(mi_df)

    #F_regression
    if "f_regress" in use_fs_algos or "all" in use_fs_algos:
        print("F regression")
        t1 = time()
        f_test, _ = f_regression(train_X_arr, train_y_arr)
        f_test /= np.max(f_test)
        f_df = var_importance_table(f_test, X, 'f_regression')
        t2 = time()
        t = (t2-t1)/60
        print("F regression completed in " + '{0:.2f}'.format(t) + "m.\n")
        algo_dfs.append(f_df)

    #Normal RF
    if "fs_rf" in use_fs_algos or "all" in use_fs_algos:
        model_name = fs_step + "_rf"
        rf_model = mm.create_rf_model(ds, train_X_arr, train_y_arr, y, X, model_name)
        imp_df = rf_model.get_importance_df()
        algo_dfs.append(imp_df)

    #Normal GBR
    if "fs_gbr" in use_fs_algos or "all" in use_fs_algos:
        model_name = fs_step + "_gbr"
        gbr_model = mm.create_gbr_model(ds, train_X_arr, train_y_arr, y, X, model_name)
        imp_df = gbr_model.get_importance_df()
        algo_dfs.append(imp_df)

    #Microsoft LightGBM
    if "fs_lgb" in use_fs_algos or "all" in use_fs_algos:
        model_name = fs_step + "_lgbm"
        eval_set = [(test_X_arr, test_y_arr)]
        lgbm_model = mm.create_lgbm_model(ds, train_X_arr, train_y_arr, y, X, model_name, eval_set)
        imp_df = lgbm_model.get_importance_df()
        algo_dfs.append(imp_df)

    rank_dfs = []
    #Boruta RF
    if "boruta_rf" in use_fs_algos or "all" in use_fs_algos:
        print("Boruta RF")
        t1 = time()
        rf = RandomForestRegressor(n_jobs=-1)
        fs_selector = BorutaPy(rf, n_estimators='auto', random_state=3142, max_iter=70)
        fs_selector.fit(train_X_arr, train_y_arr)
        scores = fs_selector.ranking_
        rf_df = var_importance_table(scores, X, 'boruta_rf')
        t2 = time()
        t = (t2-t1)/60
        print("Boruta RF completed in " + '{0:.2f}'.format(t) + "m.\n")
        rank_dfs.append(rf_df)

    #Boruta GBR
    if "boruta_gbr" in use_fs_algos or "all" in use_fs_algos:
        print("Boruta GBR")
        t1 = time()
        gbr = GradientBoostingRegressor()
        fs_selector = BorutaPy(gbr, n_estimators='auto', random_state=3142, max_iter=70)
        fs_selector.fit(train_X_arr, train_y_arr)
        scores = fs_selector.ranking_
        gbr_df = var_importance_table(scores, X, 'boruta_gbr')
        t2 = time()
        t = (t2-t1)/60
        print("Boruta GBR completed in " + '{0:.2f}'.format(t) + "m.\n")
        rank_dfs.append(gbr_df)

    #Recursive Feature Elimination with SVR
    if "recursive_svr" in use_fs_algos or "all" in use_fs_algos:
        print("Recursive SVR")
        t1 = time()
        svr = SVR(kernel="linear")
        rfecv = RFECV(estimator=svr, step=1, cv=KFold(3),
                      scoring='accuracy')
        rfecv.fit(train_X_arr, train_y_arr)

        #print("Optimal number of features : %d" % rfecv.n_features_)
        ## Plot number of features VS. cross-validation scores
        #plt.figure()
        #plt.xlabel("Number of features selected")
        #plt.ylabel("Cross validation score (nb of correct classifications)")
        #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        #plt.show()

        scores = rfecv.ranking_
        rsvc_df = var_importance_table(scores, X, 'recursive_svc')
        t2 = time()
        t = (t2-t1)/60
        print("Recursive SVR completed in " + '{0:.2f}'.format(t) + "m.\n")
        rank_dfs.append(rsvc_df)

    # set model metadata to include round # and 
    mm.set_metadata_feature_live_models("round", curr_round)

    temp_df = pd.DataFrame(X, columns=['var'])
    if not algo_dfs:
        algo_dfs = [temp_df]
    if not rank_dfs:
        rank_dfs = [temp_df]

    if algo_dfs and rank_dfs:
        #Join all dfs
        #convert below code into a function (for memory reasons)
        #Get all recorded importances
        df_final_imp = reduce(lambda left,right: pd.merge(left,right,how='outer',on='var'), algo_dfs)
        df_final_imp['sum'] = df_final_imp.sum(axis=1)
        new_rank_dfs = rank_dfs + [df_final_imp]
        df_final = reduce(lambda left,right: pd.merge(left,right,how='outer',on='var'), new_rank_dfs)
        #df_final.to_csv("./temp1.csv")
        df_final.sort_values('sum', ascending=False, inplace=True)
        sum_ordered_vars = df_final['var'].values.tolist() #keep this list to reorder garbage variables
        #Output dfs to fsdb.csv
        df_final.to_csv(output_dir + '/feature_importances.csv', index=False)

        #Get ordered lists of features 
        df_final_vars = df_final.copy(deep=True)
        final_rank_df = reduce(lambda left,right: pd.merge(left,right,how='outer',on='var'), rank_dfs)
        rank_cols = list(set(final_rank_df.columns.tolist()) - set(['var']))
        imp_cols = list(set(df_final_vars.columns.tolist()) - set(['var']) - set(rank_cols))
        for col in rank_cols:
            temp_df = df_final[['var', col]]
            temp_series = temp_df.sort_values(col)['var'].values.tolist()
            df_final_vars.loc[:,col] = temp_series
        for col in imp_cols:
            temp_df = df_final[['var', col]]
            temp_series = temp_df.sort_values(col, ascending=False)['var'].values.tolist()
            df_final_vars.loc[:,col] = temp_series
        df_final_vars = df_final_vars[rank_cols + imp_cols]
        df_final_vars.to_csv(output_dir + '/ordered_features.csv', index=False)

        #Get ordered list of selected features by random importance screening
        sel_df = pd.DataFrame()
        for col in rank_cols:
            temp_rank_df = df_final[['var', col]]
            temp_rank_df = temp_rank_df.query(col + " == 1 | " + col + " == 2")
            temp_rank_df.sort_values(col, ascending=False)
            ordered_ftrs = temp_rank_df['var'].values.tolist()
            temp_results = pd.DataFrame(ordered_ftrs, columns=[col])
            sel_df = pd.concat([sel_df, temp_results], axis=1)
        for col in imp_cols:
            temp_ftr_list = df_final_vars[col].values.tolist()
            ordered_ftrs = grab_top_features(temp_ftr_list, random_variables)
            temp_results = pd.DataFrame(ordered_ftrs, columns=[col])
            sel_df = pd.concat([sel_df, temp_results], axis=1)
        sel_df.to_csv(output_dir + '/selected_features.csv', index=False)

        #grab all important features from selected features df
        fi_cols = [x + "_importance" for x in use_fs_algos]

        if fi_cols:
            sel_cols = sel_df.columns.tolist()
            fi_cols = ordered_subset(sel_cols, fi_cols)
            sel_df = sel_df[fi_cols]
        important_features = grab_all_features(sel_df)

        #adding back random variables
        important_features = list(set(important_features + random_variables))

        if max_drop_dict:
            num_ftrs = len(sum_ordered_vars)
            max_drop = get_max_drop(num_ftrs, max_drop_dict)
            if max_drop:
                #ensure to count only non random variables in garbage
                garbaged = set(sum_ordered_vars) - set(important_features)
                garbaged = ordered_subset(sum_ordered_vars, garbaged)
                curr_drop = len(garbaged)
                if curr_drop > max_drop:
                    print("Restricting the # of variables dropped from {0} to {1}".format(curr_drop, max_drop))
                    new_garbage = garbaged[-max_drop:]
                    important_features = list(set(sum_ordered_vars) - set(new_garbage)) #this will add back randoms
                    important_features = ordered_subset(sum_ordered_vars, important_features)

        #generate importance graphs
        imp_cols_plus = imp_cols + ['var']
        imp_df = df_final[imp_cols_plus]
        annot = "# of Features: " + str(len(X))
        if curr_round:
            annot = annot + "\nRound #: " + str(curr_round)
        imp_figs = feature_importance_bargraphs(imp_df, "", annot)

        return important_features, imp_figs

