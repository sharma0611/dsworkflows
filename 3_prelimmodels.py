#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma
    sharma0611
****************************************

Script: Preliminary Models
Purpose: Create preliminary models to evaluate performance of various transformations on target variable & to
         dispose of garbage variables to make feature space smaller

"""

from common.dsm import Dataset_Manager
from common.mm import Model_Manager
from config.config import export_dir, random_variables
from common.logger import start_printer, end_printer, join_pdfs
from common.utils import ensure_folder, save_obj, load_obj
from common.graphing import bargraph_from_db, feature_importance_bargraphs, figures_to_pdf

import pandas as pd
from time import time
from numpy import array
import os

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from common.modelanalysis import r2_model
import lightgbm as lgb

#Instantiate Dataset Manager & load dataset object
dsm = Dataset_Manager(export_dir)
ds = dsm.load_dataset("impute")
ds.load_df()

#Instantiate Model Manager
mm = Model_Manager(export_dir)

#start timing
t1_main = time()

#start logging
fname = "prelim_log"
start_printer(export_dir, fname)

#load test/train in dataset object
ds.load_test_train_dfs()

#all output columns in dataframe
arr_y = ds.get_all_y()

#all features in dataframe
X = ds.get_X()

#create numpy arrays for train/test X
test_X_arr, train_X_arr = ds.get_test_train_X_arrays()
print("Shape of Train Features: " + str(test_X_arr.shape))
print("Shape of Test Features: " + str(train_X_arr.shape))

categorical_bool = [True if x in ds.category_cols else False for x in X]

#init lists to store model dicts & models
model_dict_list = []
model_list = {}

models = []

step_tag = "prelim"

for y in arr_y:

    #set up test train arrays for y
    test_y_arr, train_y_arr = ds.get_test_train_var_arrays(y)

    varimportance_list = [] # stores all the importance DFs from each model

    print("Prelim FS for " + y)
    tloop1 = time()

    ### PRELIM RF MODEL 
    print("Prelim RF Model for " + y)
    t1 = time()
    params = {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 4,
              'n_jobs':-1, 'random_state': 120}
    rf_model_object = RandomForestRegressor(**params)
    rf_model_object.fit(train_X_arr, train_y_arr)

    #save model in Model object
    model_algo = 'rf'
    model_name = model_algo + "_" + step_tag
    rf_model = mm.create_model(model_name, rf_model_object, y, X, model_algo, params, step_tag)
    rf_model.set_metadata_with_dataset(ds)

    #get ordered list of important features
    importances = rf_model_object.feature_importances_
    rf_model.set_importance_df(importances)

    #calculate time
    t2 = time()
    t = (t2-t1)/60
    rf_model.set_training_time(t) #record time it took
    print("RF complete in " + '{0:.2f}'.format(t) + "m.")

    ### PRELIM GBR MODEL 
    print("Prelim GBR Model for " + y)
    t1 = time()
    params = {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 4,
              'learning_rate': 0.1, 'loss': 'ls', 'random_state': 912}
    gbr_model_object = GradientBoostingRegressor(**params)
    gbr_model_object.fit(train_X_arr, train_y_arr)

    #save this model for later analysis
    model_algo = 'gbr'
    model_name = model_algo + "_" + step_tag
    gbr_model = mm.create_model(model_name, gbr_model_object, y, X, model_algo, params, step_tag)
    gbr_model.set_metadata_with_dataset(ds)

    #output importances
    importances = gbr_model_object.feature_importances_
    gbr_model.set_importance_df(importances)

    #calculate time
    t2 = time()
    t = (t2-t1)/60
    gbr_model.set_training_time(t)
    print("GBR complete in " + '{0:.2f}'.format(t) + "m.")

    ### PRELIM LIGHTGBM MODEL
    print("Prelim Microsoft LightGBM Model for " + y)
    t1 = time()
    params = {
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 300,
        'seed': 1231,
    }

    gbm_model_object = lgb.LGBMRegressor(**params)
    gbm_model_object.fit(train_X_arr, train_y_arr,
            eval_metric='l2',
            eval_set=[(test_X_arr, test_y_arr)],
            early_stopping_rounds=5,
            feature_name=X,
            verbose=False)

    #save this model for later analysis
    model_algo = 'lgbm'
    model_name = model_algo + "_" + step_tag
    gbm_model = mm.create_model(model_name, gbm_model_object, y, X, model_algo, params, step_tag)
    gbm_model.set_metadata_with_dataset(ds)

    #output importances
    importances = gbm_model_object.feature_importances_
    gbm_model.set_importance_df(importances)

    #calculate time
    t2 = time()
    t = (t2-t1)/60
    gbm_model.set_training_time(t)
    print("LightGBM complete in " + '{0:.2f}'.format(t) + "m.")

    # FINDING GARBAGE VARIABLES 
    percent_garbage = 1/3
    final_garbage_features = mm.garbage_features(y, percent_garbage)

    print("For the transformation: {0}, {1} out of {2} features are garbage.".format(y,
                                                                                     len(final_garbage_features),
                                                                                     len(X)))
    print("{0} features kept.".format(str(len(X) - len(final_garbage_features))))
    print(str(len(random_variables)) + " random variables are part of kept set.")
    print("Garbage:")
    print(str(final_garbage_features))

    #save garbage features to dataset
    ds.set_garbage_ftrs(y, final_garbage_features)

#model metrics 
mm.r2_test_live_models(ds)

#Create dataframe of all model metadata
metadata = mm.grab_live_models_metadata_df()

#create graph df for graphing results
mini_modeldb_r2 = metadata.drop(["mse_train", "mse_test"], axis=1)
mini_modeldb_mse = metadata.drop(["r2_train", "r2_test"], axis=1)
graph_df_r2 = pd.melt(mini_modeldb_r2, id_vars=["y", "model_algo"], value_vars=["r2_train", "r2_test"], value_name="value")
graph_df_mse = pd.melt(mini_modeldb_mse, id_vars=["y", "model_algo"], value_vars=["mse_train", "mse_test"], value_name="value")
graph_df_r2.rename(columns={'variable': 'metric'}, inplace=True)
graph_df_mse.rename(columns={'variable': 'metric'}, inplace=True)

#get graphical analysis 
r2_metrics_figs = bargraph_from_db(graph_df_r2, x='y', y='value',seperate_by='model_algo', hue="metric", y_title="Value of Metric")
mse_metrics_figs = bargraph_from_db(graph_df_mse, x='y', y='value',seperate_by='model_algo', hue="metric", y_title="Value of Metric")

#get graphical results for importances
imp_figs = []
for y in arr_y:
    master_imp_df = mm.grab_features_importance_df(y)
    curr_imp_figs = feature_importance_bargraphs(master_imp_df, y, "# of Features: " + str(len(X)))
    imp_figs = imp_figs + curr_imp_figs

#export graphical results
figures_to_pdf(r2_metrics_figs + mse_metrics_figs + imp_figs, export_dir + '/prelim_model_summary.pdf')

#calculate time
t2_main = time()
t = (t2_main-t1_main)/60
print("Step 3 complete in " + '{0:.2f}'.format(t) + "m.")

end_printer(export_dir, fname)
#merge the outputted report file with the model summary
file1 = export_dir + '/' + fname + '.pdf'
file2 = export_dir + '/prelim_model_summary.pdf'
outfile = export_dir + '/prelim_report.pdf'
join_pdfs(file1, file2, outfile)

#save dataset changes
ds.save()
