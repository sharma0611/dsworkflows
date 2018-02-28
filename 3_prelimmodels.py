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
from config.config import export_dir, random_variables, prelim_step, prelim_overwrite
from common.logger import start_printer, end_printer, join_pdfs
from common.utils import ensure_folder, save_obj, load_obj
from common.graphing import bargraph_from_db, feature_importance_bargraphs, figures_to_pdf

import pandas as pd
from time import time
from numpy import array
import os

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
prelim_dir = ds.make_custom_dir(prelim_step, prelim_overwrite)
start_printer(prelim_dir, fname)

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

categorical_bool = ds.get_categorical_bool()

for y in arr_y:
    #set up test train arrays for y
    test_y_arr, train_y_arr = ds.get_test_train_var_arrays(y)

    print("Prelim FS for " + y)
    tloop1 = time()

    ### PRELIM RF MODEL 
    mm.create_rf_model(ds, train_X_arr, train_y_arr, y, X, prelim_step)

    ### PRELIM GBR MODEL 
    mm.create_gbr_model(ds, train_X_arr, train_y_arr, y, X, prelim_step)

    ### PRELIM LIGHTGBM MODEL
    eval_set = [(test_X_arr, test_y_arr)]
    mm.create_lgbm_model(ds, train_X_arr, train_y_arr, y, X, prelim_step, eval_set)

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
figures_to_pdf(r2_metrics_figs + mse_metrics_figs + imp_figs, prelim_dir + '/prelim_model_summary.pdf')

#calculate time
t2_main = time()
t = (t2_main-t1_main)/60
print("Step 3 complete in " + '{0:.2f}'.format(t) + "m.")

end_printer(prelim_dir, fname)
#merge the outputted report file with the model summary
file1 = prelim_dir + '/' + fname + '.pdf'
file2 = prelim_dir + '/prelim_model_summary.pdf'
outfile = prelim_dir + '/prelim_report.pdf'
join_pdfs(file1, file2, outfile)

#save dataset changes
ds.save()
mm.save_all_live_models()
