
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma
    sharma0611
****************************************

"""
from common.dsm import Dataset_Manager
from common.mm import Model_Manager
from config.config import export_dir, random_variables, hyperparam_overwrite, hyperparam_step, final_features
from common.logger import start_printer, end_printer, join_pdfs, join_pdfs_list
from common.utils import ensure_folder, save_obj, load_obj
from common.graphing import bargraph_from_db, feature_importance_bargraphs, figures_to_pdf

import pandas as pd
from time import time
from numpy import array
import os

#Instantiate Dataset Manager & load dataset object
dsm = Dataset_Manager(export_dir)
ds = dsm.load_dataset("prune")
ds.load_df()

#Instantiate Model Manager
mm = Model_Manager(export_dir)

#start timing
t1_main = time()

#start logging
fname = "hyperparams"
hyperparam_dir = ds.make_custom_dir(hyperparam_step, hyperparam_overwrite)
start_printer(hyperparam_dir, fname)

#load test/train in dataset object
ds.load_test_train_dfs()

#all features in dataframe
X = final_features
y = ds.get_y()

#create numpy arrays for train/test X
test_X_arr, train_X_arr = ds.get_test_train_X_arrays(X)
test_y_arr, train_y_arr = ds.get_test_train_var_arrays(y)


fig_arr, best_params = mm.regression_gridsearch_lgbm(ds, X, train_X_arr, train_y_arr)

for param_set in best_params:
    lgbm_model = mm.create_lgbm_model(ds, train_X_arr, train_y_arr, y, X, hyperparam_step, None, update_params=param_set)
    lgbm_model.r2_test(ds)
    fig_arr_matrix = lgbm_model.confusion_matrix(ds, full_dist=False)
    fig_arr = fig_arr + fig_arr_matrix

mm.save_all_live_models()

main_model_graphs_path = hyperparam_dir + "/model_graphs.pdf"
figures_to_pdf(fig_arr, main_model_graphs_path)

end_printer(hyperparam_dir, fname)
#combine this file with the other graphs as well
reportfile = hyperparam_dir + '/' + fname + '.pdf'
outfile = hyperparam_dir + "/modelcreate_report.pdf"
join_pdfs_list([reportfile, main_model_graphs_path], outfile, True)
