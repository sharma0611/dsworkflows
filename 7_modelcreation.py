
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
from config.config import export_dir, random_variables, prelim_step, prelim_overwrite, hyperparam_step
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
fname = "hyperparams"
hyperparam_dir = ds.make_custom_dir(hyperparam_step, prelim_overwrite)
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

import pandas as pd
from numpy import array
import numpy as np

from common.config import (y, transforms, random_variables, category_cols, transform_chosen,
                           write_reports, throwaway, fs_dir, fsanalysis_dir, modeldb_path, target_features,
                           force_add_features, round_chosen, dataprune_dir, modelcreation_dir,
                           model_dict, full_subset_models, last_3_rounds_models, impute_to_value)
from common.speed import pk_dump, pk_load
from common.featureanalysis import full_analysis, ordered_subset, grab_all_features, var_importance_table
from common.modelanalysis import train_test_confusion_plot_full, grab_new_ax_array
from common.modeldb import modeldb_add, modeldb_delete, model_name_gen, load_modeldb
from common.logger import start_printer, end_printer, join_pdfs, join_pdfs_list
from common.modelanalysis import r2_mse_grab, model_metrics_sklearn, model_metrics_lgb
from common.graphutil import bargraph_fsa, figures_to_pdf, feature_importance_bargraphs
from common.datainteract import Dataset

from sklearn.model_selection import KFold
from sklearn.feature_selection import f_regression, mutual_info_regression, RFECV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from boruta import BorutaPy
from functools import reduce
from time import time
import sys
import os
import subprocess
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

#set the parameters for all gbr models that will be created

# Create a model for each one specified in the config model dict
end_printer(submodel_dir, fname)
#combine this file with the other graphs as well
reportfile = submodel_dir + '/' + fname + '.pdf'
outfile = submodel_dir + "/modelcreate_report_" + str(model_num) + ".pdf"
join_pdfs_list([reportfile, main_model_graphs_path], outfile, True)
