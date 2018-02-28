#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma
    sharma0611
****************************************

Script: Feature Selection
Purpose: To prune features to subset of X, while recording various metrics & providing analysis

"""

from common.dsm import Dataset_Manager
from common.mm import Model_Manager
from config.config import category_cols, export_dir, target_features, fs_algos_dict, max_rounds, fs_overwrite, fs_step
from common.logger import start_printer, end_printer, join_pdfs, join_pdfs_list
from common.utils import ensure_folder, save_obj, load_obj
from common.graphing import bargraph_from_db, feature_importance_bargraphs, figures_to_pdf, linegraph_from_db
from common.featureanalysis import feature_selection, ordered_subset
from matplotlib import rcParams
import matplotlib.pyplot as plt

import pandas as pd
from time import time
from numpy import array
import os

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb

init_round = False

# function designed to get the desired algorithms from the algo dictionary based on the # of features
def get_algos(num_ftrs, algo_dict):
    curr_fs_algos = False
    for tuple_pair, fs_algos, in algo_dict.items():
        lower_limit = tuple_pair[0]
        upper_limit = tuple_pair[1]
        if lower_limit < num_ftrs <= upper_limit:
            curr_fs_algos = fs_algos
            break
    return curr_fs_algos

#function designed to reassign
def reassign_algos(num_ftrs, algo_dict, new_algos):
    for tuple_pair, fs_algos, in algo_dict.items():
        lower_limit = tuple_pair[0]
        upper_limit = tuple_pair[1]
        if lower_limit < num_ftrs <= upper_limit:
            break
    algo_dict[(lower_limit, upper_limit)] = new_algos
    return algo_dict

rcParams.update({'figure.max_open_warning': False})

#Instantiate Dataset Manager & load dataset object
dsm = Dataset_Manager(export_dir)
ds = dsm.load_dataset("prune")
ds.load_df()

#Instantiate Model Manager
mm = Model_Manager(export_dir)

#init fs dir
fs_dir = ds.make_custom_dir(fs_step, fs_overwrite)
#start timing
t1_main = time()

#load test/train in dataset object
ds.load_test_train_dfs()

#all features in dataframe
X = ds.get_X()
y = ds.get_y()
random_variables = ds.get_random_variables()

if init_round:
    #start logging
    fname = "fs_init_log"
    start_printer(fs_dir, fname)

    init_dir = fs_dir + "/init"
    ensure_folder(init_dir)

    print("INITIAL FS for " + y)
    main_time1 = time()

    important_features, imp_figs = feature_selection(init_dir, mm, ds, y, X)

    main_time2 = time()
    t = (main_time2-main_time1)/60
    print("FS round completed in " + '{0:.2f}'.format(t) + "m.\n")
    print("# of Features Filtered: {0} / {1}".format(len(important_features),len(X)))
    print("All Algos Used. "+str(len(random_variables))+" random variables included.")
    print("Features Selected:")
    print(str(important_features))

    #shutdown logging
    end_printer(fs_dir, fname)
    #save graphical analysis & join with log pdf
    log_file = fs_dir + '/' + fname + '.pdf'
    graph_file = init_dir + '/importance_graphs.pdf'
    outfile = init_dir + '/fs_init_report.pdf'
    figures_to_pdf(imp_figs, graph_file)
    join_pdfs(log_file, graph_file, outfile)

else:
    curr_features = X
    curr_round = 1
    outfile_list = []
    last_round = False
    cut_round = False
    curr_fs_algos_dict = fs_algos_dict
    curr_fs_algos = False
    while curr_round <= max_rounds:
        #the behaviour of this while loop is as follows:
        #   rounds will continue until specified max_rounds is achieved
        #   the algorithms to use is sourced from the # of features + specified algos for that #
        #   curr_fs_algos keeps these algos in a list
        #   if for two consecutive rounds, features are not shortened, & the len(curr_fs_algos - ["all", "sum"]) is > 1:
        #       then, we cut down the curr_algos list by removing the first algo in the list in the dict
        #           (thus, for consecutive calls to the dict, if ftr_num stays in same range, edited list is
        #           fetched)
        #   if for two consecutive rounds, features are not shortened, & the curr algo list == 1:
        #       then, we break out of the loop
        #   if we acheive current_features < target_features at start of loop:
        #       then, finish the loop and break (so we get r2 of chosen round vars)

        #setup export dir
        curr_round_dir = fs_dir + "/round_" + str(curr_round)
        ensure_folder(curr_round_dir)

        #activate report logging
        fname = 'fs_report_' + str(curr_round)
        start_printer(curr_round_dir, fname)

        #select the algos to use here: given the # of features we select from the config
        num_ftrs = len(curr_features)

        #make it the last round if we surpass or reach target features
        if num_ftrs <= target_features or curr_round == max_rounds:
            last_round = True

        curr_fs_algos = get_algos(num_ftrs, curr_fs_algos_dict)

        if not curr_fs_algos:
            print("There are {0} # of features in round # {1}.".format(num_ftrs, curr_round))
            print("Please ensure your config ranges for fs_algos_dict is correct.")
            exit(0)
        else:
            if len(curr_fs_algos) == 1:
                if "all" in curr_fs_algos or "sum" in curr_fs_algos:
                    print("All or Sum left as only last values in algos.")
                    break

        print("ROUND # " + str(curr_round) + " FS for " + y)
        main_time1 = time()
        prev_ftrs = curr_features
        len_prev_ftrs = len(prev_ftrs)
        curr_features, imp_figs = feature_selection(curr_round_dir, mm, ds, y, curr_features, curr_fs_algos, curr_round)
        len_curr_ftrs = len(curr_features)

        #reorder the features by original order
        curr_features = ordered_subset(X, curr_features)

        main_time2 = time()
        t = (main_time2-main_time1)/60
        print("FS round "+str(curr_round)+" completed in " + '{0:.2f}'.format(t) + "m.\n")
        print("# of Features Filtered: {0} / {1}".format(len_curr_ftrs, len_prev_ftrs))
        print("Target # of Features: " + str(target_features))
        print(str(len(random_variables))+" random variables included. Algos Used:")
        print(str(curr_fs_algos))
        print("Features Selected:")
        print(str(curr_features))
        print("Features Discarded:")
        discard = list(set(prev_ftrs) - set(curr_features))
        print(str(discard))

        if last_round:
            print("\nThis last round was performed to check metrics of outputted variables from previous.")

        #shutdown logging
        end_printer(curr_round_dir, fname)
        #save graphical analysis & join with log pdf
        log_file = curr_round_dir + '/' + fname + '.pdf'
        graph_file = curr_round_dir + '/importance_graphs.pdf'
        outfile = curr_round_dir + '/fs_round_'+str(curr_round)+'_report.pdf'
        figures_to_pdf(imp_figs, graph_file)
        join_pdfs(log_file, graph_file, outfile)
        outfile_list.append(outfile)

        #setup next round 
        curr_round += 1

        #if we have reached last round by surpassing target features; break
        if last_round:
            break

        #break if we do not cut down the list of features
        if len_prev_ftrs == len_curr_ftrs:
            if len(set(curr_fs_algos) - set(["all", "sum"])) > 1:
                curr_fs_algos = curr_fs_algos[1:]
                curr_fs_algos_dict = reassign_algos(len_curr_ftrs, curr_fs_algos_dict, curr_fs_algos)
            else:
                break

    #take list of metadata's, produce new df, make graphs, push to modeldb
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
    feature_reduction_fig = linegraph_from_db(metadata, "round", "num_features")

    #export graphical results
    main_graphs_path = fs_dir + '/r2_graphs.pdf'
    figures_to_pdf(r2_metrics_figs + mse_metrics_figs + [feature_reduction_fig], main_graphs_path)

    #take list of outfiles; join all, then combine with your analysis
    master_outfile = fs_dir + '/master_outfile.pdf'
    join_pdfs_list(outfile_list, master_outfile, False)

    master_report_path = fs_dir + '/fs_rounds_report.pdf'
    join_pdfs(main_graphs_path, master_outfile, master_report_path)

    #close all open figures and live models
    mm.save_all_live_models()
    plt.close("all")

