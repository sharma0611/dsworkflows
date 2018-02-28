#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cih745
"""

from common.dsm import Dataset_Manager
from common.mm import Model_Manager
from config.config import export_dir, dataset_chosen, round_chosen, fs_step, fsa_step, fsa_overwrite, fs_id
from common.logger import start_printer, end_printer, join_pdfs
from common.utils import ensure_folder, save_obj, load_obj
from common.graphing import (bargraph_from_db, feature_importance_bargraphs, figures_to_pdf, 
        bargraph_fsa, grab_new_ax_array)

import pandas as pd
from time import time
from numpy import array
import os
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy

#Instantiate Dataset Manager & load dataset object
dsm = Dataset_Manager(export_dir)
ds = dsm.load_dataset(dataset_chosen)
ds.load_df()

#Instantiate Model Manager
mm = Model_Manager(export_dir)

#start timing
t1_main = time()

#start logging
fname = "fs_analysis"
start_printer(export_dir, fname)

#load test/train in dataset object
ds.load_test_train_dfs()

#all features in dataframe
X = ds.get_X()
y = ds.get_y()

#setting up round chosen
if not round_chosen:
    #find the last round
    modeldf = mm.grab_all_models_metadata_df()
    modeldf_subset = modeldf.query("round > 0 & dataset_name == " + str(dataset_chosen))
    if modeldf_subset.empty:
        #step 5 has not been run yet; default to entire subset
        round_chosen = False
    else:
        rounds_nums = modeldf_subset["round"].values.tolist()
        rounds_nums = [int(x) for x in rounds_num]
        round_chosen = max(rounds_nums)

#load subset chosen
ds = Dataset.load_dataset(dataset_chosen)
ds.load_raw()

#getting import directory for step 5 results
fs_dir = ds.get_custom_dir(fs_step, fs_id)
#setting export directory

fsa_step = fsa_step + "_round" + str(round_chosen) #update fsa step id
if fs_id:
    fsa_step = fsa_setp + "_id" + str(fs_id)
fsa_dir = ds.make_custom_dir(fsa_step, fsa_overwrite)

#activate report logging
fname = 'feature_analysis_log'
start_printer(fsa_dir, fname)

print("Feature Analysis Report\n")

tmain1 = time()

#setting up important features
important_features = []
if round_chosen:
    #load the DB of fs selection results
    ordered_features_df = pd.read_csv(fs_dir + '/round_' + str(round_chosen) + '/ordered_features.csv')
    df_cols = ordered_features_df.columns.tolist()
    #grab the top features 
    important_features = important_features + grab_all_features(ordered_features_df)
else:
    important_features = important_features + X

if force_add_features:
    orig_len = len(important_features)
    print("Original features: " + str(orig_len))
    important_features = important_features + force_add_features
    new_len = len(important_features)
    print("Added features: " + str(new_len-orig_len))

print("{0} Features selected for analysis :".format(len(important_features)))
print(str(important_features))

test_X_arr, train_X_arr = ds.get_test_train_X_arrays(important_features)
test_y_arr, train_y_arr = ds.get_test_train_var_arrays(y)

#Begin Wrapper Model Analysis: Independent, Dependent, Marginal tests

topfour_dep = []
topfour_ind = []
topfour_imp = []
idp = True
paths_to_be_joined = []
if idp:
    # INDEPENDENT FEATURE PREDICTIVE POWER 
    # for each var make a model with just that var & test & add to DF
    independent_results = []
    for ftr in important_features:
        #take the X of one feature for test & train
        curr_test_X_arr, curr_train_X_arr = ds.get_test_train_X_arrays(ftr)

        #create the model
        eval_set = [(curr_test_X_arr, test_y_arr)]
        lgbm_model = mm.create_lgbm_model(ds, curr_train_X_arr, train_y_arr, y, [ftr], fsa_step, eval_set)
        lgbm_model.set_metadata_feature('var', ftr)

    #model metrics 
    mm.r2_test_live_models(ds)

    #Create dataframe of all model metadata
    independent_results_df = mm.grab_live_models_metadata_df()

    independent_results_df.sort_values('r2_test', ascending=False, inplace=True)
    topfour_ind = independent_results_df["var"][:4].values.tolist()
    ind_fig1 = bargraph_fsa(independent_results_df, 'r2_test', 'Independent Learner: univariate model')
    ind_fig2 = bargraph_fsa(independent_results_df, 'r2_train', 'Independent Learner: univariate model')
    ind_graph_path = fsa_dir + '/ind_graphs.pdf'
    figures_to_pdf([ind_fig1, ind_fig2], ind_graph_path)
    paths_to_be_joined.append(ind_graph_path)

#save all live models
mm.save_all_live_models()

#DEPENDENT FEATURE PREDICTIVE POWER
#create a master model that is fit on all variables
eval_set = [(test_X_arr, test_y_arr)]
master_lgbm_model = mm.create_lgbm_model(ds, train_X_arr, train_y_arr, y, important_features, fsa_step, eval_set)

#metrics
master_lgbm_model.r2_test(ds)

#insert confusion matrix here
main_model_fig_arr = master_lgbm_model.confusion_matrix(ds)

#graph importances & confusion matrix
imp_df = master_lgbm_model.get_importance_df()
annot = "# of Features: " + str(len(important_features))
if round_chosen:
    annot = annot + "\nRound Chosen: " + str(round_chosen)
imp_figs = feature_importance_bargraphs(imp_df, "", annot)
imp_df.to_csv(fsa_dir + '/main_importances.csv')
topfour_imp = imp_df["var"][:4].values.tolist()

main_model_graphs_path = fsanalysis_dir + '/main_graphs.pdf'
figures_to_pdf(main_model_fig_arr + imp_figs, main_model_graphs_path)
paths_to_be_joined.append(main_model_graphs_path)

#save all models
mm.save_all_live_models()

dep = True
if dep:
    dep_results = []
    #make models leaving one variable out each time 
    for ftr in important_features:
        curr_ftrs = deepcopy(important_features)
        curr_ftrs.remove(ftr)

        #take the X of one feature for test & train
        curr_test_X_arr, curr_train_X_arr = ds.get_test_train_X_arrays(curr_ftrs)

        #create the model
        eval_set = [(curr_test_X_arr, test_y_arr)]
        lgbm_model = mm.create_lgbm_model(ds, curr_train_X_arr, train_y_arr, y, curr_ftrs, fsa_step, eval_set)
        lgbm_model.set_metadata_feature('discarded_var', ftr)

        #predict & metrics
        lgbm_model.r2_test(ds)

        r2_test_diff = master_r2_test - lgbm_model.r2_test
        mse_test_diff = master_mse_test - lgbm_model.mse_test
        r2_train_diff = master_r2_train - lgbm_model.r2_train
        mse_train_diff = master_mse_train - lgbm_model.mse_train

        #append results to list
        results = {"mse_train_diff": mse_train_diff,
                   "r2_train_diff": r2_train_diff,
                   "mse_test_diff": mse_test_diff,
                   "r2_test_diff":r2_test_diff,
                   "discarded_var":ftr}

        dep_results.append(results)

    #create dependent predictive power dataframe
    dep_results_df = pd.DataFrame(dep_results)
    dep_results_df.sort_values('r2_test_diff', ascending=False, inplace=True)
    topfour_dep = dep_results_df['discarded_var'][:4].values.tolist()
    dep_fig1 = bargraph_fsa(dep_results_df, 'r2_test_diff', 'Dependent Learner: r2 loss from variable omit')
    dep_fig2 = bargraph_fsa(dep_results_df, 'r2_train_diff', 'Dependent Learner: r2 loss from variable omit')
    dep_graph_path = fsa_dir + '/dep_graphs.pdf'
    figures_to_pdf([dep_fig1, dep_fig2], dep_graph_path)
    paths_to_be_joined.append(dep_graph_path)

    mm.save_all_live_models()

marg = True
if marg:
    # MARGINAL FEATURE PREDICTIVE POWER (forward selection)
    #want to see how r2 increases as features as added
    my_vars = independent_results_df['var'].values.ravel()
    curr_ftrs = [] #features currently being trained on
    marg_results = []

    for ftr in my_vars:
        curr_ftrs.append(ftr)

        #take the X of one feature for test & train
        curr_test_X_arr, curr_train_X_arr = ds.get_test_train_X_arrays(curr_ftrs)

        #create the model
        eval_set = [(curr_test_X_arr, test_y_arr)]
        lgbm_model = mm.create_lgbm_model(ds, curr_train_X_arr, train_y_arr, y, curr_ftrs, fsa_step, eval_set)
        lgbm_model.set_metadata_feature('added_var', ftr)

    mm.r2_test_live_models(ds)

    #create marginal predictive power dataframe
    marg_results_df = mm.grab_live_models_metadata_df()
    marg_fig1 = bargraph_fsa(marg_results_df, 'test_r2', 'Marginal Learner: r2 as features are added', True)
    marg_fig2 = bargraph_fsa(marg_results_df, 'train_r2', 'Marginal Learner: r2 as features are added', True)
    marg_graph_path = fsa_dir + '/marg_graphs.pdf'
    figures_to_pdf([marg_fig1, marg_fig2], marg_graph_path)
    paths_to_be_joined.append(marg_graph_path)

print("finished marg models")

# Models complete.
# Begin variable analysis using various visualizations.
take = important_features + [y_label]
sel_df = test_df[take]

#Box Plot
#first sort by maximum values for boxplot
sorted_df = sel_df.ix[:, sel_df.max().sort_values(ascending=False).index]

#boxplot of each variable
num_cols = sorted_df.shape[1]
increment = 6
curr_start = 0
curr_end = increment
boxplot_figures = []

curr_gs, curr_fig, curr_ax_array = grab_new_ax_array()

while curr_end <= num_cols:
    if curr_ax_array == []:
        curr_gs.tight_layout(curr_fig, rect=[0.05,0.05,0.95,0.95], pad=0.5)
        boxplot_figures.append(curr_fig)
        curr_gs, curr_fig, curr_ax_array = grab_new_ax_array()
    curr_ax = curr_ax_array.pop(0)
    sns.boxplot(data=sorted_df.iloc[:,curr_start:curr_end], ax=curr_ax)
    curr_start += increment
    curr_end += increment

if curr_start != num_cols:
    if curr_ax_array == []:
        curr_gs.tight_layout(curr_fig, rect=[0.05,0.05,0.95,0.95], pad=0.5)
        boxplot_figures.append(curr_fig)
        curr_gs, curr_fig, curr_ax_array = grab_new_ax_array()
    curr_ax = curr_ax_array.pop(0)
    sns.boxplot(data=sorted_df.iloc[:,curr_start:num_cols], ax=curr_ax)
    curr_gs.tight_layout(curr_fig, rect=[0.05,0.05,0.95,0.95], pad=0.5)
    boxplot_figures.append(curr_fig)

boxplot_graph_path = fsa_dir + '/boxplot_graphs.pdf'
figures_to_pdf(boxplot_figures, boxplot_graph_path)
paths_to_be_joined.append(boxplot_graph_path)

print("finished boxplots")

#Pairwise relationships & univariate distribution (pairplot)
#do this only for top four in topfour_dep, topfour_ind, topfour_imp
pairwise_figs = []
if topfour_imp != []:
    g = sns.PairGrid(sel_df[topfour_imp])
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Pairwise Relations for Top 4 Important Features', fontsize=16)
    g.map_lower(plt.scatter, rasterized=True)
    g.map_diag(plt.hist)
    g.map_upper(sns.residplot)
    pairwise_fig = g.fig
    pairwise_figs.append(pairwise_fig)

if topfour_dep != []:
    g = sns.PairGrid(sel_df[topfour_dep])
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Pairwise Relations for Top 4 Dependent Features', fontsize=16)
    g.map_lower(plt.scatter, rasterized=True)
    g.map_diag(plt.hist)
    g.map_upper(sns.residplot)
    pairwise_fig = g.fig
    pairwise_figs.append(pairwise_fig)

if topfour_ind != []:
    g = sns.PairGrid(sel_df[topfour_ind])
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('Pairwise Relations for Top 4 Independent Features', fontsize=16)
    g.map_lower(plt.scatter, rasterized=True)
    g.map_diag(plt.hist)
    g.map_upper(sns.residplot)
    pairwise_fig = g.fig
    pairwise_figs.append(pairwise_fig)

if pairwise_figs != []:
    pairwise_graph_path = fsa_dir + '/pairwise_graphs.pdf'
    figures_to_pdf(pairwise_figs, pairwise_graph_path)
    paths_to_be_joined.append(pairwise_graph_path)

plt.close("all")

print("finished pairwise")

#Joint/Marginal Distributions with output variable (jointplot)
joint_figures = []

#downsample set for intensive training
if len(sel_df) > 50000:
    sel_df = sel_df.sample(50000)
    print("shape is now 50000 rows for KDE's & distributions")

for x in important_features:
    #curr_fig = plt.figure(figsize=(8, 8))
    #curr_ax = curr_fig.add_subplot()
    trim_data = sel_df[[y_label, x]]
    g = sns.JointGrid(x=x, y=y_label, data=trim_data)
    try:
        g.plot_joint(sns.kdeplot)
    except:
        g.plot_joint(plt.scatter, rasterized=True)
    g.plot_marginals(sns.distplot)
    fig_dist = plt.figure(figsize=(8, 5))
    ax_dist = fig_dist.add_subplot(111)
    ax_dist.set_title("Distribution of " + x)
    trim_data[x].hist(ax=ax_dist, rasterized=True)
    #g = sns.jointplot(x, y_label, data=trim_data, kind="kde")
    joint_figures.append(g.fig)
    joint_figures.append(fig_dist)

joint_graph_path = fsa_dir + '/joint_graphs.pdf'
figures_to_pdf(joint_figures, joint_graph_path)
paths_to_be_joined.append(joint_graph_path)

#shutdown logging
if write_reports:
    end_printer(fsa_dir, fname)
    #combine this file with the other graphs as well
    file1 = fsa_dir + '/' + fname + '.pdf'
    outfile = fsa_dir + '/fs_analysis_report.pdf'
    paths_to_be_joined = [file1] + paths_to_be_joined
    join_pdfs_list(paths_to_be_joined, outfile, True)

