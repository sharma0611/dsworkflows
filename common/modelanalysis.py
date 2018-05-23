#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma (CIH745)
    Data Science Co-op
    Summer 2017
****************************************

Module: Model Analysis
Purpose: Hosts the functions that interact with sklearn-interface models to provide analysis on statistical
         metrics

"""
from common.utils import load_obj, save_obj
import pandas as pd
from numpy import array, random
import numpy as np
import os
import re
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix
from importlib import import_module
import importlib.util
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from common.univariateanalysis import apply_spec_to_df
from common.transforms import inverse_dictionary

#metrics
def insert_random_var(seed, new_var, dataframe):
    """Function to insert random variable into Pandas DataFrame.
    """
    random.seed(seed)
    dataframe[new_var] = random.random_sample(dataframe.shape[0])

#function used to select subset of another list, elements in order of first list
def ordered_subset(preserve_order_list, cut_list):
    d = {k:v for v,k in enumerate(preserve_order_list)}
    new = set(preserve_order_list).intersection(cut_list)
    new = list(new)
    new.sort(key=d.get)
    return new


def mse_r2_perf_output(actual_y, pred_y, title, fig=None, ax=None):

    if fig is None and ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    mse = mean_squared_error(actual_y, pred_y)
    print(title + ":")
    print("MSE: %.4f" % mse)

    r2 = r2_score(actual_y, pred_y)
    print("r2: %.4f" % r2)

    ax.scatter(actual_y, pred_y, rasterized=True)
    ax.plot([actual_y.min(), actual_y.max()], [actual_y.min(), actual_y.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    return mse, r2


def r2_mse_grab(test_y, test_y_pred):
    mse_test = mean_squared_error(test_y, test_y_pred)
    r2_test = r2_score(test_y, test_y_pred)

    return r2_test, mse_test


def get_accuracy(actual_bands, predict_bands, interval):
    len_a = len(actual_bands)
    len_b = len(predict_bands)
    if len_a == len_b:
        accuracy = len([1 for x,y in zip(actual_bands, predict_bands) if x - interval <= y <= x + interval])/len_a
        return accuracy
    else:
        print("Different shapes between actual & predict of {0} and {1}".format(len_a, len_b))
        exit(0)

def cf_twentile_matrix(actual_twentile, predicted_twentile, title, normalize=True, fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    cnf_matrix = confusion_matrix(actual_twentile, predicted_twentile)
    if normalize==True:
        cnf_matrix = cnf_matrix.astype('float') / np.amax(cnf_matrix)


    sns.heatmap(cnf_matrix, linewidths=.5, ax=ax)
    for i in range(4):
        accuracy = get_accuracy(actual_twentile, predicted_twentile, i)
        ax.text(s='{:.2%}'.format(accuracy) + " (x:x+-{0})".format(i), transform=ax.transAxes,
                x=0.75,y=0.95-i*0.07, fontsize=12)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return fig, ax

def tile_accuracy_hist(actual_tiles, predicted_tiles, tag="", predict_bands=True):
    cnf_matrix = confusion_matrix(actual_tiles, predicted_tiles)
    num_tiles = len(cnf_matrix)
    all_hist_figs = []
    all_tiles = [x + 1 for x in range(num_tiles)]
    for i in range(num_tiles):
        if predict_bands:
            row = cnf_matrix[:,i]
            tiletype = "Predict"
        else:
            row = cnf_matrix[i]
            tiletype = "Actual"
        fig = plt.figure(figsize=(18, 6))
        ax = fig.add_subplot(111)
        sns.set_style("darkgrid")
        ax.bar(left=all_tiles, height=row, tick_label=all_tiles, width=1)
        ax.set_title(tag + " Distribution for "+tiletype+" Tile " + str(i+1))
        ax.set_ylabel("# of predictions")
        ax.set_xlabel("Predicted Tile")
        ax.set_xlim(0, num_tiles+1)

        #annotate each bar with percent of predictions
        total_predictions = sum(row)
        for p in ax.patches:
            ax.annotate('{:.2%}'.format(p.get_height()/total_predictions), (p.get_x() + (0 * p.get_width()),
                                                                            p.get_height() * 1.005), fontsize=10)
        all_hist_figs.append(fig)

    return all_hist_figs

def mse_r2_graph(actual_y, pred_y, title, fig=None, ax=None):

    if fig is None and ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    mse = mean_squared_error(actual_y, pred_y)
    print("MSE: %.4f" % mse)

    r2 = r2_score(actual_y, pred_y)
    print("r2: %.4f" % r2)

    ax.scatter(actual_y, pred_y)
    ax.plot([actual_y.min(), actual_y.max()], [actual_y.min(), actual_y.max()], 'k--', lw=4)
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    return mse, r2

def predict_metrics(actual_y, predict_y, actual_bands, predict_bands, tag, fig, ax_array):
    #plot scatter plot of actual & predicted
    mse, r2 = mse_r2_perf_output(actual_y, predict_y, tag, fig, ax_array[0])

    #create confusion matrix
    cf_twentile_matrix(actual_bands, predict_bands, tag, True, fig, ax_array[1])

    mean_camaro = pd.DataFrame({'actual_y': actual_y,
                                'actual_twentiles': actual_bands,
                                'predicted_y': predict_y,
                                'predicted_twentiles': predict_bands
                                })
    actual = mean_camaro[['actual_twentiles', 'actual_y']].groupby(['actual_twentiles']).mean()
    pred = mean_camaro[['predicted_twentiles', 'predicted_y']].groupby(['predicted_twentiles']).mean()

    ax = actual.plot(ax=ax_array[2],rasterized=True)
    ax1 = pred.plot(ax=ax,rasterized=True)
    ax1.set_title(tag)
    ax1.set_ylabel("Mean predicted value for each tile")
    ax1.set_xlabel("Tile")
    return mse, r2

def r2_and_mse(clf, test_y, test_X, tag, estimators, fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    test_r2 = np.zeros((estimators,), dtype=np.float64)
    test_mse = np.zeros((estimators,), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(test_X)):
        test_r2[i] = r2_score(test_y , y_pred)
        test_mse[i] = mean_squared_error(test_y, y_pred)



    ax.plot(np.arange(estimators) + 1, test_r2, 'b-',
             label='R2', rasterized=True)
    ax.plot(np.arange(estimators) + 1, test_mse, 'r-',
             label='MSE', rasterized=True)
    ax.legend(loc='upper right')
    ax.set_xlabel('Boosting Iterations')
    ax.set_ylabel('R2 / MSE')
    ax.set_title(tag)
    return fig,ax

def deviance(clf, test_y, test_X, tag, estimators, fig=None, ax=None):
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.get_figure()
    elif ax is None:
        ax = fig.gca()

    test_score = np.zeros((estimators,), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(test_X)):
        test_score[i] = clf.loss_(test_y, y_pred)

    ax.set_title('Deviance')
    ax.plot(np.arange(estimators) + 1, clf.train_score_, 'b-',
             label='Training Set Deviance', rasterized=True)
    ax.plot(np.arange(estimators) + 1, test_score, 'r-',
             label='Test Set Deviance', rasterized=True)
    ax.legend(loc='upper right')
    ax.set_xlabel('Boosting Iterations')
    ax.set_ylabel('Deviance')
    return fig, ax

import lightgbm as lgb
def model_metrics_lgb(clf):
    fig3 = plt.figure(figsize=(8, 11))
    gs3 = gridspec.GridSpec(2, 1)
    ax7 = fig3.add_subplot(gs3[0])
    ax8 = fig3.add_subplot(gs3[1])
    lgb.plot_metric(clf, metric="l2", ax=ax7, title="l2 during Training")
    lgb.plot_metric(clf, metric="huber", ax=ax8, title="Huber Loss during Training")
    gs3.tight_layout(fig3, rect=[0.05,0.05,0.95,0.95], pad=0.5)
    return [fig3]

def model_metrics_sklearn(clf, estimators, actual_test_y, test_X, tag):
    fig3 = plt.figure(figsize=(8, 11))
    gs3 = gridspec.GridSpec(2, 1)
    ax7 = fig3.add_subplot(gs3[0])
    ax8 = fig3.add_subplot(gs3[1])
    r2_and_mse(clf, actual_test_y, test_X, tag, estimators, fig3, ax7)
    deviance(clf, actual_test_y, test_X, tag, estimators, fig3, ax8)
    gs3.tight_layout(fig3, rect=[0.05,0.05,0.95,0.95], pad=0.5)
    return [fig3]

#getting an r2 from model
def r2_model():
    #all output columns in dataframe
    y_pattern = ".*_" + str(y) + '$'
    r = re.compile(y_pattern)
    arr_y = filter(r.match, train_df.columns.tolist())

    if features_to_use:
        if type(features_to_use) == str:
            features_to_use = eval(features_to_use)
            X = ordered_subset(X, features_to_use)

    train_X_df = train_df[X]
    train_X_arr = array(train_X_df)
    test_X_df = test_df[X]
    test_X_arr = array(test_X_df)

    #getting all y arrays
    y_label = transform + "_" + y
    train_y_series = train_df[y_label]
    train_y_arr = array(train_y_series)
    test_y_series = test_df[y_label]
    test_y_arr = array(test_y_series)

    #predict
    test_y_pred = model.predict(test_X_arr)
    train_y_pred = model.predict(train_X_arr)

    #get metrics
    r2_test, mse_test = r2_mse_grab(test_y_arr, test_y_pred)
    r2_train, mse_train = r2_mse_grab(train_y_arr, train_y_pred)

    #normalised mse by dividing by range of test_y_arr & train_y_arr respectively
    #nmse_test = mse_test / abs(max(test_y_arr) - min(test_y_arr))
    #nmse_train = mse_train / abs(max(train_y_arr) - min(train_y_arr))

    #print("test r2 {0:.2f}".format(r2_test))
    #print("test mse {0:.2f}".format(mse_test))
    #print("train r2 {0:.2f}".format(r2_train))
    #print("train mse {0:.2f}".format(mse_train))

    myseries = pd.Series([model_num, r2_test, mse_test, r2_train, mse_train])
    myseries.index = ["ModelName", "test_r2", "test_mse", "train_r2", "train_mse"]
    return myseries

def r2_compare(modeldb_path, impute_dir, y, exportpath=None, SpecialTag=None):
    tag = SpecialTag
    if os.path.isfile(modeldb_path):
        modeldb = load_obj(modeldb_path)
    else:
        print("modeldb not found")
        return

    cols = modeldb.columns.tolist()
    if "test_r2" not in cols:
        curr_db = modeldb
    elif tag:
        query = "r2_test > 0 | SpecialTag == " + str(tag)
        curr_db = modeldb.query(query)
    else:
        curr_db = modeldb.query("r2_test > 0")

    #load imputed data
    cooked_data_file = impute_dir + "/imputed.pk"
    train_fp = impute_dir + "/train.pk"
    test_fp = impute_dir + "/test.pk"
    cooked_df = load_obj(cooked_data_file)
    train_i = load_obj(train_fp)
    train_df = cooked_df.iloc[train_i]
    test_i = load_obj(test_fp)
    test_df = cooked_df.iloc[test_i]

    #get all metrics from DF
    temp_metrics_df = curr_db.apply(lambda row: r2_model(row["FullPath"],row["TransformTag"], y, row['ModelNum'],
                                                         train_df, test_df), axis=1)

    new_columns = ['ModelNum', 'r2_test', 'mse_test', 'r2_train', 'mse_train']
    temp_metrics_df.columns = new_columns

    modeldb = pd.merge(modeldb, temp_metrics_df, how='left', on='ModelNum')

    #save_obj(modeldb, modeldb_path)


#CODE TO MAKE A CONFUSION MATRIX
def train_test_confusion_plot_full(predicted_train, predicted_test, actual_train, actual_test, y, curr_tile, rev_transform_spec, full_dist=True):
                                   
    pred_train_df = pd.DataFrame({y: predicted_train})
    pred_train_df = apply_spec_to_df(y, rev_transform_spec, pred_train_df)
    predicted_train_tile = pred_train_df[y].apply(curr_tile)

    pred_test_df = pd.DataFrame({y: predicted_test})
    pred_test_df = apply_spec_to_df(y, rev_transform_spec, pred_test_df)
    predicted_test_tile = pred_test_df[y].apply(curr_tile)

    actual_train_df = pd.DataFrame({y: actual_train})
    actual_train_df = apply_spec_to_df(y, rev_transform_spec, actual_train_df)
    actual_train_tile = actual_train_df[y].apply(curr_tile)

    actual_test_df = pd.DataFrame({y: actual_test})
    actual_test_df = apply_spec_to_df(y, rev_transform_spec, actual_test_df)
    actual_test_tile = actual_test_df[y].apply(curr_tile)

    #setup figure for test items
    fig1 = plt.figure(figsize=(8, 11))
    gs1 = gridspec.GridSpec(3, 1)
    ax1 = fig1.add_subplot(gs1[0])
    ax2 = fig1.add_subplot(gs1[1])
    ax3 = fig1.add_subplot(gs1[2])
    ax_array1 = [ax1, ax2, ax3]

    test_mse, test_r2 = predict_metrics(actual_test, predicted_test, actual_test_tile.values,
                                        predicted_test_tile.values, y + ' - Test' , fig1, ax_array1)
    gs1.tight_layout(fig1, rect=[0.05,0.05,0.95,0.95], pad=0.5)

    #setup figure for train items
    fig2 = plt.figure(figsize=(8, 11))
    gs2 = gridspec.GridSpec(3, 1)
    ax4 = fig2.add_subplot(gs2[0])
    ax5 = fig2.add_subplot(gs2[1])
    ax6 = fig2.add_subplot(gs2[2])
    ax_array2 = [ax4, ax5, ax6]

    train_mse, train_r2 = predict_metrics(actual_train, predicted_train, actual_train_tile.values,
                                          predicted_train_tile.values, y + ' - Train' , fig2, ax_array2)
    gs2.tight_layout(fig2, rect=[0.05,0.05,0.95,0.95], pad=0.5)

    all_figs = [fig1, fig2]

    if full_dist:
        hist_figs = tile_accuracy_hist(actual_test_tile.values, predicted_test_tile.values, "Test", True)
        all_figs = all_figs + hist_figs

    return all_figs


