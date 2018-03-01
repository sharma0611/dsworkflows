#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    Shivam Sharma
    @sharma0611
"""

import os
import pandas as pd
from common.utils import ensure_folder, fetch_db, save_obj, load_obj
from common.model import Model
import re
from functools import reduce

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import lightgbm as lgb
from time import time

import warnings

from common.univariateanalysis import apply_spec_to_df
import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics.scorer import make_scorer
import scipy

import matplotlib.pyplot as plt
import numpy as np

from config.config import gs_iterations

#Sample Model Metadata
#    
#    Rule: You must use given functions to create a modelname & number
#    model_dict = {'FullPath': model_path,
#                  'ModelName': model_name,
#                  'ModelNum': model_num,
#                  'ModelAlgo': model_algo,
#                  'TransformTag': transform,
#                  'SpecialTag': special_tag,
#                  'TrainingDataShape': shape,
#                  'NumberFeaturesUsed': len(X),
#                  'FeaturesUsed': str(features_used),
#                  'test_r2': test_r2,
#                  'test_mse': test_mse,
#                  'test_accuracy': test_acc,
#                  'train_r2': train_r2,
#                  'train_mse': train_mse,
#                  'train_acc': train_acc}

modeldb_path = "./modeldb.pk"

class Model_Manager(object):

    def __init__(self, export_dir):
        #init export directory for datasets
        model_export_dir = export_dir + "/Models"
        ensure_folder(model_export_dir)
        self.model_export_dir = model_export_dir
        #init database for models
        db_path = self.model_export_dir + "/models.db"
        self.db_path = db_path
        #stores any models that were created during runtime of mm object
        self.live_models = []

    def fetch_model_db(self):
        db = fetch_db(self.db_path)
        return db

    def save_model_db(self, db):
        save_obj(db, self.db_path)

    def create_model(self, name, model_object, y, X, model_algo, model_params, tag=""):
        db = self.fetch_model_db()
        curr_num = len(db) + 1
        name = str(curr_num) + "_" + name
        #init Dataset
        model = Model(self.model_export_dir, name, model_object, y, X, model_algo, model_params, tag)

        #save {name: path} to model db
        model_path = model.get_model_path()
        db[name] = model_path
        self.save_model_db(db)

        #add to live models
        self.live_models.append(model)

        return model

    def load_model(self, name):
        #grab model path
        db = self.fetch_model_db()
        model_path = db[name]
        model = load_obj(model_path)
        #add to live models
        self.live_models.append(model)
        return model

    def load_all_models(self):
        db = self.fetch_model_db()
        #first empty current models by saving
        self.save_all_live_models()
        for name, model_path in db.items():
            model = load_obj(model_path)
            #add to live models
            self.live_models.append(model)

    def save_all_live_models(self):
        for model in self.live_models:
            model.save()

        #reset live models
        self.live_models = []

    def grab_all_models_metadata_df(self):
        self.load_all_models()
        df = self.grab_live_models_metadata_df()
        return df

    # Live Models Functions
    # functions that operate on all models you created during runtime of model manager

    def grab_live_models(self, y): #grab live models that are for a certain y 
        return_models = []
        for model in self.live_models:
            if model.y == y:
                return_models.append(model)
        return return_models

    #Function to grab all garbage features based on the % you want to throw out & importances
    def garbage_features(self, y, percent_garbage=0.2):
        varimportance_list = []
        random_variables = []
        curr_models = self.grab_live_models(y)
        for model in curr_models:
            varimportance_list.append(model.ordered_ftrs)
            random_variables = random_variables + model.random_variables

        random_variables = list(set(random_variables))

        cut_list = []
        for imp_list in varimportance_list:
            temp_list = imp_list[int(len(imp_list) * (1-percent_garbage)):]
            cut_list.append(temp_list)

        garbage_ftrs = list(set(cut_list[0]).intersection(*cut_list)-set(random_variables))

        return garbage_ftrs

    def grab_live_models_metadata_df(self):
        dict_list = []
        for model in self.live_models:
            curr_attrs = model.__dict__
            wanted_attrs = {k: v for k, v in curr_attrs.items() if isinstance(v, int) or isinstance(v, str) or isinstance(v, float)}
            dict_list.append(wanted_attrs)
        df = pd.DataFrame(dict_list)
        self.metadata_df = df
        return df

    def set_metadata_feature_live_models(self, feature_name, value):
        for model in self.live_models:
            model.set_metadata_feature(feature_name, value)

    def r2_test_live_models(self, ds):
        for model in self.live_models:
            model.r2_test(ds)

    def grab_features_importance_df(self, y): #use live models + filter by y 
        use_models = self.grab_live_models(y)
        imp_dfs = []
        for model in use_models:
            imp_df = model.imp_df
            imp_dfs.append(imp_df)
        if len(imp_dfs) >= 1:
            master_imp_df = reduce(lambda left,right: left.merge(right, how='outer', on='var'), imp_dfs)
        else:
            master_imp_df = pd.DataFrame()
        return master_imp_df

    # default model creation functions
    
    def create_rf_model(self, ds, X_arr, y_arr, y, X, step_tag, update_params=False):
        print("RF Model for " + y)
        t1 = time()
        params = {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 4,
                  'n_jobs':-1, 'random_state': 120}
        if update_params:
            params.update(update_params)

        rf_model_object = RandomForestRegressor(**params)
        rf_model_object.fit(X_arr, y_arr)

        #save model in Model object
        model_algo = "rf"
        model_name = step_tag + "_" + model_algo
        rf_model = self.create_model(model_name, rf_model_object, y, X, model_algo, params, step_tag)
        rf_model.set_metadata_with_dataset(ds)

        #get ordered list of important features
        importances = rf_model_object.feature_importances_
        rf_model.set_importance_df(importances)

        #calculate time
        t2 = time()
        t = (t2-t1)/60
        rf_model.set_training_time(t) #record time it took
        print("RF complete in " + '{0:.2f}'.format(t) + "m.")

        return rf_model

    def create_gbr_model(self, ds, X_arr, y_arr, y, X, step_tag, update_params=False):
        print("GBR Model for " + y)
        t1 = time()
        params = {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 4,
                  'learning_rate': 0.1, 'loss': 'ls', 'random_state': 912}
        if update_params:
            params.update(update_params)
        gbr_model_object = GradientBoostingRegressor(**params)
        gbr_model_object.fit(X_arr, y_arr)

        #save this model for later analysis
        model_algo = 'gbr'
        model_name = step_tag + "_" + model_algo
        gbr_model = self.create_model(model_name, gbr_model_object, y, X, model_algo, params, step_tag)
        gbr_model.set_metadata_with_dataset(ds)

        #output importances
        importances = gbr_model_object.feature_importances_
        gbr_model.set_importance_df(importances)

        #calculate time
        t2 = time()
        t = (t2-t1)/60
        gbr_model.set_training_time(t)
        print("GBR complete in " + '{0:.2f}'.format(t) + "m.")

        return gbr_model

    def create_lgbm_model(self, ds, X_arr, y_arr, y, X, step_tag, eval_set=None, update_params=False):
        assert X_arr.shape[1] == len(X)
        print("Microsoft LightGBM Model for " + y)
        t1 = time()
        params = {
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 300,
            'seed': 1231,
        }
        if update_params:
            params.update(update_params)

        gbm_model_object = lgb.LGBMRegressor(**params)
        category_cols = ds.get_category_cols(X)
        warnings.filterwarnings("ignore")
        try:
            gbm_model_object.fit(X_arr, y_arr,
                    eval_metric='l2',
                    eval_set=eval_set,
                    early_stopping_rounds=5,
                    categorical_feature=category_cols,
                    feature_name=X,
                    verbose=-1)
        except:
            gbm_model_object.fit(X_arr, y_arr,
                    categorical_feature=category_cols,
                    feature_name=X,
                    verbose=-1)
        #save this model for later analysis
        model_algo = 'lgbm'
        model_name = step_tag + "_" + model_algo
        gbm_model = self.create_model(model_name, gbm_model_object, y, X, model_algo, params, step_tag)
        gbm_model.set_metadata_with_dataset(ds)

        #output importances
        importances = gbm_model_object.feature_importances_
        gbm_model.set_importance_df(importances)

        #calculate time
        t2 = time()
        t = (t2-t1)/60
        gbm_model.set_training_time(t)
        print("LightGBM complete in " + '{0:.2f}'.format(t) + "m.")

        return gbm_model

    def regression_gridsearch_lgbm(self, ds, X, X_arr, y_arr):
        category_cols = ds.get_category_cols(X)

        scoring = {"Accuracy": make_scorer(ds.custom_accuracy, greater_is_better=True),
                   "Explained_Variance": make_scorer(metrics.explained_variance_score, greater_is_better=True),
                   "r2_score": make_scorer(metrics.r2_score, greater_is_better=True)}

        params_distr = {'learning_rate': scipy.stats.expon(scale=0.1),
                        'n_estimators': scipy.stats.randint(50, 400),
                        'max_depth': scipy.stats.randint(3, 15), 
                        'num_leaves': [5,10,20,30,40]}

        fit_params = {"feature_name":X,
                "categorical_feature":category_cols}

        fig_arr = []
        best_params = []
        for score_fn in scoring.keys():

            gs = RandomizedSearchCV(lgb.LGBMRegressor(), param_distributions=params_distr, scoring=scoring,
                    n_iter=gs_iterations, n_jobs=1, fit_params=fit_params, refit=score_fn, cv=3, return_train_score=True)

            gs.fit(X_arr, y_arr)

            results = gs.cv_results_
            best_param = gs.best_params_
            print("Best Parameters using %s :" % score_fn)
            print(best_param)
            best_params.append(best_param)

            res = pd.DataFrame(results)

            for i in params_distr.keys():
                fig = plt.figure(figsize=(8,6), dpi=200)
                ax = fig.add_subplot(111)
                ax.set_title("GridSearchCV evaluating using " + str(score_fn))
                ax.set_xlabel(i)
                ax.set_ylabel("Score")
                res.sort_values("param_%s" % i, inplace=True)
                X_axis = np.array(res['param_%s' % i].values, dtype=float)
                
                for scorer, color in zip(sorted(scoring), ['g', 'k']):
                    for sample, style in (('train', '--'), ('test', '-')):
                        sample_score_mean = res['mean_%s_%s' % (sample, scorer)].values.tolist()
                        sample_score_std = res['std_%s_%s' % (sample, scorer)].values.tolist()
                        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                                sample_score_mean + sample_score_std,
                                alpha=0.1 if sample == 'test' else 0, color=color)
                        ax.plot(X_axis, sample_score_mean, style, color=color,
                                alpha=1 if sample == 'test' else 0.7,
                                label="%s (%s)" % (scorer, sample))

                best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
                best_score = results['mean_test_%s' % scorer][best_index]

                # Plot a dotted vertical line at the best score for that scorer marked by x
                ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                        linestyle='-.', color=color, marker='s', markeredgewidth=3, ms=8)

                # Annotate the best score for that scorer
                ax.annotate("%0.2f" % best_score,
                            (X_axis[best_index], best_score + 0.005))

                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles,labels)

                fig_arr.append(fig)

        return fig_arr, best_params

