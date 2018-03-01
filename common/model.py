#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma 
    sharma0611
****************************************

Module: Models
Purpose: Hosts the Model object & its functions to interact with model properties

"""
from common.utils import ensure_folder, save_obj, load_obj
from common.multivariateanalysis import var_importance_table
from common.modelanalysis import r2_mse_grab, train_test_confusion_plot_full

class Model(object):
    """ 
    A Model object...
    On instantiation, the folder for the model is created. 
    On call to save() method, the dataframe associated & class object is saved locally

    Rule: You must use given functions to create a modelname & number
    model_dict = {'FullPath': model_path,
                  'ModelName': model_name,
                  'ModelNum': model_num,
                  'ModelAlgo': model_algo,
                  'TransformTag': transform,
                  'SpecialTag': special_tag,
                  'TrainingDataShape': shape,
                  'NumberFeaturesUsed': len(X),
                  'FeaturesUsed': str(features_used),
                  'test_r2': test_r2,
                  'test_mse': test_mse,
                  'test_accuracy': test_acc,
                  'train_r2': train_r2,
                  'train_mse': train_mse,
                  'train_acc': train_acc}

    """
    rs = 8439  #random state seed used for all seeds; change if you want to see a different seed
    
    def __init__(self, model_export_dir, name, model_object, y, X, model_algo, model_params, tag=""):
        #init export dir for current model
        curr_model_dir = model_export_dir + "/" + str(name)
        ensure_folder(curr_model_dir)
        #init path for model object, to hold model metadata
        model_path = curr_model_dir + "/metadata.pk"
        #init path for actual model_object to make predictions on
        model_object_path = curr_model_dir + "/model_object.pk"

        #init attributes
        self.export_dir = model_export_dir
        self.curr_model_dir = curr_model_dir
        self.name = name
        self.model_path = model_path
        self.model_object_path = model_object_path
        self.model_algo = model_algo
        self.model_params = model_params
        self.tag = tag
        self.y = y
        self.X = X
        self.num_features = len(X)
        self.train_time_mins = 0
        self.imp_df = None
        self.ordered_ftrs = None

        #init model_object
        self.model_object = model_object

    # function to save the model object into pickled file & associated model_object to pickled file
    def save(self):
        #save the model_object first
        save_obj(self.model_object, self.model_object_path)

        #delete model_object from Model object
        del self.model_object
        
        #save Model object
        save_obj(self, self.model_path)

    #function to load in the model_object associated with the Model object
    #It is not loaded on instantiation since we sometimes use the skeleton Model object to perform operations without the model_object
    #such as viewing model results, etc.
    def load_model_object(self):
        model_object = load_obj(self.model_object_path)
        self.model_object = model_object

    # Getters
    def get_name(self):
        return self.name

    def get_model_dir(self):
        return self.curr_model_dir

    def get_model_path(self):
        return self.model_path

    def get_model_object(self):
        return self.model_object

    def get_X(self):
        return self.X

    def get_importance_df(self):
        return self.imp_df

    # Setters
    def set_metadata_with_dataset(self, dataset):
        train_shape = (dataset.get_train_X_shape()[0], self.num_features)
        test_shape = (dataset.get_test_X_shape()[0], self.num_features)
        random_vars = dataset.get_random_variables()

        self.dataset_name = dataset.get_name()
        self.random_variables = random_vars
        self.train_shape = train_shape
        self.test_shape = test_shape

    def set_training_time(self, train_time_mins):
        self.train_time_mins = train_time_mins

    def set_metadata_feature(self, feature_name, value):
        self.__dict__[feature_name] = value

    def set_importance_df(self, importances):
        imp_df = var_importance_table(importances, self.X, self.name)
        #set importance df
        self.imp_df = imp_df
        ordered_ftrs = imp_df['var'].values
        #set ordered features
        self.ordered_ftrs = ordered_ftrs

    def r2_test(self, ds):
        test_X_arr, train_X_arr = ds.get_test_train_X_arrays(self.X)
        test_y_arr, train_y_arr = ds.get_test_train_var_arrays(self.y)

        #predict
        test_y_pred = self.model_object.predict(test_X_arr)
        train_y_pred = self.model_object.predict(train_X_arr)

        #metrics
        r2_test, mse_test = r2_mse_grab(test_y_arr, test_y_pred)
        r2_train, mse_train = r2_mse_grab(train_y_arr, train_y_pred)

        #save attributes
        self.r2_test = r2_test
        self.r2_train = r2_train
        self.mse_test = mse_test
        self.mse_train = mse_train
        self.test_y_pred = test_y_pred
        self.train_y_pred = train_y_pred

    def confusion_matrix(self, ds):
        test_y_arr, train_y_arr = ds.get_test_train_var_arrays(self.y)
        reverse_var, reverse_transform_spec = ds.get_reverse_transform_spec(self.y)
        curr_tile = ds.tile_func
        fig_arr = train_test_confusion_plot_full(self.train_y_pred, self.test_y_pred, train_y_arr, test_y_arr, self.y, curr_tile, reverse_transform_spec)
        return fig_arr

