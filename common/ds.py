#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma 
    sharma0611 
****************************************

Module: Dataset
Purpose: Hosts the Dataset object & its functions to interact with the dataset

"""
from common.utils import ensure_folder, save_obj, load_obj
from common.tilefile import create_tiling, ntile
from sklearn.model_selection import train_test_split
from common.univariateanalysis import suggest_transform_fn, apply_spec_to_df
from common.multivariateanalysis import full_analysis, safe_loc
from common.graphing import figures_to_pdf, hist_prob_plot, histogram_boxcox_plot
from common.imputations import (impute_categories, apply_le_dict, get_means, impute_na_rows, impute_blacklist, 
        add_random_variables, impute_conditions, impute_to_value)
from copy import deepcopy
from numpy import array
import os
import shutil


class Dataset(object):
    """ 
    A Dataset object...
    On instantiation, the folder for the dataset is created. 
    On call to save() method, the dataframe associated & class object is saved locally

    transforms_metadata -> dict{transform_variable_name: transform_spec}
        transform_variable_name -> str
        transform_spec -> tuple(transform_fn, transform_fn_args, transform_fn_kwargs, default_val)
            transform_fn -> str
            transform_fn_args -> array(str) of variables used as input to the transform fn
            transform_fn_kwargs -> dict(str:mixed)
    transforms_metadata is a dictionary where keys are transformed variable names and values are tuples
    that represent the variable spec

    """
    rs = 8439  #random state seed used for all seeds; change if you want to see a different seed
    
    def __init__(self, dataset_export_dir, name, df):
        #init export dir for current dataset
        curr_dataset_dir = dataset_export_dir + "/" + str(name)
        ensure_folder(curr_dataset_dir)
        #init path for dataset object
        dataset_path = curr_dataset_dir + "/dataset.pk"
        #init path for dataset dataframe
        dataframe_path = curr_dataset_dir + "/dataframe.pk"

        #init attributes
        self.export_dir = dataset_export_dir
        self.curr_dataset_dir = curr_dataset_dir
        self.name = name
        self.dataset_path = dataset_path
        self.dataframe_path = dataframe_path
        self.transforms_metadata = {}
        self.transforms_y = []
        self.le_dict = {}
        self.impute_dict = {}
        self.op_history = []
        self.tiling = {}
        self.y = None
        self.carry_along = []
        self.category_cols = []
        self.test_df = None
        self.train_df = None
        self.random_variables = []
        self.garbage_features = {}
        self.train_i = None
        self.test_i = None

        #init dataframe
        self.df = df

        #setting X (features)
        X = df.columns.tolist()
        self.X = X

    # function to save the dataset object into pickled file & associated dataframe to pickled file
    def save(self):
        #save the dataframe first
        save_obj(self.df, self.dataframe_path)

        #delete dataframe from Dataset object
        del self.df
        del self.train_df
        del self.test_df
        
        #save Dataset object
        save_obj(self, self.dataset_path)
        self.new_op("save")

    # function to apply operations in place
    def apply_ops(self, ops):
        for op in ops:
            method_name = op[0]
            args = op[1]
            curr_method = getattr(self, method_name)
            return_val = curr_method(*args) #this should also add the op to the op history

    # function to record a new operation performed
    def new_op(self, method, *args):
        new_op = (method, args)
        self.op_history.append(new_op)

    # copy a dataframe with all attributes; deepcopy of df and new name & directory for dataset
    def copy(self, new_name):
        copy_df = self.df.copy(deep=True)
        copy_ds = Dataset(self.export_dir, new_name, copy_df)

        #init attributes
        copy_ds.transforms_metadata = self.transforms_metadata 
        copy_ds.transforms_y = self.transforms_y 
        copy_ds.le_dict = self.le_dict 
        copy_ds.impute_dict = self.impute_dict 
        copy_ds.op_history = self.op_history 
        copy_ds.tiling = self.tiling 
        copy_ds.y = self.y 
        copy_ds.carry_along = self.carry_along 
        copy_ds.category_cols = self.category_cols
        copy_ds.random_variables = self.random_variables 
        copy_ds.garbage_features = self.garbage_features
        copy_ds.X = self.X
        copy_ds.train_i = self.train_i
        copy_ds.test_i = self.test_i

        return copy_ds

    #function to load in the dataframe associated with the Dataset object
    #It is not loaded on instantiation since we sometimes use the skeleton Dataset object to perform operations without the dataframe
    #such as applying transformations on new data, viewing related model results, etc.
    def load_df(self):
        df = load_obj(self.dataframe_path)
        self.df = df

    def make_custom_dir(self, dir_name, overwrite):
        custom_dir = self.export_dir + "/" + dir_name
        curr_dir = custom_dir
        if not overwrite:
            id = 0 
            while(os.path.isdir(curr_dir)):
                curr_dir = custom_dir + "_" + str(id)
                id += 1
        else:
            #to overwrite clear directory first
            shutil.rmtree(curr_dir)
        ensure_folder(curr_dir)
        return curr_dir

    def get_custom_dir(self, dir_name, id=False):
        curr_dir = self.export_dir + "/" + dir_name
        if id:
            curr_dir = curr_dir + "_" + str(id) 
        return curr_dir

    # Getters
    def get_name(self):
        return self.name

    def get_ds_dir(self):
        return self.curr_dataset_dir

    def get_ds_path(self):
        return self.dataset_path

    def get_all_y(self):
        return [self.y] + self.transforms_y

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y

    def get_non_X(self):
        non_X = [self.y] + self.transforms_y + self.carry_along
        return non_X

    def get_non_y(self):
        non_y = self.X + self.carry_along
        return non_y

    def get_random_variables(self):
        return self.random_variables

    def get_garbage_ftrs(self, y):
        garbage_ftrs = self.garbage_features[y]
        return garbage_ftrs

    def get_category_cols(self, X):
        return list(set(self.category_cols).intersection(X))

    def get_categorical_bool(self, X=False):
        if not X:
            X = self.X
        categorical_bool = [True if x in self.category_cols else False for x in X]
        return categorical_bool

    # Setters
    def set_target(self, y):
        assert y in self.X
        self.y = y

        #remove target from X features
        self.X.remove(y)

        #add an op
        self.new_op("set_target", y)

    #function to set aside a feature as a transformed y variable 
    def add_y_transform(self, transform_y):
        # this is a NON OP function; should not be called from anywhere except by apply_transform_metadata
        self.transforms_y.append(transform_y)

        #remove target from X features
        try:
            self.X.remove(transform_y)
        except:
            pass

    def update_le_dict(self, le_dict):
        self.labelencode_dict = {**self.le_dict, **le_dict}

    def update_impute_dict(self, impute_dict):
        self.impute_dict = {**self.impute_dict, **impute_dict}

    def set_y_tiles(self, tiling):
        self.tiling = tiling
        self.new_op("set_y_tiles", tiling)

    def tile_func(self, value): #must set y tiles for this to be a thing
        return ntile(value, self.tiling)

    def set_category_cols(self, category_cols):
        self.category_cols = category_cols
        self.new_op("set_category_cols", category_cols)

    def set_garbage_ftrs(self, y, garbage_ftrs):
        self.garbage_features[y] = garbage_ftrs
        self.new_op("set_garbage_ftrs", y, garbage_ftrs)

    # Data Transformations

    def create_train_test(self):
        #create train test indices
        train, test = train_test_split(self.df, train_size=0.7, random_state=Dataset.rs)
        train_i = train.index.tolist()
        test_i = test.index.tolist()
        self.train_i = train_i
        self.test_i = test_i
        self.new_op("create_train_test")

    def load_test_train_dfs(self):
        cooked_df = self.df
        train_i = self.train_i
        test_i = self.test_i
        test_df = safe_loc(cooked_df, test_i)
        train_df = safe_loc(cooked_df, train_i)
        self.test_df = test_df
        self.train_df = train_df

    #only call this after loading test/train dfs
    def get_test_train_X_arrays(self, X=False):
        if not X:
            X = self.X
        test_df = self.test_df
        train_df = self.train_df
        train_X_df = train_df[X]
        train_X_arr = array(train_X_df)
        test_X_df = test_df[X]
        test_X_arr = array(test_X_df)
        train_X_shape = train_X_arr.shape
        test_X_shape = test_X_arr.shape
        self.train_X_shape = train_X_shape
        self.test_X_shape = test_X_shape
        return test_X_arr, train_X_arr

    def get_train_X_shape(self):
        return self.train_X_shape
    
    def get_test_X_shape(self):
        return self.test_X_shape

    #only call this after loading test/train dfs
    def get_test_train_var_arrays(self, var):
        test_df = self.test_df
        train_df = self.train_df
        train_var_series = train_df[var]
        train_var_arr = array(train_var_series)
        test_var_series = test_df[var]
        test_var_arr = array(test_var_series)
        return test_var_arr, train_var_arr

    #dyanmic op; so not a part of operations
    def tile_y(self, num_tiles):
        y_data = self.df[self.y]
        tiling = create_tiling(y_data, num_tiles)
        self.set_y_tiles(tiling)

    # implement so that you can either create a new dataset or keep the old one
    def apply_transform_metadata(self, transform_metadata_in):
        for var_name, var_spec in transform_metadata_in.items():
            if self.y in var_spec[1]:
                # y is being transformed thus add to transforms_y
                self.add_y_transform(var_name)
            new_df = apply_spec_to_df(var_name, var_spec, self.df)
            self.df = new_df
        #update transforms metadata
        self.transforms_metadata =  {**self.transforms_metadata, **transform_metadata_in}
        #record operation in ops history
        self.new_op("apply_transform_metadata", transform_metadata_in)
        return self

    #takes an array of variables you want to auto transform, and a new name for the dataset
    #this does not get recorded as a new op since it is dynamic; the underlying static apply_transform_metadata saves its ops instead
    def auto_transform(self, auto_vars):
        auto_transform_metadata = {}
        for var in auto_vars:
            var_data = self.df[var]
            auto_spec = suggest_transform_fn(var_data, var)
            auto_transform_metadata["auto_" + var] = auto_spec
        ds = self.apply_transform_metadata(auto_transform_metadata)
        return ds

    # Data Analysis 

    def analyse_dataframe(self, category_threshold=100):
        full_analysis(self.df, self.curr_dataset_dir, category_threshold)
        self.new_op("analyse_dataframe", category_threshold)

    def analyse_target_distribution(self):
        #get y data
        y = self.y
        output_var_data = self.df[y]

        #Histogram and boxcox plot for output
        hist_boxcox_fig = histogram_boxcox_plot(y, output_var_data)

        #get all transformed y variable names
        transforms = self.transforms_y

        #Take each transform & produce a histogram & probplot
        figures = [hist_boxcox_fig]
        for transform in transforms:
            curr_output_set = self.df[transform]
            curr_fig = hist_prob_plot(transform, curr_output_set)
            figures.append(curr_fig)

        pdffile_path = self.curr_dataset_dir + '/distributionofoutput.pdf'
        figures_to_pdf(figures, pdffile_path)
        self.new_op("analyse_target_distribution")

    # change impute function so that it works on the same dataset 
    # make it so another function has the calls to make a new dataset and apply the required imputations

    def impute_categories(self, category_cols):
        le_dict = impute_categories(self.df, category_cols)
        self.update_le_dict(le_dict)
        self.set_category_cols(category_cols)
        self.new_op("impute_categories", category_cols)

    def apply_le_dict(self, category_cols, le_dict):
        apply_le_dict(self.df, category_cols, le_dict)
        self.update_le_dict(le_dict)
        self.new_op("apply_le_dict", category_cols, le_dict)

    # this is dynamic; so not included in ops
    def impute_means(self, impute_to_mean):
        mean_dict = get_means(self.df, impute_to_mean)
        self.impute_to_value(mean_dict)

    def impute_na_rows(self, remove_na_rows):
        impute_na_rows(self.df, remove_na_rows)
        self.new_op("impute_na_rows", remove_na_rows)

    def impute_blacklist(self, blacklist):
        impute_blacklist(self.df, blacklist)
        self.new_op("impute_blacklist", blacklist)

    def add_random_variables(self, random_variables):
        add_random_variables(self.df, self.y, random_variables) # add them to df
        self.random_variables = self.random_variables + random_variables # update random variables
        self.X = self.X + random_variables #update X
        self.new_op("add_random_variables", random_variables)

    def impute_conditions(self, conditions):
        impute_conditions(self.df, conditions)
        self.new_op("impute_conditions", conditions)

    def impute_to_value(self, impute_values):
        impute_to_value(self.df, impute_values)
        self.update_impute_dict(impute_values)
        self.new_op("impute_to_value", impute_values)

    def reindex(self):
        self.df.index = list(range(len(self.df)))
        self.new_op("reindex")

    def sort_on(self, var):
        self.df.sort_values(var, inplace=True)
        self.new_op("sort_on", var)

    #fresh impute should create a new dataset and return it; no changes should happen to the original data
    def apply_imputations(self, category_cols, impute_to_mean, impute_to_value, remove_na_rows, 
            blacklist, random_variables, conditions):
        self.impute_na_rows(remove_na_rows)
        self.impute_conditions(conditions)
        self.impute_blacklist(blacklist)
        self.add_random_variables(random_variables)
        self.impute_means(impute_to_mean)
        self.impute_to_value(impute_to_value)
        self.impute_categories(category_cols)

    def carry_along_split(self, carry_along):
        the_cols = self.df.columns.tolist()
        #split into the actual & carry along set if carry along exists
        carry_along = list(set(carry_along).intersection(the_cols))
        #add carry along attribute
        self.carry_along = self.carry_along + carry_along
        #remove carry along from X
        self.X = list(set(self.X) - set(carry_along))
        self.new_op("carry_along_split", carry_along)

    def prune_features(self, new_X, keep_random_vars=False):
        curr_cols = self.X
        not_included = set(new_X) - set(curr_cols)
        if len(not_included) > 0:
            print("You are trying to select columns that do not exist in the current dataframe.")
            print("These cols are: {0}".format(str(not_included)))
        include = list(set(new_X).intersection(curr_cols))
        print("Selecting the following {0} of {1} columns in the dataframe...".format(len(include),
                                                                                      len(curr_cols)))
        print("Now there are only {0} intersection columns in the dataframe".format(len(include)))
        #reset X, category_cols
        self.X = include
        self.category_cols = list(set(self.category_cols).intersection(include))
        include = include + self.get_non_X()
        if keep_random_vars:
            include = list(set(include + self.random_variables))
        self.df = self.df[include]
        self.new_op("prune_features", new_X, keep_random_vars)

    def prune_y(self, select_y):
        if select_y in self.transforms_y:
            self.y = select_y
        assert self.y == select_y
        self.transforms_y = []
        include = [select_y] + self.get_non_y()
        print(include)
        self.df = self.df[include]
        self.new_op("prune_y", select_y)
