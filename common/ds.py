#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma 
    sharma0611
****************************************

Module: Dataset
Purpose: Hosts the Dataset object & its functions to interact with the dataset

# cannot load dataset from Dataset object; must use Dataset_Manager
"""
from common.utils import ensure_folder, save_obj, load_obj
from common.tilefile import create_tiling
from common.univariateanalysis import suggest_transform_fn, apply_spec_to_df

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
        self.parent = None
        self.tiling = {}

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
        
        #save Dataset object
        save_obj(self, self.dataset_path)

    def copy(self, new_name):
        copy_ds = Dataset(self.export_dir, new_name, self.df)
        copy_ds.set_transforms_metadata(self.transforms_metadata)
        copy_ds.set_transforms_y(self.transforms_y)
        copy_ds.set_le_dict(self.le_dict)
        copy_ds.set_impute_dict(self.impute_dict)
        return copy_ds


    #function to load in the dataframe associated with the Dataset object
    def load_df(self):
        df = load_obj(self.dataframe_fp)
        self.df = df

    # Getters
    def get_name(self):
        return self.name

    def get_ds_dir(self):
        return self.curr_dataset_dir

    def get_ds_path(self):
        return self.dataset_path

    def set_target(self, y):
        assert y in self.X
        self.y = y

        #remove target from X features
        X = self.X
        X.remove(y)
        self.X = X

    # Setters

    def set_parent(self, parent_ds):
        parent_name = parent_ds.get_name()
        self.parent = parent_name

    def set_le_dict(self, le_dict):
        self.labelencode_dict = le_dict

    def set_impute_dict(self, impute_dict):
        self.impute_dict = impute_dict

    def set_transforms_y(self, transforms_y):
        self.transforms_y = transforms_y
        #ensure it is not in X
        X = self.X
        X = list(set(X) - set(transforms_y))
        self.X = X

    # Data Transformations

    def create_train_test(self):
        #create train test indices
        train, test = train_test_split(self.df, train_size=0.7, random_state=Dataset.rs)
        train_i = train.index.tolist()
        test_i = test.index.tolist()
        self.train_i = train_i
        self.test_i = test_i

    def tile_y(self, num_tiles):
        y_data = self.df[self.y]
        tiling = create_tiling(y_data, num_tiles)
        self.tiling = tiling


    # implement so that you can either create a new dataset or keep the old one
    def apply_transform_metadata(self, transform_metadata_in, new_ds=False, new_ds_name=""):
        if new_ds:
            new_ds = self.copy(new_ds_name)
            new_ds.apply_transform_metadata(transform_metadata_in)
            new_ds.set_parent(self)
            return new_ds
        else:
            for var_name, var_spec in transform_metadata_in.items():
                if self.y in var_spec[1]:
                    # y is being transformed thus add to transforms_y
                    self.transforms_y.append(var_name)
                new_df = apply_spec_to_df(var_name, var_spec, self.df)
                self.df = new_df
            #update transforms metadata
            self.transforms_metadata =  {**self.transforms_metadata, **transform_metadata_in}
            return self

    #takes an array of variables you want to auto transform, and a new name for the dataset
    def auto_transform(self, auto_vars, new_ds=False, new_ds_name=""):
        auto_transform_metadata = {}
        for var in auto_vars:
            var_data = self.df[var]
            auto_spec = suggest_transform_fn(var_data, var)
            auto_transform_metadata["auto_" + var] = auto_spec

        ds = self.apply_transform_metadata(auto_transform_metadata, new_ds, new_ds_name)
        return ds



# when applying transformations, ensure you filter for when you apply any transform to the set target variable
