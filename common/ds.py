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
from .utils import ensure_folder, save_obj, read_obj
from .tilefile import create_tiling
from .univariateanalysis import suggest_transform_fn, apply_spec_to_df

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
        self.curr_dataset_dir = curr_dataset_dir
        self.name = name
        self.dataset_path = dataset_path
        self.dataframe_path = dataframe_path
        self.transforms_metadata = {}
        self.transforms_y = []

        #init dataframe
        self.df = df

    # function to save the dataset object into pickled file & associated dataframe to pickled file
    def save(self):
        #save the dataframe first
        save_obj(self.df, self.dataframe_path)

        #delete dataframe from Dataset object
        del self.df
        
        #save Dataset object
        save_obj(self, self.dataset_path)

    #function to load in the dataframe associated with the Dataset object
    def load_df(self):
        df = load_obj(self.dataframe_fp)
        self.df = df

    # Getters

    def get_ds_dir(self):
        return self.curr_dataset_dir

    def get_ds_path(self):
        return self.dataset_path

    def set_target(self, y):
        assert y in self.df.columns.tolist()
        self.y = y

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
    def apply_transform_metadata(self, transform_metadata, new_ds=False, new_ds_name=""):
        for var_name, var_spec in var_spec_dict.items():
            curr_df = apply_spec_to_df(var_name, var_spec, self.df)


    #transform_spec -> tuple(transform_fn, transform_fn_args, transform_fn_kwargs, default_val)
    #if 'X' is one of the keys; use its values to trim the dataset
    #only keep specified X, & variables from get_non_X()
    #ensure order of var_spec_dict is respected; as to split with X when neccessary, etc
    def apply_transform_metadata(self, var_spec_dict):
        # take current raw dataframe and perform deep copy
        raw_df = self.raw_df
        curr_df = raw_df.copy(deep=True)
        transform_mod = import_module('.transforms', package='common')
        curr_name = self.name
        for var_name, var_spec in var_spec_dict.items():
            if var_name == "X": #interpret this as a cut step; where X is pruned
                #trim immediately
                X = var_spec
                keep = X + self.get_non_X()
                curr_df = curr_df[keep]
            elif var_name == "Name": #interpret this as a rename in the sequential transform dict
                #add to the name
                curr_name = curr_name + "_" + str(var_spec)
            elif var_name == "Function":
                #must be a function that outputs a dataframe
                #add to name
                curr_name = curr_name + "_" + str(var_spec)
                df_func = getattr(transform_mod, var_spec)
                curr_df = df_func(curr_df) #apply the function
            elif var_name == "Function_metadata":
                #must be a function that takes a dataframe & returns a suggested metadata to apply
                #does not return out from this function block
                #add to name
                df_func = getattr(transform_mod, var_spec)
                curr_transform_metadata = df_func(curr_df) #apply the function
                curr_transform_metadata["Name"] = str(var_spec)
                ds = self.apply_transform_metadata(curr_transform_metadata)
                return ds
            else:
                #otherwise, the entry is a legitimate var_name, var_spec pair
                curr_df = apply_spec_to_df(var_name, var_spec, curr_df)

        #take new dataframe & initialize new dataset
        ds = Dataset(curr_name, curr_df, self.y)
        #first we should update the metadata with previous metadata + new metadata
        curr_spec_dict = self.transforms_metadata
        curr_transforms_metadata = {**curr_spec_dict, **var_spec_dict}
        ds.set_transforms_metadata(curr_transforms_metadata)
        #update rest of the attributes to parent's attributes
        ds.set_le_dict(self.labelencode_dict)
        ds.set_impute_dict(self.impute_dict)
        ds.set_transforms_y(self.transforms_y)
        ds.carry_along_split(self.carry_along) #updates X to not include carry along
        #identify parent dataset
        ds.set_parent(self.dataset_num)
        return ds

    #takes an array of variables you want to auto transform, and a new name for the dataset
    def auto_transform(self, auto_vars, new_ds=False, new_ds_name=""):
        auto_transform_metadata = {}
        for var in auto_vars:
            auto_spec = suggest_transform_fn(var_data, var_name)
            auto_transform_metadata["auto_" + var] = auto_spec

        self.apply_transform_metadata



# when applying transformations, ensure you filter for when you apply any transform to the set target variable
