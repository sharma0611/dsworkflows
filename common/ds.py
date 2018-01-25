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

class Dataset(object):
    """ 
    A Dataset object...

    """
    
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

