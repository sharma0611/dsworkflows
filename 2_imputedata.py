#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma
    sharma0611
****************************************

Script: Impute Data
Purpose: Splices original datasets into multiple subsets using settings & transformations found in config &
         common/dataselect

"""

from config.config import train_data, col_dtypes_dict, encoding, export_dir, custom_transforms, auto_transform_vars, num_tiles, y
from common.dsm import Dataset_Manager
import pandas as pd

#Read in the raw data
raw_df = pd.read_csv(train_data, dtype=col_dtypes_dict, encoding=encoding)

#Instantiate Dataset Manager & create the Dataset object
dsm = Dataset_Manager(export_dir)
ds = dsm.create_dataset("raw", raw_df)

#set the target variable
ds.set_target(y)

#Apply any transforms you specified in config
#The Dataset object understands when you apply transformations to the target variable; saving them as alt target variables for futher analysis
print("Applying custom transformations...")
ds.apply_transform_metadata(custom_transforms)

#Apply auto-transforms to any variables you specify
print("Applying auto transformations...")
ds.auto_transform(auto_transform_vars)

#Create a tiling for the target variable
print("Creating a tiling for y...")
ds.tile_y(num_tiles)

#Analyse dataframe & save analysis
print("Analysing raw data & features...")
ds.analyse_dataframe()

#Analyse transformed & non-transformed output variable distributions 
print("Analysing target variable distribution...")
ds.analyse_target_distribution()

#Save the dataset
print("Saving your dataset...")
ds.save()

print("Initial Analysis Complete.")


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from common.speed import pk_dump, pk_load
from common.config import carry_along
from importlib import import_module

from numpy import random
from common.datainteract import Dataset

#Instantiate Dataset Manager & load dataset object
dsm = Dataset_Manager(export_dir)
ds = dsm.load_dataset("raw")
ds.load_df()

#apply imputations specified in config
impute_ds = ds.apply_fresh_impute()

#provide analysis on dataframe
impute_ds.analyse_dataframe()

#handle carry along
impute_ds.carry_along_split(carry_along)

#create train & test indicies
impute_ds.create_train_test()

#save the imputed dataset
impute_ds.save()

