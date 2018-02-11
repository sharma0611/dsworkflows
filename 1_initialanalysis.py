#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma
    sharma0611
****************************************

Script: Initial Analysis & Setup
Purpose: Runs setup for dataset; initializes objects from config settings

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
