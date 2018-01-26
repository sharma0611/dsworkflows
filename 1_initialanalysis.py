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

from importlib import import_module
from importlib.util import find_spec, module_from_spec
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import numpy as np
from math import log

#import config settings
from common.config import y, raw_data_file, col_dtypes_dict, encoding
from common.featureanalysis import full_analysis, produce_transform_fn
from common.tilefile import createtilefile, add_to_file
import csv
import warnings

import os
from shutil import copyfile
from common.datainteract import Dataset

# new imports
from config.config import train_data, col_dtypes_dict, encoding, export_dir, custom_transforms, auto_transform_vars
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
ds.apply_transforms_metadata(custom_transforms)

#Apply auto-transforms to any variables you specify
ds.auto_transform(auto_transform_vars)

#Create a Dataset object to store the raw data and metadata
name = "raw"
#use dataset manager instead this time
mydataset = Dataset(name, raw_df, y)

#Apply the transformations to y suggested in config
#make transforms spec clear first in Dataset object
#apply transforms from specs always
mydataset.apply_transforms_y()

#Create a tile file if neccessary
#make this programmatic so no file is ever needed to be imported
mydataset.make_tile_file()

#Analyse dataframe & save analysis
mydataset.analyse_dataframe()

#Analyse transformed & non-transformed output variable distributions 
mydataset.analyse_target_distribution()

#Save the dataset
mydataset.save()

