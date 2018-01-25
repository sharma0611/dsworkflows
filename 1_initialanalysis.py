#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma (CIH745)
    Data Science Co-op
    Summer 2017
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

#Turn off runtime warnings (for comparisons with np.nan)
warnings.filterwarnings("ignore",category =RuntimeWarning)

#Read in the raw data
raw_df = pd.read_csv(raw_data_file, dtype=col_dtypes_dict, encoding=encoding)

#Create a Dataset object to store the raw data and metadata
name = "Raw"
mydataset = Dataset(name, raw_df, y)


#Apply the transformations to y suggested in config
mydataset.apply_transforms_y()

#Create a tile file if neccessary
mydataset.make_tile_file()

#Analyse dataframe & save analysis
mydataset.analyse_dataframe()

#Analyse transformed & non-transformed output variable distributions 
mydataset.analyse_target_distribution()

#Save the dataset
mydataset.save()


