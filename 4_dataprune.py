#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma (CIH745)
    Data Science Co-op
    Summer 2017
****************************************

Script: Data Prune
Purpose: Splices original datasets into multiple subsets using settings & transformations found in config &
         common/dataselect

"""

import pandas as pd
import numpy as np

from config.config import export_dir, select_y
from common.dsm import Dataset_Manager


#Instantiate Dataset Manager & load dataset object
dsm = Dataset_Manager(export_dir)
ds = dsm.load_dataset("impute")
ds.load_df()

#create prune set
prune_ds = dsm.copy_dataset(ds, "prune")

#prune y subset
prune_ds.prune_y(select_y)

#prune away garbage features
garbage_ftrs = prune_ds.get_garbage_ftrs(select_y)
X = prune_ds.get_X()
include = list(set(X) - set(garbage_ftrs))
prune_ds.prune_features(include, True)

#analyse & save
prune_ds.analyse_dataframe()
prune_ds.save()

