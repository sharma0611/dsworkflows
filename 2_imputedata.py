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

from common.dsm import Dataset_Manager
import pandas as pd
from config.config import category_cols, impute_to_mean, impute_to_value, remove_na_rows, random_variables, export_dir, conditions, blacklist, carry_along

#Instantiate Dataset Manager & load dataset object
dsm = Dataset_Manager(export_dir)
ds = dsm.load_dataset("raw")
ds.load_df()

#apply imputations specified in config
impute_ds = ds.apply_imputations(category_cols, impute_to_mean, impute_to_value, remove_na_rows, blacklist, random_variables, conditions, True, "impute")

#provide analysis on dataframe
impute_ds.analyse_dataframe()

#handle carry along
impute_ds.carry_along_split(carry_along)

#create train & test indicies
impute_ds.create_train_test()

#save the imputed dataset
impute_ds.save()

#from config.config import train_data, col_dtypes_dict, encoding, export_dir, custom_transforms, auto_transform_vars, num_tiles, y
#from common.dsm import Dataset_Manager
#import pandas as pd
#
##Read in the raw data
#raw_df = pd.read_csv(train_data, dtype=col_dtypes_dict, encoding=encoding)
#
##Instantiate Dataset Manager & create the Dataset object
#test_ds = dsm.create_dataset("test", raw_df)
#dsm.apply_all_ops(test_ds, impute_ds)
#
#ds = dsm.load_dataset("test")
#ds.load_df()
