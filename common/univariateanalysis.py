#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma 
    sharma0611
****************************************

Module: Univariate Analysis
Purpose: Hosts functions to analyse univariate data
"""
import numpy as np


def safe_fill(row, func, default_val, args, kwargs):
    try:
        arg_vals = [row[arg] for arg in args]
        curr_val = func(*arg_vals, **kwargs)
    except:
        curr_val = default_val
    return curr_val

def apply_transform_func(raw_df, var_name, default_val, transform_fn, args, kwargs):
    raw_df.loc[:,var_name] = raw_df.apply(lambda x: safe_fill(x, transform_fn, default_val, args, kwargs), axis=1)
    return raw_df

def apply_spec_to_df(var_name, var_spec, raw_df):
    transform_mod = import_module('.transforms', package='common')
    transform_fn_str = var_spec[0]
    transform_fn_args = var_spec[1]
    transform_fn_kwargs = var_spec[2]
    default_val = var_spec[3]
    transform_fn = getattr(transform_mod, transform_fn_str) #get the function
    raw_df = apply_transform_func(raw_df, var_name, default_val, transform_fn, transform_fn_args, transform_fn_kwargs)
    return raw_df

def suggest_transform_fn(var_data, input_var_name):

    results = {}
    #Set a Significance level for our hypothesis tests
    sig_lvl = 0.01

    #First, check if data is not normal
    is_normal, pval_orig = normality_test(var_data, sig_lvl)
    original_spec = var_data, ("original", [input_var_name], {})
    if is_normal:
        #then it is a normal distribution and we need not apply a transform
        return original_spec
    #join to results
    results["original"] = (pval_orig, original_spec)

    #otherwise, it is a non-normal distribution
    #Let's try boxcox

    #determine if data needs a shift
    shift = np.amin(var_data)
    if shift == 0:
        shift = 1
    elif shift > 0:
        shift = 0
    else:
        shift = abs(shift) + 1 #add one here to ensure no 0's in final array

    #shift it as such
    shift_var = [x + shift for x in var_data]

    #boxcox of shifted data
    shift_var_t, maxlog = stats.boxcox(shift_var)
    var_data_boxcox = [x - shift for x in shift_var_t]

    #inverse test; ensure inverse is possible
    temp_set = [maxlog * (y + shift) + 1 for y in var_data_boxcox]
    #check normality
    is_normal, pval_boxcox = normality_test(var_data_boxcox, sig_lvl)
    boxcox_spec = var_data_boxcox, ("boxcox", [input_var_name], {'maxlog': maxlog, 'shift': shift})
    if np.any(temp_set) <= 0: #or abs(maxlog) > 5:
        pass
        #you cannot perform the inverse so do not use
        #valid range for boxcox maxlog is from -5 to 5
    elif is_normal:
        #then this transformation is good
        #return boxcox spec
        return boxcox_spec
    #join to results
    results["boxcox"] = (pval_boxcox, boxcox_spec)

    #otherwise, the boxcox is no good, lets try z-score standardization
    mean = np.mean(var_data)
    std_dev = np.std(var_data)
    var_data_zscore = [(x_i - mean)/std_dev for x_i in var_data]


    #check if zscore is satisfactory
    is_normal, pval_zscore = normality_test(var_data_zscore, sig_lvl)
    zscore_spec = var_data_zscore, ("zscore", [input_var_name], {'mean':mean,'std':std_dev})
    if is_normal:
        #then the transformation is good; return the data and the spec
        return zscore_spec
    results["zscore"] = (pval_zscore, zscore_spec)

    #Otherwise, none of our normalization techniques worked
    #Let's try to use the highest pval & warn the user about this transformation
    max_key = max(results, key=lambda k: results[k][0])
    max_spec = results[max_key][1]
    print("Warning: The auto transformation did not pass hypothesis testing.")
    print("Using transform {0} with largest pvalue".format(max_key))
    return max_spec

def list_dict_intersect(list_a, dict_b):
    keys_a = set(list_a)
    keys_b = set(dict_b.keys())
    keys_c = keys_a & keys_b
    dict_c = {a : dict_b[a] for a in keys_c}
    return dict_c

