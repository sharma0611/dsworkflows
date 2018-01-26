#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:14:29 2017

@author: cih745
"""
from math import log, exp
from numpy import amin

### INVERTABLE TRANSFORMS
# specify any transformations & their inverses below

def original(y):
    return y

def natlog(y):
    return log(y)

def natexp(y):
    return exp(y)

def logit(y):
    return log(y/(1-y))

def proba(logit):
    return exp(logit)/(1+exp(logit))

def boxcox(y, maxlog, shift):
    y_shift = y + shift
    if maxlog == 0:
        return log(y_shift) - shift
    else:
        y_box = (y_shift ** maxlog - 1)/maxlog
        return y_box - shift

def inv_boxcox(y, maxlog, shift):
    y_shift = y + shift
    if maxlog == 0:
        return exp(y_shift) - shift
    else:
        y_box = (y_shift * maxlog) + 1
        y_box = y_box ** (1/maxlog)
        return y_box - shift

def zscore(y, mean, std_dev):
    return (y - mean) / std_dev

def inv_zscore(y, mean, std_dev):
    return (y * std_dev) + mean


### NON INVERTABLE TRANSFORMS
#non invertable transforms, specify any transformations that involve multiple variables below
#Ex. def sum_it(var_1, var_2): return var_1 + var_2



### PROCESSING
#this dictionary maps inverse operations, ensure to add to this dict if you make new transformations
inverse_dictionary = {"original": "original",
                      "natlog": "natexp",
                      "logit": "inv_logit",
                      "boxcox": "inv_boxcox",
                      "zscore": "inv_zscore"}

#post processing: adding the originals as inverses of inverses
new_dict = {}
for k, v in inverse_dictionary.items():
    new_dict[v] = k

inverse_dictionary = {**inverse_dictionary, **new_dict}
