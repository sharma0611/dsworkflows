#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:14:29 2017

@author: cih745
"""

import os
import pandas as pd
import sys
import pickle as pk

def createtilefile(y_data, num_tiles, tile_file):
    with open(tile_file, 'w') as f:
        header ="#!/usr/bin/env python3\n"
        defn ="def ntile(p_1):"
        check ="    if p_1 is None: return " + str(num_tiles+1)
        f.write(header + '\n')
        f.write(defn + '\n')
        f.write(check + '\n')
        tile_size = 1/num_tiles
        curr_quantile = tile_size
        quantiles = []
        for i in range(num_tiles - 1):
            quantiles.append(curr_quantile)
            curr_quantile += tile_size
        curr_tile = 1
        for quantile in quantiles:
            y_quantile = y_data.quantile(quantile)
            mid_q ="    elif p_1 <= {0} : return ".format(y_quantile) + str(curr_tile)
            f.write(mid_q + '\n')
            curr_tile += 1
        end ="    else: return " + str(curr_tile)
        f.write(end + '\n')
        f.close()

def add_to_file(file_path, texttoadd):
    with open(file_path, 'a') as f:
        f.write('\n' + texttoadd + '\n')

