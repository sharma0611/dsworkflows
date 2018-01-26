#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import pandas as pd

def create_tiling(y_data, num_tiles):
    tiling = {}
    tile_size = 1/num_tiles
    curr_quantile = tile_size
    quantiles = []
    for i in range(num_tiles - 1):
        quantiles.append(curr_quantile)
        curr_quantile += tile_size
    curr_tile = 1
    for quantile in quantiles:
        y_quantile = y_data.quantile(quantile)
        tiling[curr_tile] = y_quantile
        curr_tile += 1

    return tiling

#tiling is a dictionary of {tiles: upper bounds}
def ntile(val, tiling):
    if val is None:
        return -1
    for tile, upper_bound in tiling.items():
        if val <= upper_bound:
            return tile
    return len(tiling) + 1

