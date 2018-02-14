#!/usr/bin/env python3

def ntile(p_1):
    if p_1 is None: return 21
    elif p_1 <= 88000.0 : return 1
    elif p_1 <= 106475.0 : return 2
    elif p_1 <= 115000.0 : return 3
    elif p_1 <= 124000.0 : return 4
    elif p_1 <= 129975.0 : return 5
    elif p_1 <= 135500.0 : return 6
    elif p_1 <= 141000.0 : return 7
    elif p_1 <= 147000.0 : return 8
    elif p_1 <= 155000.0 : return 9
    elif p_1 <= 163000.0 : return 10
    elif p_1 <= 172500.0 : return 11
    elif p_1 <= 179280.0 : return 12
    elif p_1 <= 187500.0 : return 13
    elif p_1 <= 198619.99999999997 : return 14
    elif p_1 <= 214000.0 : return 15
    elif p_1 <= 230000.0 : return 16
    elif p_1 <= 250000.0 : return 17
    elif p_1 <= 278000.0 : return 18
    elif p_1 <= 326100.0000000008 : return 19
    else: return 20
