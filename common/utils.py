#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma 
    sharma0611
****************************************

Module: Utilities
Purpose: Hosts functions for interacting with folders, os, etc.

"""

import os

def ensure_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)
