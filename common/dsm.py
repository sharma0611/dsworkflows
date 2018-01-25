#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
****************************************
    Shivam Sharma 
    sharma0611
****************************************

Module: Dataset Manager
Purpose: Hosts the manager for all datasets, dataset instatiation, loading, & viewing

"""

from .utils import ensure_folder


class Dataset_Manager(object):
    """
    The Dataset Manager manages all datasets in a given folder it is initialized with
    """

    def __init__(self, export_dir):
        #init export directory for datasets
        dataset_dir = export_dir + "/Datasets"
        ensure_folder(dataset_dir)
        self.dataset_dir = dataset_dir



