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

from common.utils import ensure_folder, fetch_db, save_obj, load_obj
from common.ds import Dataset

class Dataset_Manager(object):
    """
    The Dataset Manager manages all datasets in a given folder it is initialized with
    """

    def __init__(self, export_dir):
        #init export directory for datasets
        dataset_export_dir = export_dir + "/Datasets"
        ensure_folder(dataset_export_dir)
        self.dataset_export_dir = dataset_export_dir
        #init database for datasets
        db_path = self.dataset_export_dir + "/datasets.db"
        self.db_path = db_path

    def fetch_dataset_db(self):
        db = fetch_db(self.db_path)
        return db

    def save_dataset_db(self, db):
        save_obj(db, self.db_path)

    def copy_dataset(self, ds, name):
        new_ds = ds.copy(name)
        name = new_ds.name

        #save {name: path} to dataset db
        ds_path = new_ds.get_ds_path()
        db = self.fetch_dataset_db()
        db[name] = ds_path
        self.save_dataset_db(db)

        return new_ds

    def create_dataset(self, name, df):
        #init Dataset
        ds = Dataset(self.dataset_export_dir, name, df)

        #save {name: path} to dataset db
        ds_path = ds.get_ds_path()
        db = self.fetch_dataset_db()
        db[name] = ds_path
        self.save_dataset_db(db)

        return ds

    def load_dataset(self, name):
        #grab dataset path
        db = self.fetch_dataset_db()
        ds_path = db[name]
        ds = load_obj(ds_path)
        return ds

    def apply_all_ops(self, new_dataset, target_dataset):
        op_history = target_dataset.op_history
        new_dataset.apply_ops(op_history)

