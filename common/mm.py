#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cih745
"""

import os
import pandas as pd
from common.utils import ensure_folder, fetch_db, save_obj, load_obj
from common.model import Model
import re
from functools import reduce
#Sample Model Metadata
#    
#    Rule: You must use given functions to create a modelname & number
#    model_dict = {'FullPath': model_path,
#                  'ModelName': model_name,
#                  'ModelNum': model_num,
#                  'ModelAlgo': model_algo,
#                  'TransformTag': transform,
#                  'SpecialTag': special_tag,
#                  'TrainingDataShape': shape,
#                  'NumberFeaturesUsed': len(X),
#                  'FeaturesUsed': str(features_used),
#                  'test_r2': test_r2,
#                  'test_mse': test_mse,
#                  'test_accuracy': test_acc,
#                  'train_r2': train_r2,
#                  'train_mse': train_mse,
#                  'train_acc': train_acc}

modeldb_path = "./modeldb.pk"


class Model_Manager(object):

    def __init__(self, export_dir):
        #init export directory for datasets
        model_export_dir = export_dir + "/Models"
        ensure_folder(model_export_dir)
        self.model_export_dir = model_export_dir
        #init database for models
        db_path = self.model_export_dir + "/models.db"
        self.db_path = db_path
        #stores any models that were created during runtime of mm object
        self.live_models = []

    def fetch_model_db(self):
        db = fetch_db(self.db_path)
        return db

    def save_model_db(self, db):
        save_obj(db, self.db_path)

    def create_model(self, name, model_object, y, X, model_algo, model_params, tag=""):
        db = self.fetch_model_db()
        curr_num = len(db) + 1
        name = str(curr_num) + "_" + name
        #init Dataset
        model = Model(self.model_export_dir, name, model_object, y, X, model_algo, model_params, tag)

        #save {name: path} to model db
        model_path = model.get_model_path()
        db[name] = model_path
        self.save_model_db(db)

        #add to live models
        self.live_models.append(model)

        return model

    def load_model(self, name):
        #grab model path
        db = self.fetch_model_db()
        model_path = db[name]
        model = load_obj(model_path)
        #add to live models
        self.live_models.append(model)
        return model

    def load_all_models(self):
        db = self.fetch_model_db()
        #first empty current models by saving
        self.save_all_live_models()
        for name, model_path in db.items():
            model = load_obj(model_path)
            #add to live models
            self.live_models.append(model)

    def save_all_live_models(self):
        for model in self.live_models:
            model.save()

        #reset live models
        self.live_models = []

    # Live Models Functions
    # functions that operate on all models you created during runtime of model manager

    def grab_live_models(self, y): #grab live models that are for a certain y 
        return_models = []
        for model in self.live_models:
            if model.y == y:
                return_models.append(model)
        return return_models

    #Function to grab all garbage features based on the % you want to throw out & importances
    def garbage_features(self, y, percent_garbage=0.2):
        varimportance_list = []
        random_variables = []
        curr_models = self.grab_live_models(y)
        for model in curr_models:
            varimportance_list.append(model.ordered_ftrs)
            random_variables = random_variables + model.random_variables

        random_variables = list(set(random_variables))

        cut_list = []
        for imp_list in varimportance_list:
            temp_list = imp_list[int(len(imp_list) * (1-percent_garbage)):]
            cut_list.append(temp_list)

        garbage_ftrs = list(set(cut_list[0]).intersection(*cut_list)-set(random_variables))

        return garbage_ftrs

    def grab_live_models_metadata_df(self):
        dict_list = []
        for model in self.live_models:
            curr_attrs = model.__dict__
            wanted_attrs = {k: v for k, v in curr_attrs.items() if isinstance(v, int) or isinstance(v, str) or isinstance(v, float)}
            dict_list.append(wanted_attrs)
        df = pd.DataFrame(dict_list)
        self.metadata_df = df
        return df

    def r2_test_live_models(self, ds):
        for model in self.live_models:
            model.r2_test(ds)

    def grab_features_importance_df(self, y): #use live models + filter by y 
        use_models = self.grab_live_models(y)
        imp_dfs = []
        for model in use_models:
            imp_df = model.imp_df
            imp_dfs.append(imp_df)
        if len(imp_dfs) >= 1:
            master_imp_df = reduce(lambda left,right: left.merge(right, how='outer', on='var'), imp_dfs)
        else:
            master_imp_df = pd.DataFrame()
        return master_imp_df
        

def load_modeldb(path_to_db=modeldb_path):
    if os.path.isfile(path_to_db):
        models_df = load_obj(path_to_db)
    else:
        models_df = pd.DataFrame()
    return models_df

def update_base_paths_db(db, ignore_exists=False):
    curr_base_path = export_dir
    modeldb_paths = db["FullPath"].values.tolist()
    relative_match = "/exports(.+)"
    re_relative_match = re.compile(relative_match)
    new_modeldb_paths = []
    for x in modeldb_paths:
        matches = re.findall(re_relative_match, x)
        if matches:
            relative_path = matches[0]
        else:
            print(x)
            print("match not found")
            return
        new_path = curr_base_path + relative_path
        if os.path.isfile(new_path) or ignore_exists:
            new_modeldb_paths.append(new_path)
        else:
            print(x)
            print(new_path)
            print("new path does not exist")
            return
    db["FullPath"] = new_modeldb_paths
    return db

def update_base_paths(ignore_exists=False, path_to_db=modeldb_path):
    db = load_modeldb(path_to_db)
    db = update_base_paths_db(db, ignore_exists)
    if not db.empty:
        #replace existing DB with updated version
        save_obj(db, path_to_db)
    else:
        print("Model db was not updated")

def load_all_models(models_df):
    model_dict = {}
    for index, row in models_df.iterrows():
        curr_model = load_obj(row['FullPath'])
        model_dict[row['ModelNum']] = curr_model
    return model_dict

def load_model(model_num, path_to_db=modeldb_path):
    if not os.path.isfile(path_to_db):
        print("No Model DB found.")
        return

    models_df = load_obj(path_to_db)
    results = models_df.query("ModelNum == " + str(model_num))

    if len(results) == 1:
        model_path = results['FullPath'].iloc[0]
        model = load_obj(model_path)
        return model
    elif len(results) > 1:
        print("More than one model with this # found.")
    else:
        print("No matches for this given model #.")

def load_model_features(model_num, path_to_db=modeldb_path):
    if not os.path.isfile(path_to_db):
        print("No Model DB found.")
        return

    models_df = load_obj(path_to_db)
    results = models_df.query("ModelNum == " + str(model_num))

    if len(results) == 1:
        features_used = results['FeaturesUsed'].iloc[0]
        features_used = eval(features_used)
        return features_used
    elif len(results) > 1:
        print("More than one model with this # found.")
    else:
        print("No matches for this given model #.")

#given a step #, pull all the models created from that step
def load_modelnums_fromstep(stepnum):
    df = load_modeldb()
    try:
        #in a try statement since modeldb may not have a Step column yet
        df = df.query("Step == " + str(stepnum))
        if not df.empty:
            modelnums = df["ModelNum"].values.tolist()
            return modelnums
        else:
            return []
    except:
        return []

def load_subsetnum_frommodelnum(modelnum):
    df = load_modeldb()
    df = df.query("ModelNum == " + str(modelnum))
    if not df.empty and len(df) == 1:
        subsetnum = df["Subset"].iloc[0]
        return int(subsetnum)
    else:
        return False


def modeldb_add(model_dict, step, path_to_db=modeldb_path):
    model_dict["Step"] = step
    curr_df = pd.DataFrame(model_dict, index=[0])

    #load in models_df
    models_df = load_modeldb(path_to_db)

    #update models_df with new model data and save
    models_df = models_df.append(curr_df, ignore_index=True)
    save_obj(models_df,path_to_db)

def modeldb_add_df(curr_models_df, step, path_to_db=modeldb_path):
    curr_models_df["Step"] = step
    #load in models_df
    models_df = load_modeldb(path_to_db)

    #update models_df with new model data and save
    models_df = pd.concat([models_df, curr_models_df], ignore_index=True)
    save_obj(models_df,path_to_db)


#this function takes a unique model number and the path to the DB
def modeldb_delete(model_num, path_to_db):
    #load in models_df
    if os.path.isfile(path_to_db):
        models_df = load_obj(path_to_db)
    else:
        print("model db is not at specified path.")
        return

    results = models_df.query('ModelNum == ' + str(model_num))
    if len(results) == 1:
        #delete the model
        del_path = results['ModelPath'].iloc[0]
        os.remove(del_path)
        #save new df
        models_df = models_df[models_df['ModelNum'] != model_num]
        save_obj(models_df,path_to_db)
    else:
        print("No Unique Match Found")
        print("Results: ")
        print(results)

def modeldb_delete_step(step, path_to_db=modeldb_path):
    #load in models_df
    if os.path.isfile(path_to_db):
        models_df = load_obj(path_to_db)
    else:
        print("model db is not at specified path.")
        return

    del_df = models_df.query('Step == ' + str(step))
    models_df = models_df.query('Step != ' + str(step))
    save_obj(models_df,path_to_db)
    print("Removed " + str(len(del_df)) + " models.")


def model_name_gen(attributes_list):
    counter_path = os.path.dirname(__file__) + '/modelcounter.pk'
    if os.path.isfile(counter_path):
        old_count = load_obj(counter_path)
        new_count = old_count + 1
    else:
        new_count = 0

    model_name = "_".join(attributes_list + ['model' + str(new_count)])
    save_obj(new_count, counter_path)
    return model_name, new_count

def reset_model_counter():
    counter_path = os.path.dirname(__file__) + '/modelcounter.pk'
    if os.path.isfile(counter_path):
        os.remove(counter_path)
    else:
        print("The counter does not exist.")

def reset_modeldb():
    if os.path.isfile(modeldb_path):
        reset_model_counter()
        os.remove(modeldb_path)
    else:
        print("Model DB does not exist.")
