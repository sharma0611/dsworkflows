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
import pickle as pk

#fix to get around issue 24658 in github of pickle module
class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
#        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
#            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
#            print("done.", flush=True)
            idx += batch_size

def pk_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pk.dump(obj, MacOSFile(f), protocol=pk.HIGHEST_PROTOCOL)

def pk_load(file_path):
    with open(file_path, "rb") as f:
        return pk.load(MacOSFile(f))

def fetch_db(db_path):
    if not os.path.isfile(db_path):
        return {}
    else:
        db = pk_load(db_path)
        return db

def ensure_folder(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def save_obj(obj, file_path):
    pk_dump(obj, file_path)

def load_obj(file_path):
    obj = pk_load(file_path)
    return obj

