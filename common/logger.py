#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cih745
"""
import subprocess
import sys
import os 

#dependencies:
    #Ghostscript
    #ps2pdf
    #enscript

class Logger(object):
    def __init__(self, fname):
        self.terminal = sys.stdout
        self.log = open(fname, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def shutdown(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass

def start_printer(file_dir, file_name):
    textfile = file_dir + '/' + file_name + '.txt'
    printer = Logger(textfile)
    sys.stdout = printer

def end_printer(file_dir, file_name):
    sys.stdout.shutdown()
    textfile = file_dir + '/' + file_name + '.txt'
    pdffile = file_dir + '/' + file_name + '.pdf'
    bashcmd1 = "enscript -p " + file_dir + "/outputps.ps " + textfile
    bashcmd2 = "ps2pdf " + file_dir + '/outputps.ps ' + pdffile
    subprocess.call(bashcmd1, shell=True)
    subprocess.call(bashcmd2, shell=True)

    #remove intermediate files
    os.remove(textfile)
    os.remove(file_dir + "/outputps.ps")

def join_pdfs(file_path1, file_path2, output_path, delete_original=True):
    bashcmd3 = "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile="+output_path+" "+file_path1+" " + file_path2
    subprocess.call(bashcmd3, shell=True)
    if delete_original:
        #remove the other two files
        os.remove(file_path1)
        os.remove(file_path2)


def join_pdfs_list(infile_list, output_path, delete_original=True):
    infile_txt = " ".join(infile_list)
    bashcmd4 = "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile="+output_path+" "+infile_txt
    subprocess.call(bashcmd4, shell=True)
    if delete_original:
        for file_path in infile_list:
            os.remove(file_path)


#Usage
#writeprints = True
#if writeprints:
#    start_printer()
#
### code with print statements
#
#if writeprints:
#    end_printer()
