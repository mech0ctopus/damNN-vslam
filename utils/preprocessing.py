# -*- coding: utf-8 -*-
"""
Rename util
"""
import os
from glob import glob

def rename_file(directory, old_name, new_name, filetype='.pdf'):
    '''Renames file using shell commands.'''
    
    #Format is: ren "old filename.pdf" "new filename.pdf"
    cmd_string='ren "'+old_name+filetype+'" "'+new_name+filetype+'"'
    
    #Send command to shell
    os.chdir(directory)
    os.system(cmd_string)
    
def rename_files(directory, prefix, filetype='.pdf'):
    '''Renames file using shell commands.'''
    #Build list of files
    old_filenames=glob(directory+'\\*'+filetype)
    
    #For all files
    for old_filename in old_filenames:
        #Get basename
        old_filename=os.path.basename(old_filename).split('.')[0]
        #Add prefix
        new_filename=prefix+'_'+old_filename
        #Rename file
        rename_file(directory,old_filename,new_filename,filetype)

if __name__=="__main__":
    directory=r"G:\Documents\KITTI\raw_data\RGB\2011_09_30_drive_0027_sync"
    prefix=directory.split('\\')[-1]
    ftype=".png"
    rename_files(directory,prefix,ftype)