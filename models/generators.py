# -*- coding: utf-8 -*-
"""
Generators
"""
from utils.deep_utils import depth_read, rgb_read, simul_shuffle
import numpy as np

def _batchGenerator(X_filelist,y_filelist,batchSize):
    """
    Yield X and Y data when the batch is filled.
    """
    #Sort filelists to confirm they are same order
    X_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    y_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    #Shuffle order of filenames
    X_filelist,y_filelist=simul_shuffle(X_filelist,y_filelist)

    while True:
        idx=0
        
        while idx<len(X_filelist):
            X_train=np.zeros((batchSize,480,640,3),dtype=np.uint8)
            y_train=np.zeros((batchSize,480,640),dtype=np.uint8)
            
            for i in range(batchSize):
                #Load images
                X_train[i]=rgb_read(X_filelist[idx+i])
                y_train[i]=depth_read(y_filelist[idx+i])
    
            #Reshape [samples][width][height][pixels]
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 
                                      X_train.shape[2], X_train.shape[3]).astype(np.uint8)

            y_train = y_train.reshape((y_train.shape[0],1,-1)).astype(np.uint8)
            y_train = y_train.squeeze()
                 
            # normalize inputs and outputs from 0-255 to 0-1
            X_train=np.divide(X_train,255).astype(np.float16)   
            y_train=np.divide(y_train,255).astype(np.float16)
            
            if (idx % 1024)==0:
                print(str(idx)+'/'+str(len(X_filelist)))
                
            idx+=batchSize
            
            yield X_train, y_train
            
def _valBatchGenerator(X_val_filelist,y_val_filelist,batchSize):
    """
    Yield X and Y data when the batch is filled.
    """
    #Sort filelists to confirm they are same order
    X_val_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    y_val_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    #Shuffle order of filenames
    X_val_filelist,y_val_filelist=simul_shuffle(X_val_filelist,y_val_filelist)

    while True:
        idx=0
        
        while idx<len(X_val_filelist):
            X_val=np.zeros((batchSize,480,640,3),dtype=np.uint8)
            y_val=np.zeros((batchSize,480,640),dtype=np.uint8)
            
            for i in range(batchSize):
                #Load images
                X_val[i]=rgb_read(X_val_filelist[idx+i])
                y_val[i]=depth_read(y_val_filelist[idx+i])
    
            #Reshape [samples][width][height][pixels]
            X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 
                                  X_val.shape[2], X_val.shape[3]).astype(np.uint8)

            y_val = y_val.reshape((y_val.shape[0],1,-1)).astype(np.uint8)
            y_val = y_val.squeeze()
                 
            # normalize inputs and outputs from 0-255 to 0-1
            X_val=np.divide(X_val,255).astype(np.float16)   
            y_val=np.divide(y_val,255).astype(np.float16)
            
            if (idx % 1024)==0:
                print(str(idx)+'/'+str(len(X_val_filelist)))
                
            idx+=batchSize
            
            yield X_val, y_val