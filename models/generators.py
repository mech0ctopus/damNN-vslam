# -*- coding: utf-8 -*-
"""
Generators
"""
from utils.deep_utils import depth_read, rgb_read
import numpy as np

def _batchGenerator(X_filelist,y_filelist,batchSize):
    """
    Yield X and Y data when the batch is filled.
    """
    #Sort filelists to confirm they are same order
    X_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    y_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    #Do not shuffle!

    while True:
        idx=1 #Skip first image since we need t-1
        
        while idx+(batchSize-1)<len(X_filelist):
            X_train_1=np.zeros((batchSize,192,640,3),dtype=np.uint8)
            X_train_2=np.zeros((batchSize,192,640,3),dtype=np.uint8)
            y_train=np.zeros((batchSize,192,640),dtype=np.uint8)
            
            for i in range(batchSize):
                #Load images
                X_train_1[i]=rgb_read(X_filelist[idx+i])   #time=t
                X_train_2[i]=rgb_read(X_filelist[idx-1+i]) #time=t=1
                y_train[i]=depth_read(y_filelist[idx+i])   #time=t
    
            #Reshape [samples][width][height][pixels]
            X_train_1 = X_train_1.reshape(X_train_1.shape[0], X_train_1.shape[1], 
                                          X_train_1.shape[2], X_train_1.shape[3]).astype(np.uint8)
            X_train_2 = X_train_2.reshape(X_train_2.shape[0], X_train_2.shape[1], 
                                          X_train_2.shape[2], X_train_2.shape[3]).astype(np.uint8)
            
            y_train = y_train.reshape((y_train.shape[0],1,-1)).astype(np.uint8)
            y_train = y_train.squeeze()
                 
            # normalize inputs and outputs from 0-255 to 0-1
            X_train_1=np.divide(X_train_1,255).astype(np.float16)   
            X_train_2=np.divide(X_train_2,255).astype(np.float16)
            y_train=np.divide(y_train,255).astype(np.float16)
            
            if (idx % 1024)==0:
                print(str(idx)+'/'+str(len(X_filelist)))
                
            idx+=batchSize
            
            #Provide both images
            X_train=[X_train_1, X_train_2]
            
            yield X_train, y_train
            
def _valBatchGenerator(X_val_filelist,y_val_filelist,batchSize):
    """
    Yield X and Y data when the batch is filled.
    """
    #Sort filelists to confirm they are same order
    X_val_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    y_val_filelist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    
    #Do not shuffle!

    while True:
        idx=1 #Skip first image since we need t-1
        
        while idx+(batchSize-1)<len(X_val_filelist):
            X_val_1=np.zeros((batchSize,192,640,3),dtype=np.uint8)
            X_val_2=np.zeros((batchSize,192,640,3),dtype=np.uint8)
            y_val=np.zeros((batchSize,192,640),dtype=np.uint8)
            
            for i in range(batchSize):
                #Load images
                X_val_1[i]=rgb_read(X_val_filelist[idx+i])   #time=t
                X_val_2[i]=rgb_read(X_val_filelist[idx-1+i]) #time=t-1
                y_val[i]=depth_read(y_val_filelist[idx+i])   #time=t
    
            #Reshape [samples][width][height][pixels]
            X_val_1 = X_val_1.reshape(X_val_1.shape[0], X_val_1.shape[1], 
                                      X_val_1.shape[2], X_val_1.shape[3]).astype(np.uint8)
            X_val_2 = X_val_2.reshape(X_val_2.shape[0], X_val_2.shape[1], 
                                      X_val_2.shape[2], X_val_2.shape[3]).astype(np.uint8)
            
            y_val = y_val.reshape((y_val.shape[0],1,-1)).astype(np.uint8)
            y_val = y_val.squeeze()
                 
            # normalize inputs and outputs from 0-255 to 0-1
            X_val_1=np.divide(X_val_1,255).astype(np.float16)
            X_val_2=np.divide(X_val_2,255).astype(np.float16)
            y_val=np.divide(y_val,255).astype(np.float16)
            
            if (idx % 1024)==0:
                print(str(idx)+'/'+str(len(X_val_filelist)))
                
            idx+=batchSize
            
            #Provide both images [time=t, time(t-1)]
            X_val=[X_val_1, X_val_2]
            
            yield X_val, y_val