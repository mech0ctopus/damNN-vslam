# -*- coding: utf-8 -*-
"""
Generators
"""
from utils.deep_utils import depth_read, rgb_read
import numpy as np
from os.path import basename
from utils.read_odom import read_odom, normalize

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
            X_train_1=np.zeros((batchSize,192,640,3),dtype=np.uint8)   #time=t
            X_train_2=np.zeros((batchSize,192,640,3),dtype=np.uint8)   #time=t-1
            y_train_depth=np.zeros((batchSize,192,640),dtype=np.uint8)   #time=t depth
            y_train_odom=np.zeros((batchSize,6),dtype=np.float64)   #dt odom

            for i in range(batchSize):
                #Load images
                X_train_1[i]=rgb_read(X_filelist[idx+i])   #time=t
                X_train_2[i]=rgb_read(X_filelist[idx-1+i]) #time=t-1
                y_train_depth[i]=depth_read(y_filelist[idx+i])   #time=t

                #Calculate change in odometry
                current_filename=X_filelist[idx+i]   #time=t
                prev_filename=X_filelist[idx-1+i]   #time=t-1
                
                sequence_id, frame_id=basename(current_filename).split('_sync_')
                prev_sequence_id, prev_frame_id=basename(prev_filename).split('_sync_')
                
                #print('original:' + frame_id)
                frame_id=int(frame_id.split('.')[0])
                prev_frame_id=int(prev_frame_id.split('.')[0])
                
                #print('converted:' + str(frame_id))
                
                #prev_frame_id=frame_id-1
                
                current_odom=read_odom(sequence_id, frame_id)
                prev_odom=read_odom(prev_sequence_id, prev_frame_id)
                # print('Train: '+f'{sequence_id}, {prev_sequence_id}')
                # print('Train: '+f'{frame_id}, {prev_frame_id}')
                y_train_odom[i]=normalize(current_odom-prev_odom)

    
            #Reshape [samples][width][height][pixels]
            X_train_1 = X_train_1.reshape(X_train_1.shape[0], X_train_1.shape[1], 
                                          X_train_1.shape[2], X_train_1.shape[3]).astype(np.uint8)
            X_train_2 = X_train_2.reshape(X_train_2.shape[0], X_train_2.shape[1], 
                                          X_train_2.shape[2], X_train_2.shape[3]).astype(np.uint8)
            
            y_train_depth = y_train_depth.reshape((y_train_depth.shape[0],1,-1)).astype(np.uint8)
            y_train_depth = y_train_depth.squeeze()
                 
            # normalize inputs and outputs from 0-255 to 0-1
            X_train_1=np.divide(X_train_1,255).astype(np.float16)   
            X_train_2=np.divide(X_train_2,255).astype(np.float16)
            y_train_depth=np.divide(y_train_depth,255).astype(np.float16)
            
            #if (idx % 36)==0:
            print(str(idx)+'/'+str(len(X_filelist)))
                
            idx+=batchSize
            
            #y_train_depth=y_train_depth.reshape((batchSize,len(y_train_depth)))
            y_train_depth=y_train_depth.reshape((batchSize,y_train_depth[0].size))
            
            #Provide both images [time=t, time(t-1)]
            X_train=[X_train_1, X_train_2]
            #Provide depth and odom
            y_train=[y_train_depth, y_train_odom]
            
            #print('Train: '+str(y_train_depth.shape)+','+str(y_train_odom.shape))
            
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
            X_val_1=np.zeros((batchSize,192,640,3),dtype=np.uint8)   #time=t
            X_val_2=np.zeros((batchSize,192,640,3),dtype=np.uint8)   #time=t-1
            y_val_depth=np.zeros((batchSize,192,640),dtype=np.uint8)   #time=t depth
            y_val_odom=np.zeros((batchSize,6),dtype=np.float64)   #dt odom

            for i in range(batchSize):
                #Load images
                X_val_1[i]=rgb_read(X_val_filelist[idx+i])   #time=t
                X_val_2[i]=rgb_read(X_val_filelist[idx-1+i]) #time=t-1
                y_val_depth[i]=depth_read(y_val_filelist[idx+i])   #time=t

                #Calculate change in odometry
                current_filename=X_val_filelist[idx+i]   #time=t
                prev_filename=X_val_filelist[idx-1+i]   #time=t-1
                
                sequence_id, frame_id=basename(current_filename).split('_sync_')
                prev_sequence_id, prev_frame_id=basename(prev_filename).split('_sync_')
                
                #print('original:' + frame_id)
                frame_id=int(frame_id.split('.')[0])
                prev_frame_id=int(prev_frame_id.split('.')[0])
                
                #print('converted:' + str(frame_id))
                #prev_frame_id=frame_id-1
                
                current_odom=read_odom(sequence_id, frame_id)
                prev_odom=read_odom(prev_sequence_id, prev_frame_id)
                #print('Val: '+f'{frame_id}, {prev_frame_id}')
                
                y_val_odom[i]=normalize(current_odom-prev_odom)
                
            #Reshape [samples][width][height][pixels]
            X_val_1 = X_val_1.reshape(X_val_1.shape[0], X_val_1.shape[1], 
                                      X_val_1.shape[2], X_val_1.shape[3]).astype(np.uint8)
            X_val_2 = X_val_2.reshape(X_val_2.shape[0], X_val_2.shape[1], 
                                      X_val_2.shape[2], X_val_2.shape[3]).astype(np.uint8)
            
            y_val_depth = y_val_depth.reshape((y_val_depth.shape[0],1,-1)).astype(np.uint8)
            y_val_depth = y_val_depth.squeeze()
                 
            # normalize inputs and outputs from 0-255 to 0-1
            X_val_1=np.divide(X_val_1,255).astype(np.float16)
            X_val_2=np.divide(X_val_2,255).astype(np.float16)
            y_val_depth=np.divide(y_val_depth,255).astype(np.float16)
            
            #if (idx % 36)==0: #1024
            print(str(idx)+'/'+str(len(X_val_filelist)))
                
            idx+=batchSize
            
            #y_val_depth=y_val_depth.reshape((batchSize,len(y_val_depth)))
            y_val_depth=y_val_depth.reshape((batchSize,y_val_depth[0].size))
            
            #Provide both images [time=t, time(t-1)]
            X_val=[X_val_1, X_val_2]
            #Provide depth and odom
            y_val=[y_val_depth, y_val_odom]
            
            #print('Val: '+str(y_val_depth.shape)+','+str(y_val_odom.shape))
            
            yield X_val, y_val