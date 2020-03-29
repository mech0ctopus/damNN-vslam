# -*- coding: utf-8 -*-
"""
Utility functions for working with neural networks.
"""
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def save_model(model,serialize_type,model_name='model',save_weights=False):
    '''Saves model and weights to file.'''
    serialize_type=serialize_type.lower()
    
    if serialize_type=='yaml':
        model_yaml = model.to_yaml()
        with open(model_name+".yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
    elif serialize_type=='json':
        model_json = model.to_json()
        with open(model_name+".json", "w") as json_file:
            json_file.write(model_json)
    if save_weights:
        model.save_weights(model_name+".h5")
        print(model_name+' & weights saved to disk.')
    else:
        print(model_name+' saved to disk.')
        
def simul_shuffle(mat1, mat2):
    '''Shuffles two matrices in the same order'''
    
    if type(mat1)==list:
        temp = list(zip(mat1, mat2)) 
        random.shuffle(temp) 
        mat1, mat2 = zip(*temp)
    else:
        idx=np.arange(0,mat1.shape[0])   
        random.shuffle(idx)
        mat1=mat1[idx]
        mat2=mat2[idx]
    return mat1, mat2

def rgb_read(filename):
    '''Reads RGB image from png file and returns it as a numpy array'''
    #Load image
    image=Image.open(filename)
    #store as np.array
    rgb=np.array(image)
    image.close()
    return rgb

def depth_read(filename):
    '''Loads depth map D from png file and returns it as a numpy array'''
    #Lower is closer
    # From KITTI devkit
    
    image=Image.open(filename)
    depth_png = np.array(image, dtype=int)

    if depth_png.shape==(375,1242,3):
        depth_png=(depth_png[:,:,0]+depth_png[:,:,1]+depth_png[:,:,2])/3
    
    #Convert to 8 bit
    depth_png=depth_png/(2**8)
    
    assert(np.max(depth_png) <= 255)
    depth=depth_png.astype('int8') #np.float
 
    image.close()

    return depth

def heatmap(image,save=False,name='heatmap',cmap='gray',show=True):
    '''Plots heatmap of depth data from image or np.ndarray.'''
    if type(image)==np.ndarray:
        pic_array=image
    else:
        #Convert to np.ndarray
        pic=Image.open(image)
        pic_array=np.array(pic)
    #Plot heatmap
    plt.imshow(pic_array, cmap=cmap, interpolation='nearest') #cmap=binary, plasma, gray
    if show==True:
        plt.show()
    if save==True:
        plt.imsave(name+'.png',pic_array, cmap=cmap)