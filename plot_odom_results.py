# -*- coding: utf-8 -*-
"""
Compare predicted vs. ground truth RPYXYZ on validation data.
"""
from models import models
from tensorflow.keras.optimizers import Adam
from utils.read_odom import read_odom, denormalize
from utils.deep_utils import rgb_read
import numpy as np
from glob import glob
from os.path import basename
import matplotlib.pyplot as plt

def load_model():
    '''Load pretrained model.'''
    #Load model
    model=models.parallel_unets_with_tf()
    model_name='parallel_unets_with_tf'
    losses = {'depth_output': 'mean_squared_error',
              "vo_output": 'mean_squared_logarithmic_error'}
    model.load_weights(f"{model_name}_weights_best.hdf5")
    model.compile(loss=losses,optimizer=Adam(lr=5e-6))
    return model

def normalize_image(image_path):
    '''Normalize RGB image.'''
    image=rgb_read(image_path)
    image=image.reshape(1,192,640,3)
    image=np.divide(image,255).astype(np.float16)
    return image
    
def predict_odom(image1_path,image2_path,model):
    '''Return predicted odom data.'''
    #Read test images
    image1=normalize_image(image1_path)
    image2=normalize_image(image2_path)
    #Predict relative odometry    
    _,odom_dt=model.predict([image1,image2])
    #Denormalize
    odom_dt=denormalize(odom_dt.reshape(6))
    #odom_dt=odom_dt.reshape(6)
    return odom_dt

def parse_path(image_path):
    '''Get sequence ID and frame # from image path'''
    base=basename(image_path)
    filename=base.split('.')[0]
    sequence_id, frame = filename.split('_sync_')
    frame=int(frame)
    return sequence_id, frame

def get_actual_odom(image1_path,image2_path):
    '''Get actual relative odometry.'''
    image1_seq_id, image1_frame=parse_path(image1_path)
    image2_seq_id, image2_frame=parse_path(image2_path)
    
    image1_odom=read_odom(sequence_id=image1_seq_id,desired_frame=image1_frame)
    image2_odom=read_odom(sequence_id=image2_seq_id,desired_frame=image2_frame)
    
    odom_dt_actual=image1_odom-image2_odom

    return odom_dt_actual

def build_results(model, val_path):
    image_list=glob(val_path+'*.png')
    idx=1
    predicted_results=np.zeros((len(image_list),6),dtype=np.float64)
    actual_results=np.zeros((len(image_list),6),dtype=np.float64)
    while idx<101: #len(image_list):
        image1, image2 = image_list[idx], image_list[idx-1]
        predicted_results[idx-1]=predict_odom(image1,image2,model)
        actual_results[idx-1]=get_actual_odom(image1,image2)
        print('Computed: ' + str(idx) +'/'+str(len(image_list)))
        idx+=1
    return predicted_results, actual_results

def plot_result(predicted_results,actual_results,idx,name):
    '''Plots result along one axis.'''
    plt.figure(idx)
    plt.plot(predicted_results[0:100,idx])
    plt.plot(actual_results[0:100,idx])
    plt.title(f'{name}: Predicted vs. Actual')
    if name.lower() in ['roll','pitch','yaw']:
        plt.ylabel('Radians')
    else:
        plt.ylabel('Meters')
    plt.xlabel('Starting Frame #')
    plt.legend(['Predicted', 'Actual'], loc='upper left')
    plt.show()

def plot_results(predicted_results,actual_results):
    '''Plots results for all parameters.'''
    names=['Roll','Pitch','Yaw','X','Y','Z']
    for idx, name in enumerate(names):
        plot_result(predicted_results,actual_results,idx,name)
    
def plot_3d_path(predicted_results,actual_results):
    '''Plots 3D Vehicle Path.'''
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(7)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(predicted_results[:,0], predicted_results[:,1], predicted_results[:,2])
    ax.scatter(actual_results[:,0], actual_results[:,1], actual_results[:,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    
if __name__=='__main__':
    model=load_model()
    val_path=r"G:\Documents\KITTI\data\val\X\\"
    predicted_results, actual_results=build_results(model, val_path)
    plot_results(predicted_results,actual_results)
    plot_3d_path(predicted_results,actual_results)
    