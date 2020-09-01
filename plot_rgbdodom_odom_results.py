# -*- coding: utf-8 -*-
"""
Compare predicted vs. ground truth RPYXYZ on validation data.
"""
from models import models
from tensorflow.keras.optimizers import Adam
from utils.read_odom import read_odom
from utils.deep_utils import rgb_read, depth_read
import numpy as np
from glob import glob
from os.path import basename
import matplotlib.pyplot as plt
from models.losses import undeepvo_rpy_mse, undeepvo_xyz_mse, root_mean_squared_error

def load_model():
    '''Load pretrained model.'''
    #Load model
    model=models.parallel_unets_with_odom()
    losses = {"rpy_output": root_mean_squared_error,
              "xyz_output": root_mean_squared_error}
    model.load_weights(r"C:\Users\craig\Documents\GitHub\damNN-vslam\parallel_unets_with_odom_weights_best.hdf5")
    model.compile(loss=losses,optimizer=Adam(0.001))
    return model

def normalize_image(image_path,depth=False):
    '''Normalize RGB or depth image.'''
    if depth:
        image=depth_read(image_path)
        image=image.reshape(1,192,640,1)
        image=np.divide(image,255).astype(np.float16)
    else:
        image=rgb_read(image_path)
        image=image.reshape(1,192,640,3)
        image=np.divide(image,255).astype(np.float16)
    return image
    
def predict_odom(image1_path,image2_path,image3_path,image4_path,prev_odom,model):
    '''Return predicted odom data.'''
    #Read test images
    image1=normalize_image(image1_path)
    image2=normalize_image(image2_path)
    image3=normalize_image(image3_path,depth=True)
    image4=normalize_image(image4_path,depth=True)
    #Predict relative odometry    
    # _,rpy_dt,xyz_dt=model.predict([image1,image2])
    # print('In predict odom')
    # print([image1,image2,image3,image4,prev_odom])
    rpy_dt,xyz_dt=model.predict([image1,image2,image3,image4,prev_odom])
    # rpyxyz_dt=model.predict(image1)
    #Denormalize
    rpyxyz_dt=np.array((rpy_dt,xyz_dt))
    # rpyxyz_dt=denormalize(rpyxyz_dt.reshape(6))
    rpyxyz_dt=rpyxyz_dt.reshape(6)
    return rpyxyz_dt

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
    image_list=glob(val_path+'X\\'+'*.png')
    depth_image_list=glob(val_path+'y\\'+'*.png')
    idx=1
    # predicted_results=np.zeros((len(image_list),6),dtype=np.float64)
    # actual_results=np.zeros((len(image_list),6),dtype=np.float64)
    predicted_results=np.zeros((500,6),dtype=np.float64)
    actual_results=np.zeros((500,6),dtype=np.float64)
    while idx<=500:
        image1, image2 = image_list[idx], image_list[idx-1]
        image3, image4=depth_image_list[idx], depth_image_list[idx-1]
        
        
        prev_odom=get_actual_odom(image_list[idx-1], image_list[idx-2])
        print(prev_odom)
        print(prev_odom.shape)
        predicted_results[idx-1]=predict_odom(image1,image2,image3,image4,
                                              prev_odom.flatten(),model)
        
        actual_results[idx-1]=get_actual_odom(image1,image2)
        print('Computed: ' + str(idx) +'/'+str(len(image_list)))
        idx+=1
    return predicted_results, actual_results

def plot_result(predicted_results,actual_results,idx,name):
    '''Plots result along one axis.'''
    plt.figure(idx)
    plt.plot(predicted_results[:,idx])
    plt.plot(actual_results[:,idx])
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
    ax.scatter(predicted_results[:,3], predicted_results[:,4], predicted_results[:,5])
    ax.scatter(actual_results[:,3], actual_results[:,4], actual_results[:,5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plot_overhead(predicted_results,actual_results):
    '''Plots overhead projection.'''
    from math import sin, cos
    predicted_pitch=predicted_results[:,1]
    predicted_z=predicted_results[:,5]
    
    actual_pitch=predicted_results[:,1]
    actual_z=predicted_results[:,5]

    actual_x_plot=[0]
    actual_y_plot=[0]
    predicted_x_plot=[0]
    predicted_y_plot=[0]    
    
    ii=1
    while ii<len(actual_pitch):
        predicted_x_plot.append(predicted_z[ii]*cos(predicted_pitch[ii])+predicted_x_plot[ii-1])
        predicted_y_plot.append(predicted_z[ii]*sin(predicted_pitch[ii])+predicted_x_plot[ii-1])
        
        actual_x_plot.append(actual_z[ii]*cos(actual_pitch[ii])+actual_x_plot[ii-1])
        actual_y_plot.append(actual_z[ii]*sin(actual_pitch[ii])+actual_x_plot[ii-1])
        ii+=1
        
    plt.figure(8)
    plt.title('Overhead: Predicted vs. Actual')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.plot(predicted_x_plot,predicted_y_plot)
    plt.plot(actual_x_plot,actual_y_plot)
    plt.legend(['Predicted', 'Actual'], loc='upper left')
    plt.show()

def mean_adjust_results(actual_results,predicted_results):
    #Adjust predicted data based on actual means
    actual_means=np.mean(actual_results,axis=0,dtype=np.float64)
    predicted_means=np.mean(predicted_results,axis=0,dtype=np.float64)
    #Initialize adjusted results
    adj_predicted_results=predicted_results
    #Adjust each parameter
    for i in range(6):
        adj_predicted_results[:,i]=adj_predicted_results[:,i]-predicted_means[i]+actual_means[i]
    return adj_predicted_results

def scale_match_results(actual_results,predicted_results):
    actual_maxes=np.max(actual_results,axis=0)
    actual_mins=np.min(actual_results,axis=0)
    pred_maxes=np.max(predicted_results,axis=0)
    pred_mins=np.min(predicted_results,axis=0)
    
    scaled_predicted_results=predicted_results
    scaled_actual_results=actual_results
    for i in range(6):
        scaled_predicted_results[:,i]=np.interp(scaled_predicted_results[:,i], (pred_mins[i],pred_maxes[i]), (0,1))
        scaled_actual_results[:,i]=np.interp(scaled_actual_results[:,i], (actual_mins[i],actual_maxes[i]), (0,1))
    return scaled_predicted_results, scaled_actual_results

def adjust_and_scale(actual_results,predicted_results):
    adj_scaled_predicted_results=mean_adjust_results(actual_results, predicted_results)
    adj_scaled_predicted_results, adj_scaled_actual_results=scale_match_results(actual_results, adj_scaled_predicted_results)
    return adj_scaled_predicted_results, adj_scaled_actual_results

if __name__=='__main__':
    model=load_model()
    val_path=r"data\val\\"
    predicted_results, actual_results=build_results(model, val_path)
    plot_results(predicted_results,actual_results)
    save_results=False
    
    # plot_overhead(predicted_results,actual_results)
    # plot_3d_path(predicted_results,actual_results)

    # # Plot mean-adjusted results    
    # adj_predicted_results=mean_adjust_results(actual_results,predicted_results)
    # plot_results(adj_predicted_results,actual_results)
    
    # Adjust predicted data based on scale
    # scaled_predicted_results,scaled_actual_results=scale_match_results(actual_results, predicted_results)
    # plot_results(scaled_predicted_results,scaled_actual_results)
    # plot_3d_path(scaled_predicted_results,scaled_actual_results)
    
    # p,a=adjust_and_scale(actual_results,predicted_results)
    # plot_results(p,a)
    
    if save_results:
        #Save data to text files
        np.savetxt(r'predicted_results_deepvo.txt',predicted_results,header='Roll, Pitch, Yaw, X, Y, Z')
        np.savetxt(r'actual_results.txt',actual_results,header='Roll, Pitch, Yaw, X, Y, Z')
        # Save adjusted results
        np.savetxt(r'adjusted_predicted_results_deepvo.txt',adj_predicted_results,header='Roll, Pitch, Yaw, X, Y, Z')
        #Save scale-adjusted results
        np.savetxt(r'scaled_predicted_results_deepvo.txt',adj_predicted_results,header='Roll, Pitch, Yaw, X, Y, Z')