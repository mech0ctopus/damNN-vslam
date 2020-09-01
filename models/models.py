# -*- coding: utf-8 -*-
"""
Final Models.
"""
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Concatenate, Dense, Convolution2D, Conv3D
from tensorflow.keras.layers import Flatten, Input, Reshape, Dropout, LSTM
import segmentation_models
from tensorflow.keras.layers import PReLU
import numpy as np
import cv2 as cv
# from models import se3 
import tensorflow as tf

segmentation_models.set_framework('tf.keras')

def get_optical_flow(frame1,frame2):
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY) 
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    return bgr

def unet(input_shape=(192,640,3)):
    '''Define U-Net model.'''
    #Load unet with resnet34 backbone with no weights
    premodel = segmentation_models.Unet('vgg16', 
                                        input_shape=input_shape, 
                                        encoder_weights=None,
                                        encoder_freeze=False)
    
    #Get final conv. output and skip sigmoid activation layer
    x=premodel.layers[-2].output 
    reshape=Reshape((input_shape[0]*input_shape[1],))(x)
    model = Model(inputs=premodel.input, outputs=reshape)

    return model

def parallel_unets(input_shape=(192,640,3)): #375,1242 TODO: Need to update dim input
    '''Define Parallel U-Nets model.'''
    #Define input size
    input_1=Input(input_shape) #Image at time=t
    input_2=Input(input_shape) #Image at time=(t-1)
                                     
    #Load unet with vgg backbone with no weights
    unet_1 = segmentation_models.Unet('vgg16', 
                                      input_shape=input_shape, 
                                      encoder_weights=None,
                                      encoder_freeze=False)
    unet_2 = segmentation_models.Unet('vgg16',  #
                                      input_shape=input_shape, 
                                      encoder_weights=None,
                                      encoder_freeze=False)
    
    #Get final conv. output and skip sigmoid activation layer
    unet_1=Model(inputs=unet_1.input, outputs=unet_1.layers[-2].output)
    unet_2=Model(inputs=unet_2.input, outputs=unet_2.layers[-2].output)
    
    #Run input through both unets
    unet_1_out=unet_1(input_1)
    unet_2_out=unet_2(input_2)
    
    #Merge unet outputs
    merged=Concatenate()([unet_1_out,unet_2_out])
    #Reduce outputs from U-Nets
    flatten=Flatten()(merged)
    #Add dense layers
    dense1=Dense(16,activation='relu')(flatten)
    final_output=Dense(input_shape[0]*input_shape[1],activation='linear')(dense1)
    
    #Define inputs and outputs    
    model = Model(inputs=[input_1,input_2], outputs=final_output)

    return model

def cnn4(input_shape=(192,640,4)):
    '''Define CNN model'''
    model = Sequential()
    model.add(Convolution2D(filters=64, kernel_size=7, strides=(2,2), padding='valid', input_shape=input_shape,activation=PReLU()))
    model.add(Convolution2D(filters=128, kernel_size=5, strides=(2,2), padding='same', activation=PReLU()))
    model.add(Convolution2D(filters=256, kernel_size=5, strides=(2,2), padding='same',activation=PReLU()))
    model.add(Convolution2D(filters=256, kernel_size=3, strides=(1,1), padding='same',activation=PReLU()))
    model.add(Convolution2D(filters=512, kernel_size=3, strides=(2,2), padding='same',activation=PReLU()))
    model.add(Convolution2D(filters=512, kernel_size=3, strides=(1,1), padding='same',activation=PReLU()))
    model.add(Convolution2D(filters=512, kernel_size=3, strides=(2,2), padding='same',activation=PReLU()))
    model.add(Convolution2D(filters=512, kernel_size=3, strides=(1,1), padding='same',activation=PReLU()))
    model.add(Convolution2D(filters=1024, kernel_size=3, strides=(2,2), padding='same',activation=PReLU()))
    model.add(Flatten())

    return model

def cnn_3d(input_shape=(2,192,640,4)):
    '''Define CNN model'''
    model = Sequential()
    model.add(Conv3D(filters=64, kernel_size=7, strides=(2,2,2), padding='valid', input_shape=input_shape,activation=PReLU()))
    model.add(Conv3D(filters=128, kernel_size=5, strides=(2,2,2), padding='same', activation=PReLU()))
    model.add(Conv3D(filters=256, kernel_size=5, strides=(2,2,2), padding='same',activation=PReLU()))
    model.add(Conv3D(filters=256, kernel_size=3, strides=(1,1,1), padding='same',activation=PReLU()))
    model.add(Conv3D(filters=512, kernel_size=3, strides=(2,2,2), padding='same',activation=PReLU()))
    model.add(Conv3D(filters=512, kernel_size=3, strides=(1,1,1), padding='same',activation=PReLU()))
    model.add(Conv3D(filters=512, kernel_size=3, strides=(2,2,2), padding='same',activation=PReLU()))
    model.add(Conv3D(filters=512, kernel_size=3, strides=(1,1,1), padding='same',activation=PReLU()))
    model.add(Conv3D(filters=1024, kernel_size=3, strides=(2,2,2), padding='same',activation=PReLU()))
    # model.add(Flatten())

    return model

def DenseBlock(input_shape = (4320,1)):
    model = Sequential()
    model.add(Dropout(0.5))
    model.add(Dense(512, input_shape = input_shape, activation=PReLU()))
    model.add(Dropout(0.5)) 
    model.add(Dense(512, activation=PReLU()))
    model.add(Dropout(0.5)) 
    model.add(Dense(512, activation=PReLU()))
    model.add(Dropout(0.5)) 
    model.add(Dense(512, activation=PReLU()))
    model.add(Dropout(0.5)) 
    return model

def wnet_connected():   
    #Load unet with resnet34 backbone. (densenet201,resnet34,vgg16,resnet18,resnet152)
    firstU = segmentation_models.Unet('resnet50', input_shape=(192, 640, 3), encoder_weights='imagenet',encoder_freeze = True)
    secondU = segmentation_models.Unet('resnet50', input_shape=(192, 640, 4), encoder_weights=None)
    #Get final conv. output and keep sigmoid activation layer
    firstU = Model(inputs=firstU.input, outputs=firstU.layers[-1].output)
    #Get final conv. output and skip sigmoid activation layer
    secondU=Model(inputs=secondU.input, outputs=secondU.layers[-2].output) 

    for layer in secondU.layers:
        layer.trainable = True
    
    inputs = Input((192, 640, 3))
    m1=firstU(inputs)
    merged=Concatenate()([inputs,m1])
    reshape1=Reshape((192, 640, 4))(merged)
    m2=secondU(reshape1)
    reshape2=Reshape((192*640,))(m2)
    
    wnet_c=Model(inputs=inputs,outputs=reshape2)
    
    wnet_c.layers[2].trainable=True #Concat
    wnet_c.layers[3].trainable=True #Reshape
    wnet_c.layers[4].trainable=True #Second U
    wnet_c.layers[5].trainable=True #Reshape
    
    return wnet_c

def cnn_new(input_shape=(192,640,6)):
    '''Define CNN model'''
    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=7, strides=(2,2), padding='valid', input_shape=input_shape,activation=PReLU()))
    model.add(Convolution2D(filters=32, kernel_size=5, strides=(2,2), padding='same', activation=PReLU()))
    model.add(Convolution2D(filters=64, kernel_size=5, strides=(2,2), padding='same',activation=PReLU()))
    model.add(Convolution2D(filters=128, kernel_size=3, strides=(1,1), padding='same',activation=PReLU()))
    model.add(Convolution2D(filters=256, kernel_size=3, strides=(2,2), padding='same',activation=PReLU()))
    model.add(Convolution2D(filters=256, kernel_size=3, strides=(1,1), padding='same',activation=PReLU()))
    model.add(Convolution2D(filters=512, kernel_size=3, strides=(2,2), padding='same',activation=PReLU()))
    model.add(Flatten())

    return model

def parallel_unets_with_tf(input_shape=(192,640,3)):
    '''Define Parallel U-Nets model.'''
    #Define inputs
    rgb_input_1=Input(input_shape,name='input1') #RGB Image at time=t
    rgb_input_2=Input(input_shape,name='input2') #RGB Image at time=(t-1)
    d_input_1=Input((192,640,1),name='input3') #Depth Image at time=t
    d_input_2=Input((192,640,1),name='input4') #Depth Image at time=(t-1)
    d1_reshape=Reshape((192,640,1))(d_input_1)
    d2_reshape=Reshape((192,640,1))(d_input_2)
        
    #Build RGBD
    rgbd1=Concatenate()([rgb_input_1,d1_reshape]) #RGBD Image at time=t
    rgbd2=Concatenate()([rgb_input_2,d2_reshape]) #RGBD Image at time=t-1
    
    cnn1=cnn4()(rgbd1)
    cnn2=cnn4()(rgbd2)
    flatten1=Flatten()(cnn1)
    flatten2=Flatten()(cnn2)
    dense_block1=DenseBlock(input_shape=(flatten1.shape[0],1))(flatten1)
    dense_block2=DenseBlock(input_shape=(flatten2.shape[0],1))(flatten2)
    merged=Concatenate()([dense_block1,dense_block2])
    flatten3=Flatten()(merged)
    
    dense2=Dense(128,activation=PReLU())(flatten3)
    rpy_output=Dense(3,activation='linear',name='rpy_output')(dense2) #RPY
 
    dense4=Dense(128,activation=PReLU())(flatten3)
    xyz_output=Dense(3,activation='linear',name='xyz_output')(dense4) #XYZ
    
    #Define inputs and outputs    
    model = Model(inputs=[rgb_input_1,rgb_input_2,d_input_1,d_input_2], 
                  outputs=[rpy_output,xyz_output])
    
    return model

def parallel_unets_with_odom(input_shape=(192,640,3)):
    '''Define Parallel U-Nets model.'''
    #Define inputs
    rgb_input_1=Input(input_shape,name='input1') #RGB Image at time=t
    rgb_input_2=Input(input_shape,name='input2') #RGB Image at time=(t-1)
    d_input_1=Input((192,640,1),name='input3') #Depth Image at time=t
    d_input_2=Input((192,640,1),name='input4') #Depth Image at time=(t-1)
    prev_odom_input=Input((6),name='input5') #Odom between time=(t-1) and (t-2)
    d1_reshape=Reshape((192,640,1))(d_input_1)
    d2_reshape=Reshape((192,640,1))(d_input_2)
        
    #Build RGBD
    rgbd1=Concatenate()([rgb_input_1,d1_reshape]) #RGBD Image at time=t
    rgbd2=Concatenate()([rgb_input_2,d2_reshape]) #RGBD Image at time=t-1
    
    cnn1=cnn4()(rgbd1)
    cnn2=cnn4()(rgbd2)
    flatten1=Flatten()(cnn1)
    flatten2=Flatten()(cnn2)
    
    flatten1_with_odom=Concatenate()([flatten1,prev_odom_input])
    flatten2_with_odom=Concatenate()([flatten2,prev_odom_input])
    
    dense_block1=DenseBlock(input_shape=(flatten1_with_odom.shape[0],1))(flatten1_with_odom)
    dense_block2=DenseBlock(input_shape=(flatten2_with_odom.shape[0],1))(flatten2_with_odom)
    
    merged=Concatenate()([dense_block1,dense_block2])
    flatten3=Flatten()(merged)
    
    dense2=Dense(128,activation=PReLU())(flatten3)
    rpy_output=Dense(3,activation='linear',name='rpy_output')(dense2) #RPY
 
    dense4=Dense(128,activation=PReLU())(flatten3)
    xyz_output=Dense(3,activation='linear',name='xyz_output')(dense4) #XYZ
    
    #Define inputs and outputs    
    model = Model(inputs=[rgb_input_1,rgb_input_2,d_input_1,d_input_2,
                          prev_odom_input], 
                  outputs=[rpy_output,xyz_output])
    
    return model

def mock_deepvo(input_shape=(192,640,3)):
    '''Replicate DeepVO model.'''
    #Define input size
    input_1=Input(input_shape) #Image at time=t
    input_2=Input(input_shape) #Image at time=(t-1)
    #Stack input images
    stacked_images=Concatenate()([input_1,input_2])
    
    #Pass through CNN together (Is the padding the same as deepvo?)
    #Should this be pre-trained from flownet?
    cnn_out=cnn4(input_shape=(192,640,6))(stacked_images)
    #Should this be reshaped like (3,10,1024)? or (1,3*10*1024) ?
    reshaped_cnn_out=Reshape((1,3*10*1024))(cnn_out)
    
    #Pass through LSTM layers
    lstm1=LSTM(512,return_sequences=True)(reshaped_cnn_out) #Should be 1000
    lstm2=LSTM(512,return_sequences=False)(lstm1) #Should be 1000
    
    rpyxyz_output=Dense(6,activation='linear',name='rpyxyz_output')(lstm2)
    
    #Define inputs and output
    model = Model(inputs=[input_1,input_2], outputs=rpyxyz_output)
    
    return model

def mock_undeepvo(input_shape=(192,640,3)):
    '''Replicates RGB pose estimate portion of UnDeepVO model.'''
    #Define input size
    input_1=Input(input_shape) #Image at time=t
    input_2=Input(input_shape) #Image at time=(t-1)
    #Stack input images
    stacked_images=Concatenate()([input_1,input_2])
    
    #Pass through CNN together (Is the padding the same as deepvo?)
    #Should this be pre-trained from flownet?
    cnn_out=cnn_new(input_shape=(192,640,6))(stacked_images)
    
    rpy_dense1=Dense(512,activation='relu')(cnn_out)
    rpy_dense2=Dense(512,activation='relu')(rpy_dense1)
     
    xyz_dense1=Dense(512,activation='relu')(cnn_out)
    xyz_dense2=Dense(512,activation='relu')(xyz_dense1)
    
    rpy_output=Dense(3,activation='linear',name='rpy_output')(rpy_dense2)
    xyz_output=Dense(3,activation='linear',name='xyz_output')(xyz_dense2)
    
    #Define inputs and output
    model = Model(inputs=[input_1,input_2], outputs=[rpy_output,xyz_output])
    
    return model

def mock_deepvo2(input_shape=(192,640,3)):
    '''Replicate DeepVO model.'''
    #Define input size
    input_1=Input(input_shape) #Image at time=t
    input_2=Input(input_shape) #Image at time=(t-1)
    
    #Pass through CNN separately (Is the padding the same as deepvo?)
    #Should this be pre-trained from flownet?
    # cnn_out_1=cnn4(input_shape=(192,640,3))(input_1)
    # cnn_out_2=cnn4(input_shape=(192,640,3))(input_2)
    
    #Concatenate CNN outputs
    # cnn_outputs=Concatenate()([cnn_out_1,cnn_out_2])
    # reshaped_cnn_outputs=Reshape((2,192,640,3))(cnn_outputs)
    
    frame1=cv.imread(input_2) #t-1
    frame2=cv.imread(input_1) #t
    test=get_optical_flow(frame1, frame2)
    #Pass through LSTM layers
    lstm1=LSTM(512,return_sequences=True)(test) #Should be 1000
    lstm2=LSTM(512,return_sequences=False)(lstm1) #Should be 1000
    
    rpyxyz_output=Dense(6,activation='linear',name='rpyxyz_output')(lstm2)
    
    #Define inputs and output
    model = Model(inputs=[input_1,input_2], outputs=rpyxyz_output)
    
    return model

def mock_undeepvo_withflow(input_shape=(192,640,3)):
    '''Replicates RGB pose estimate portion of UnDeepVO model.'''
    #Define input size
    input_1=Input(input_shape) #Image at time=t
    input_2=Input(input_shape) #Image at time=(t-1)
    input_3=Input(input_shape) #Optical Flow Image
    #Stack input images
    stacked_images=Concatenate()([input_1,input_2,input_3])
    
    cnn_out=cnn_new(input_shape=(192,640,9))(stacked_images)
    
    rpy_dense1=Dense(512,activation='relu')(cnn_out)
    rpy_dense2=Dense(512,activation='relu')(rpy_dense1)
     
    xyz_dense1=Dense(512,activation='relu')(cnn_out)
    xyz_dense2=Dense(512,activation='relu')(xyz_dense1)
    
    rpy_output=Dense(3,activation='linear',name='rpy_output')(rpy_dense2)
    xyz_output=Dense(3,activation='linear',name='xyz_output')(xyz_dense2)
    
    #Define inputs and output
    model = Model(inputs=[input_1,input_2,input_3], outputs=[rpy_output,xyz_output])
    
    return model

def lvo_cnn(input_shape=(192,640,3)):
    '''Define CNN model'''
    model = Sequential()
    model.add(Convolution2D(filters=64, kernel_size=3, strides=(2,2), 
                            padding='valid', input_shape=input_shape,
                            activation='relu'))
    model.add(Convolution2D(filters=128, kernel_size=3, strides=(2,2), 
                            padding='same',activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=3, strides=(2,2), 
                            padding='same',activation='relu'))
    model.add(Convolution2D(filters=512, kernel_size=3, strides=(2,2), 
                            padding='same',activation='relu'))
    model.add(Flatten())

    return model

def esp_cnn(input_shape=(192,640,4)):
    '''Define CNN model for ESP-VO'''
    model = Sequential()
    model.add(Convolution2D(filters=64, kernel_size=7, strides=(2,2), padding='valid', input_shape=input_shape,activation='relu'))
    model.add(Convolution2D(filters=128, kernel_size=5, strides=(2,2), padding='same', activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=5, strides=(2,2), padding='same',activation='relu'))
    model.add(Convolution2D(filters=256, kernel_size=3, strides=(1,1), padding='same',activation='relu'))
    model.add(Convolution2D(filters=512, kernel_size=3, strides=(2,2), padding='same',activation='relu'))
    model.add(Convolution2D(filters=512, kernel_size=3, strides=(1,1), padding='same',activation='relu'))
    model.add(Convolution2D(filters=512, kernel_size=3, strides=(2,2), padding='same',activation='relu'))
    model.add(Convolution2D(filters=512, kernel_size=3, strides=(1,1), padding='same',activation='relu'))
    model.add(Convolution2D(filters=1024, kernel_size=3, strides=(2,2), padding='same',activation='relu'))
    model.add(Flatten())

    return model

def vo_from_flow(input_shape=(192,640,3)):
    '''Replicate UnDeepVO model.'''
    #Define input size
    input_1=Input(input_shape) #Optical Flow between t and t-1
    cnn_out=esp_cnn(input_shape)(input_1)

    rpy_dense1=Dense(256,activation='relu')(cnn_out)
    dropout1=Dropout(0.5)(rpy_dense1)
    rpy_dense2=Dense(256,activation='relu')(dropout1)
    dropout2=Dropout(0.5)(rpy_dense2)
    rpy_dense3=Dense(256,activation='relu')(dropout2)
    dropout3=Dropout(0.5)(rpy_dense3)
    rpy_dense4=Dense(256,activation='relu')(dropout3)
    dropout4=Dropout(0.5)(rpy_dense4)
    rpy_dense5=Dense(256,activation='relu')(dropout4)
    dropout5=Dropout(0.5)(rpy_dense5)
     
    xyz_dense1=Dense(256,activation='relu')(cnn_out)
    dropout6=Dropout(0.5)(xyz_dense1)
    xyz_dense2=Dense(256,activation='relu')(dropout6)
    dropout7=Dropout(0.5)(xyz_dense2)
    xyz_dense3=Dense(256,activation='relu')(dropout7)
    dropout8=Dropout(0.5)(xyz_dense3)
    xyz_dense4=Dense(256,activation='relu')(dropout8)
    dropout9=Dropout(0.5)(xyz_dense4)
    xyz_dense5=Dense(256,activation='relu')(dropout9)
    dropout10=Dropout(0.5)(xyz_dense5)
    
    #Stack input images
    stacked_images=Concatenate(name='concat')([dropout5,dropout10])
    dense=Dense(6,activation='relu',name='dense2')(stacked_images)
    se3_layer=se3.SE3CompositeLayer()(dense)
    se3_layer.trainable=False
    
    rpy_output=Dense(3,activation='linear',name='rpy_output')(rpy_dense2)
    xyz_output=Dense(3,activation='linear',name='xyz_output')(xyz_dense2)
    
    #Define inputs and output
    model = Model(inputs=input_1, outputs=[rpy_output,xyz_output])
    
    return model

def depth_only(input_shape=(192,640,1)):
    '''Get relative RPYXYZ from 2x Depth Images.'''
    #Define input size
    input_1=Input(input_shape,name='input1') #Depth Image at time=t
    input_2=Input(input_shape,name='input2') #Depth Image at time=(t-1)
    #Stack input images
    stacked_images=Concatenate(name='concat')([input_1,input_2])
    cnn_out=esp_cnn(input_shape=(input_shape[0],input_shape[1],2*input_shape[2]))(stacked_images)

    dense1=Dense(128,activation='relu',name='dense1')(cnn_out)
    dense2=Dense(6,activation='relu',name='dense2')(dense1)

    se3_layer=se3.SE3CompositeLayer()(dense2)
    se3_layer.trainable=False

    rpy_out=Dense(3,activation='linear',name='rpy_output',dtype=tf.float32)(se3_layer)
    xyz_out=Dense(3,activation='linear',name='xyz_output',dtype=tf.float32)(se3_layer)
    
    #Define inputs and output
    model = Model(inputs=[input_1,input_2], outputs=[rpy_out,xyz_out])
    
    return model

def mock_espvo(input_shape=(192,640,3)):
    '''Replicate ESP-VO model.'''
    #Define input size
    input_1=Input(input_shape,name='input1') #Image at time=t
    input_2=Input(input_shape,name='input2') #Image at time=(t-1)
    #Stack input images
    stacked_images=Concatenate(name='concat')([input_1,input_2])
    
    #Pass through CNN together
    cnn_out=esp_cnn(input_shape=(input_shape[0],input_shape[1],2*input_shape[2]))(stacked_images)
    reshaped_cnn_out=Reshape((1,3*10*1024))(cnn_out)
    
    #Pass through LSTM layers
    lstm1=LSTM(128,return_sequences=True,name='lstm1')(reshaped_cnn_out) #Should be 1000
    lstm2=LSTM(128,return_sequences=False,name='lstm2')(lstm1) #Should be 1000
    
    dense1=Dense(128,activation='relu',name='dense1')(lstm2)
    dense2=Dense(6,activation='relu',name='dense2')(dense1)

    se3_layer=se3.SE3CompositeLayer()(dense2)
    se3_layer.trainable=False

    rpy_out=Dense(3,activation='linear',name='rpy_output',dtype=tf.float32)(se3_layer)
    xyz_out=Dense(3,activation='linear',name='xyz_output',dtype=tf.float32)(se3_layer)
    
    #Define inputs and output
    model = Model(inputs=[input_1,input_2], outputs=[rpy_out,xyz_out])
    
    return model

def mock_espvo_rgbd(input_shape=(192,640,3)):
    '''Replicate ESP-VO RGBD model.'''
    #Define input size
    rgb_input_1=Input(input_shape,name='input1') #RGB Image at time=t
    rgb_input_2=Input(input_shape,name='input2') #RGB Image at time=(t-1)
    d_input_1=Input((192,640,1),name='input3') #Depth Image at time=t
    d_input_2=Input((192,640,1),name='input4') #Depth Image at time=(t-1)
    d1_reshape=Reshape((192,640,1))(d_input_1)
    d2_reshape=Reshape((192,640,1))(d_input_2)
    #Stack input images
    stacked_images=Concatenate(name='concat')([rgb_input_1,rgb_input_2,
                                               d1_reshape,d2_reshape])
    
    cnn_out=esp_cnn(input_shape=(input_shape[0],input_shape[1],8))(stacked_images)
    reshaped_cnn_out=Reshape((1,3*10*1024))(cnn_out)
    
    #Pass through LSTM layers
    lstm1=LSTM(128,return_sequences=True,name='lstm1')(reshaped_cnn_out) #Should be 1000
    lstm2=LSTM(128,return_sequences=False,name='lstm2')(lstm1) #Should be 1000
    
    dense1=Dense(128,activation='relu',name='dense1')(lstm2)
    dense2=Dense(6,activation='relu',name='dense2')(dense1)

    se3_layer=se3.SE3CompositeLayer()(dense2)
    se3_layer.trainable=False

    rpy_out=Dense(3,activation='linear',name='rpy_output',dtype=tf.float32)(se3_layer)
    xyz_out=Dense(3,activation='linear',name='xyz_output',dtype=tf.float32)(se3_layer)
    
    #Define inputs and output
    model = Model(inputs=[rgb_input_1,rgb_input_2,d_input_1,d_input_2], 
                  outputs=[rpy_out,xyz_out])
    
    return model

if __name__=='__main__':
    model=parallel_unets_with_odom()
    model.summary()
    plot_model(model, to_file='parallel_unets_with_odom.png', 
                show_shapes=True, 
                show_layer_names=False, 
                rankdir='TB',  #LR or TB for vertical or horizontal
                expand_nested=False, 
                dpi=96)