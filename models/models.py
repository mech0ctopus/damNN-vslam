# -*- coding: utf-8 -*-
"""
Final Models.
"""
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Concatenate, Dense, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Input, Reshape, Dropout, LSTM
import segmentation_models
from tensorflow.keras.layers import PReLU

segmentation_models.set_framework('tf.keras')

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

def DenseBlock(input_shape = (4320,1)):
    model = Sequential()
    model.add(Dropout(0.5))
    model.add(Dense(288, input_shape = input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(288, activation='relu'))
    model.add(Dropout(0.5))
    #New
    model.add(Dense(288, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5)) 
    model.add(Dense(288, activation='relu'))
    model.add(Dropout(0.5))
    #End New
    model.add(Dense(144, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(72, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(72, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(36, activation='relu'))
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

def parallel_unets_with_tf(input_shape=(192,640,3)):
    '''Define Parallel U-Nets model.'''
    #Define input size
    input_1=Input(input_shape) #Image at time=t
    input_2=Input(input_shape) #Image at time=(t-1)
                                    
    # #Load unet with backbone with no imagnet weights
    # unet_1 = segmentation_models.Unet('resnet50', #resnet50, mobilenetv2
    #                                   input_shape=input_shape, 
    #                                   encoder_weights='imagenet', #None
    #                                   encoder_freeze=False)
    wnet_c=wnet_connected()
    
    #Get final conv. output and skip sigmoid activation layer
    # unet_1=Model(inputs=unet_1.input, outputs=unet_1.layers[-2].output)
    
    #Load unet weights from depth-only pretraining
    wnet_c.load_weights(r"C:\Users\Craig\Documents\GitHub\depth-estimation\W-Net_Connected_weights_best_KITTI_35Epochs.hdf5")
    for layer in wnet_c.layers:
        layer.trainable=False
    
    #Run input through both unets
    wnet_c_1_out=wnet_c(input_1)
    wnet_c_2_out=wnet_c(input_2)

    depth_out=Flatten(name='depth_output')(wnet_c_1_out)
    
    wnet_c_1_out=Reshape((192,640,1))(wnet_c_1_out)
    wnet_c_2_out=Reshape((192,640,1))(wnet_c_2_out)
    
    #Merge unet outputs
    rbgd1=Concatenate()([input_1,wnet_c_1_out])
    rbgd2=Concatenate()([input_2,wnet_c_2_out])
    
    #Create transform branch for predicting rpy/xyz odom matrix/Networks for VO
    tf_cnn_t_1=cnn4()(rbgd1) #t
    tf_cnn_t_2=cnn4()(rbgd2) #t-1
    
    flatten1=Flatten()(tf_cnn_t_1)
    flatten2=Flatten()(tf_cnn_t_2)
    dense_block1=DenseBlock(input_shape=(flatten1.shape[0],1))(flatten1)
    dense_block2=DenseBlock(input_shape=(flatten2.shape[0],1))(flatten2)

    #Merge VO CNN ouputs
    #merged2=Concatenate()([tf_cnn_t_1,tf_cnn_t_2])
    merged2=Concatenate()([dense_block1,dense_block2])
    # flatten2=Flatten()(merged2)
    # print(flatten2.shape[0])
    # dense_block1=DenseBlock(input_shape=(flatten2.shape[0],1))(flatten2)
    
    #reshape=Reshape((2,tf_cnn_t_1.shape[1]))(merged2)
    reshape=Reshape((2,dense_block1.shape[1]))(merged2)
    lstm1=LSTM(512,return_sequences=True)(reshape) #128
    lstm2=LSTM(512,return_sequences=False)(lstm1) #256
    
    dense2=Dense(128, activation=PReLU())(lstm2)
    transform=Dense(6,activation='linear',name='vo_output')(dense2) #RPYXYZ
    
    #Define inputs and outputs    
    model = Model(inputs=[input_1,input_2], outputs=[depth_out,transform])
    model.layers[2].trainable=False
    
    return model

if __name__=='__main__':
    model=parallel_unets_with_tf()
    print(model)
    model.summary()
    plot_model(model, to_file='parallel_unets_with_tf.png', 
                show_shapes=True, 
                show_layer_names=False, 
                rankdir='TB',  #LR or TB for vertical or horizontal
                expand_nested=False, 
                dpi=96)