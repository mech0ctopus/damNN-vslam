# -*- coding: utf-8 -*-
"""
Final Models.
"""
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Concatenate, Dense, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Input, Reshape, Dropout, LSTM
import segmentation_models

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

def cnn(input_shape=(192,640,4)):
	'''Define CNN model'''
	model = Sequential()
	model.add(Convolution2D(30, 5, 5, padding='valid',input_shape=input_shape, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, 3, 3, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))

	return model

def parallel_unets_with_tf(input_shape=(192,640,3)):
    '''Define Parallel U-Nets model.'''
    #Define input size
    input_1=Input(input_shape) #Image at time=t
    input_2=Input(input_shape) #Image at time=(t-1)
            
    #CURRENTLY IN USE                         
    # #Load unet with vgg backbone with no weights
    unet_1 = segmentation_models.Unet('resnet50', 
                                      input_shape=input_shape, 
                                      encoder_weights='imagenet',
                                      encoder_freeze=False)
    unet_2 = segmentation_models.Unet('resnet50',
                                      input_shape=input_shape, 
                                      encoder_weights='imagenet',
                                      encoder_freeze=False)
    
    # #Get final conv. output and skip sigmoid activation layer
    unet_1=Model(inputs=unet_1.input, outputs=unet_1.layers[-2].output,name='depth_output')
    unet_2=Model(inputs=unet_2.input, outputs=unet_2.layers[-2].output)
        
    #Run input through both unets
    unet_1_out=unet_1(input_1)
    unet_2_out=unet_2(input_2)
    
    #Merge unet outputs
    rbgd1=Concatenate()([input_1,unet_1_out])
    rbgd2=Concatenate()([input_2,unet_2_out])
    
    #Create transform branch for predicting rpy/xyz odom matrix/Networks for VO
    tf_cnn_t_0=cnn()(rbgd1) #t
    tf_cnn_t_1=cnn()(rbgd2) #t-1
    # tf_unet_t_0_flat=Flatten()(tf_cnn_t_0)
    # tf_unet_t_1_flat=Flatten()(tf_cnn_t_1)
    
    #Merge VO CNN ouputs
    merged2=Concatenate()([tf_cnn_t_0,tf_cnn_t_1])
    transform=Dense(6,activation='linear',name='vo_output')(merged2)
    
    #Define inputs and outputs    
    model = Model(inputs=[input_1,input_2], outputs=[unet_1_out,transform])
    
    return model

if __name__=='__main__':
    model=parallel_unets_with_tf()
    model.summary()
    plot_model(model, to_file='parallel_unets_with_tf.png', 
               show_shapes=True, 
               show_layer_names=False, 
               rankdir='TB',  #LR or TB for vertical or horizontal
               expand_nested=False, 
               dpi=96)