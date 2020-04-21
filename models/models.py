# -*- coding: utf-8 -*-
"""
Final Models.
"""
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Concatenate, Dense, Convolution2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Input, Reshape, Dropout, LSTM
import segmentation_models
from tensorflow.keras.applications import vgg19

segmentation_models.set_framework('tf.keras')

def vgg19_convs(input_shape=(192,640,4)):
    premodel=vgg19.VGG19(include_top=False, 
                          weights='imagenet', 
                          input_tensor=None, 
                          input_shape=input_shape, 
                          pooling=None, 
                          classes=1000)

    #Get output of flatten layer after convolutional blocks
    x=premodel.layers[-4].output
    reshape=Reshape((input_shape[0]*input_shape[1],))(x)
    model = Model(inputs=premodel.input, outputs=reshape)

    return model

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

def cnn2(input_shape=(192,640,4)):
    '''Define CNN model'''
    model = Sequential()
    model.add(Convolution2D(288, 5, 5, padding='valid',input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(144, 3, 3, padding='valid',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(72, 3, 3, padding='valid',activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())

    return model

def cnn3(input_shape=(192,640,4)):
    '''Define CNN model'''
    model = Sequential()
    model.add(Convolution2D(filters=64, kernel_size=11, strides=(2,2), padding='valid',input_shape=input_shape,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(filters=128, kernel_size=9, strides=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(filters=256, kernel_size=5, strides=(1,1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(filters=512, kernel_size=3, strides=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())

    return model

def cnn4(input_shape=(192,640,4)):
    '''Define CNN model'''
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

def DenseBlock(input_shape = (4320,1)):
    model = Sequential()
    model.add(Dropout(0.5))
    model.add(Dense(576, input_shape = input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(288, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(144, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(72, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(72, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(36, activation='relu'))
    model.add(Dropout(0.5))
    return model

def parallel_unets_with_tf(input_shape=(192,640,3)):
    '''Define Parallel U-Nets model.'''
    #Define input size
    input_1=Input(input_shape) #Image at time=t
    input_2=Input(input_shape) #Image at time=(t-1)
                                    
    #Load unet with backbone with no imagnet weights
    unet_1 = segmentation_models.Unet('mobilenetv2', #resnet50
                                      input_shape=input_shape, 
                                      encoder_weights='imagenet',
                                      encoder_freeze=False)
    
    #Get final conv. output and skip sigmoid activation layer
    unet_1=Model(inputs=unet_1.input, outputs=unet_1.layers[-2].output)
        
    #Run input through both unets
    unet_1_out=unet_1(input_1)
    unet_2_out=unet_1(input_2)

    depth_out=Flatten(name='depth_output')(unet_1_out)
    
    #Merge unet outputs
    rbgd1=Concatenate()([input_1,unet_1_out])
    rbgd2=Concatenate()([input_2,unet_2_out])
    
    #Create transform branch for predicting rpy/xyz odom matrix/Networks for VO
    tf_cnn_t_1=cnn4()(rbgd1) #t
    tf_cnn_t_2=cnn4()(rbgd2) #t-1

    #Merge VO CNN ouputs
    merged2=Concatenate()([tf_cnn_t_1,tf_cnn_t_2])
    
    reshape=Reshape((2,tf_cnn_t_1.shape[1]))(merged2)
    lstm1=LSTM(64,return_sequences=True)(reshape) #32
    lstm2=LSTM(64,return_sequences=False)(lstm1) #32
    dense2=Dense(128, activation='relu')(lstm2)
    transform=Dense(6,activation='linear',name='vo_output')(dense2)
    
    #Define inputs and outputs    
    model = Model(inputs=[input_1,input_2], outputs=[depth_out,transform])
    
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