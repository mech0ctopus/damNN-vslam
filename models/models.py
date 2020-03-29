# -*- coding: utf-8 -*-
"""
Final Models.
"""
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate, Dense
from tensorflow.keras.layers import Flatten, Input, Reshape
import segmentation_models

segmentation_models.set_framework('tf.keras')

def unet(input_shape=(375,1242,3)):
    '''Define U-Net model.'''
    #Load unet with resnet34 backbone with no weights
    premodel = segmentation_models.Unet('resnet34', 
                                        input_shape=input_shape, 
                                        encoder_weights=None,
                                        encoder_freeze=False)
    
    #Get final conv. output and skip sigmoid activation layer
    x=premodel.layers[-2].output 
    reshape=Reshape((input_shape[0]*input_shape[1],))(x)
    model = Model(inputs=premodel.input, outputs=reshape)

    return model

def parallel_unets(input_shape=(375,1242,3)):
    '''Define Parallel U-Nets model.'''
    #Define input size
    input_1=Input(input_shape) #Image at time=t
    input_2=Input(input_shape) #Image at time=(t-1)
                                        
    #Load unet with resnet34 backbone with no weights
    unet_1 = segmentation_models.Unet('resnet34', 
                                      input_shape=input_shape, 
                                      encoder_weights=None,
                                      encoder_freeze=False)
    unet_2 = segmentation_models.Unet('resnet34', 
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

if __name__=='__main__':
    model=parallel_unets()
    model.summary()
    plot_model(model, to_file='model.png', 
               show_shapes=True, 
               show_layer_names=True, 
               rankdir='LR',  #LR or TB for vertical or horizontal
               expand_nested=False, 
               dpi=96)