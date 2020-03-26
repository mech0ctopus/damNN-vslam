# -*- coding: utf-8 -*-
"""
Final Models.
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Reshape, LSTM
import segmentation_models

segmentation_models.set_framework('tf.keras')

def unet():
    '''Define U-Net model.'''
    #Load unet with resnet34 backbone with no weights
    premodel = segmentation_models.Unet('resnet34', 
                                        input_shape=(480, 640, 3), 
                                        encoder_weights=None,
                                        encoder_freeze=False)
    
    #Get final conv. output and skip sigmoid activation layer
    x=premodel.layers[-2].output 
    x=Reshape((480,640))(x)
    x=LSTM(16)(x)
    reshape=Reshape((307200,))(x)
    model = Model(inputs=premodel.input, outputs=reshape)

    return model