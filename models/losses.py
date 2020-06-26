# -*- coding: utf-8 -*-
"""
Loss Functions 
"""
import tensorflow.keras.backend as K

def deepvo_mse(yTrue, yPred, scale_factor=100):
    '''Define loss function per DeepVO paper.'''
    rpyTrue=yTrue[:,0:3]
    xyzTrue=yTrue[:,3:6]
    rpyPred=yPred[:,0:3]
    xyzPred=yPred[:,3:6]
    
    #scale_factor is a variable called 'K' in DeepVO paper
    xyz_loss=K.square(xyzPred - xyzTrue)
    rpy_loss=scale_factor*K.square(rpyPred - rpyTrue)
    
    return K.mean(xyz_loss+rpy_loss)

