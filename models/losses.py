# -*- coding: utf-8 -*-
"""
Loss Functions 
"""
import tensorflow.keras.backend as K
import tensorflow as tf

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

def undeepvo_rpy_mse(yTrue, yPred, scale_factor=100):
    '''Define loss function per DeepVO paper.'''   
    #scale_factor is a variable called 'K' in DeepVO paper
    rpy_loss=scale_factor*K.square(yPred - yTrue)
    return K.mean(rpy_loss)

def undeepvo_xyz_mse(yTrue, yPred):
    '''Define loss function per DeepVO paper.'''   
    #scale_factor is a variable called 'K' in DeepVO paper
    xyz_loss=K.square(yPred - yTrue)
    return K.mean(xyz_loss)

if __name__=='__main__':
    #Test out custom loss function
    a = tf.constant([[1,2,3,4,5,6],[1,2,3,4,5,6]])
    b = tf.constant([[2,2,3,4,5,6],[1,2,3,4,5,6]])
    loss = deepvo_mse(a, b)
    loss = tf.print(loss, [loss], "loss")