# -*- coding: utf-8 -*-
"""
SE3 Layer

Functions from/based off of ROS' TF package.
"""
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.framework import tensor_shape
import numpy as np
import math

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())

def translation_from_matrix(matrix):
    """Return translation vector from 3x4 transformation matrix."""
    return np.array(matrix)[0:3,3]

def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    From ROS' TF transformations.py
    
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def rpyxyz_from_matrix(matrix):
    '''Returns array of roll, pitch, yaw, x, y, z from 3x4 transform matrix.
    Angles in radians.'''
    # if len(matrix)==12:
    #     matrix=matrix.reshape((3,4))
    #matrix=np.array(matrix)
    #print(matrix)
    roll, pitch, yaw=euler_from_matrix(matrix, axes='sxyz')
    x, y, z = translation_from_matrix(matrix)
    return tf.constant([roll,pitch,yaw,x,y,z],dtype='float32')

def rpyxyz_from_matrices(matrices,N=32):
    '''Returns array of roll, pitch, yaw, x, y, z from Nx3x4 transform matrix.
    Angles in radians.'''
    output=np.zeros((N,6))
    for idx,matrix in enumerate(matrices):
        roll, pitch, yaw=euler_from_matrix(matrix, axes='sxyz')
        x, y, z = translation_from_matrix(matrix)
        output[idx,:]=np.array([roll,pitch,yaw,x,y,z],dtype='float32')
    output=tf.constant(output,dtype='float32')
    #output=tf.convert_to_tensor(output,dtype='float32')
    print(output)
    return output

class SE3Layer(Layer):
    """ Extends keras Layer to support SE3 rpyxyz from transform matrix
    Args:
        Layer (keras.Layer): Generic Keras Layer
    """
    def __init__(self):
        super(SE3Layer, self).__init__()
    
    def call(self, xin):
        """Tensorflow hook 
        Args:
            xin (tensor): 12-element tensor
        Returns:
            RPYXYZ_vector : a 6-element tensor containing the results of SE3
        """
        xout = tf.py_function(rpyxyz_from_matrices, 
                              [xin],
                              'float32',
                              name='SE3')
        #xout = K.stop_gradient(xout) # explicitly set no grad
        xout.set_shape([None,6]) # explicitly set output shape
        
        #return tf.reshape(xout, (-1,6))
        return xout