import os

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.keras.layers import Layer
import numpy as np
import sys

def quaternion_to_euler_angle_vectorized1(w, x, y, z):
    #https://stackoverflow.com/questions/56207448/efficient-quaternions-to-euler-transformation
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = tf.math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = tf.where(t2>+1.0,+1.0,t2)
    #t2 = +1.0 if t2 > +1.0 else t2

    t2 = tf.where(t2<-1.0, -1.0, t2)
    #t2 = -1.0 if t2 < -1.0 else t2
    Y = tf.math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = tf.math.atan2(t3, t4)

    return X, Y, Z 
    
def layer_xyzq(matrix_rt, scope='pose', name='xyzq',rpy=True):
    # Rotation Matrix to quaternion + xyz
    qw = tf.sqrt(tf.reduce_sum(tf.linalg.diag_part(matrix_rt), axis=-1)) / 2.0
    qx = (matrix_rt[2, 1] - matrix_rt[1, 2]) / (4 * qw)
    qy = (matrix_rt[0, 2] - matrix_rt[2, 0]) / (4 * qw)
    qz = (matrix_rt[1, 0] - matrix_rt[0, 1]) / (4 * qw)

    x = matrix_rt[0, 3]
    y = matrix_rt[1, 3]
    z = matrix_rt[2, 3]
    if rpy:
        roll, pitch, yaw = quaternion_to_euler_angle_vectorized1(qw, qx, qy, qz)
        xyzq = tf.stack([roll, pitch, yaw, x, y, z], axis=-1, name=name)
        return xyzq
    else:
        xyzq = tf.stack([qw, qx, qy, qz, x, y, z], axis=-1, name=name)
        return xyzq

class SE3CompositeLayer(Layer):
    def __init__(self, **kwargs):
        super(Layer, self).__init__(**kwargs)
        self._state_size = tf.keras.Input((None, 4, 4)) # Accumulated SE3 Matrix
        self._output_size = tf.keras.Input((None, 6))   # xyz + q

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, input_shape):
        print(type(input_shape), input_shape)
        self.built = True
        
    def __call__(self, inputs, scope=None, *args, **kwargs): #state, 
        x_trans, x_rot = tf.split(inputs, 2, 1)
        x_trans=tf.transpose(x_trans)

        col_1 = tf.linalg.cross(x_rot, tf.constant([1, 0, 0], shape=(1, 3), dtype=tf.float32))
        col_2 = tf.linalg.cross(x_rot, tf.constant([0, 1, 0], shape=(1, 3), dtype=tf.float32))
        col_3 = tf.linalg.cross(x_rot, tf.constant([0, 0, 1], shape=(1, 3), dtype=tf.float32))

        omega = tf.concat([col_1, col_2, col_3],axis=0)
        omega_t = tf.transpose(omega)

        theta = tf.math.sqrt(tf.reduce_sum(tf.square(x_rot)))
        sin_theta = tf.math.sin(theta)

        sin_t_by_t = tf.math.truediv(sin_theta, theta)
        one_min_cos_t_by_t = tf.math.truediv(tf.math.add(1.0, -tf.math.cos(theta)), tf.math.square(theta))

        #Rotation matrix
        R = tf.scalar_mul(sin_t_by_t,omega_t) + \
          tf.scalar_mul(one_min_cos_t_by_t, tf.matmul(omega_t, omega_t)) + \
          tf.eye(3,dtype=tf.float32)
        print('R: ',R)
        
        theta_minus_sin_theta_div_theta3 = tf.math.truediv(tf.math.add(theta, -sin_theta), tf.math.pow(theta,3))

        V = tf.scalar_mul(one_min_cos_t_by_t, omega_t) +\
          tf.scalar_mul(theta_minus_sin_theta_div_theta3, tf.matmul(omega_t, omega_t)) \
          + tf.eye(3,dtype=tf.float32)
          
        t=tf.reshape(x_trans,(3,1))
        print('x_trans: ',x_trans)
        print('t: ',t)
        print('V: ',V)
        Vu = tf.matmul(V,x_trans) #V * x_trans
        #-----------------
        #xyzq = layer_xyzq(r_matrix)
        print('Vu: ',Vu)
        #Vu = tf.map_fn(lambda x: tf.concat((x, [[0, 0, 0, 1]]), axis=0), Vu)
        Rt=tf.concat((R,t), axis=1)
        print('Rt: ',Rt)
        H=tf.concat((Rt,[[0.0,0.0,0.0,1.0]]),axis=0)
        print('H: ',H)
        xyzq = layer_xyzq(H,rpy=True)
        xyzq=tf.reshape(xyzq,shape=(1,6))
        print(xyzq)

        return xyzq