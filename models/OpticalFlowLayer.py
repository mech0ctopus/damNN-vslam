import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, Layer
from keras.models import Model
from keras import backend as K
import cv2

class OpticalFlowLayer( Layer ) :
    """ Extends keras Layer to support opencv Farneback Optical Flow Calculation

    Args:
        Layer (keras.Layer): Generic Keras Layer
    """
    def call( self, xin )  :
        """Tensorflow hook 

        Args:
            xin (tensor): 2 image tensor

        Returns:
            optical_flow_vector : a tensor containing the results of the optical flow computation
        """
        xout = tf.py_function( compute_optical_flow, 
                           [xin],
                           'float32',
                           name='OpticalFlow')
        xout = K.stop_gradient( xout ) # explicitly set no grad
        xout.set_shape( [xin.shape[0], 66, 200, xin.shape[-1]] ) # explicitly set output shape
        return xout
    
    def optical_flow(prevs,next):
        """Computes Farneback vector between two images

        Args:
            prevs (image tensor): 
            next (image tensor): 

        Returns:
            tensor containing optical flow calculation
        """
        img=cv2.cvtColor(img,cv2.COLOR_BGR2YUV) 
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        return rgb.astype('float32')

    def compute_optical_flow(img4d) :
        """Parses images from tensor

        Args:
            img4d (two-image tensor): 

        Returns:
            tensor: contains optical flow results
        """
        prev = img4d[:,:,:,0]
        next = img4d[:,:,:,1]
        return optical_flow(prevs,next)
