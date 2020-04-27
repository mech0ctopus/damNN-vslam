# -*- coding: utf-8 -*-
"""
Tool for testing depth estimation models on live video stream.
"""
import cv2
import numpy as np
import deep_utils
import image_utils
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import model_from_json
from json_config import json_config
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge

class DepthPub:
    def __init__(self):
        #Load weights (h5)
        self.weights={#'unet':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Weights & Models\20Epochs_NoAugment_LowerLR\20200213-065621\U-Net_weights_best.hdf5",
                #'wnet':r"G:\WPI\Courses\2019\Deep Learning for Advanced Robot Perception, RBE595\Project\VEHITS\Weights & Models\20Epochs_NoAugment_LowerLR\20200213-082137\W-Net_weights_best.hdf5",
                'wnet_c':"W-Net_Connected_weights_best_KITTI_35Epochs.hdf5"
                 }

        print 'Loading model'
        self.model_name='wnet_c'
        self.model=model_from_json(json_config)

        print 'Loading Weights'
        self.model.load_weights(self.weights['wnet_c'])

        print 'Compiling model'
        self.model.compile(loss='mean_squared_error',optimizer=Adam(),metrics=['mse']) 

        self.bridge=CvBridge()
        self.pub = rospy.Publisher('depth_image', Image, queue_size=10)
        self.rgb_pub = rospy.Publisher('rgb_image', Image, queue_size=10)
        self.rgb_pub_cal = rospy.Publisher('rgb_pub_cal', CameraInfo, queue_size=10)
        self.rgb_sub=rospy.Subscriber('/raspicam_node/image_rgb',Image,self.callback)

    def callback(self,raw_rgb_img):
        '''Runs depth estimation on live webcam video stream'''
        #Resize image
        rgb_img=self.bridge.imgmsg_to_cv2(raw_rgb_img,"bgr8") #544 High x 1024 wide
        # print type(rgb_img) #np.ndarray
        # print rgb_img.shape #544x1024x3
        img=rgb_img[288:480,:,:]
        rgb_img=img

        img=img.reshape(1,192,640,3)
        img=np.divide(img,255).astype(np.float16)
        #Predict depth
        y_est=self.model.predict(img)
        y_est=y_est.reshape((192,640))

        #Thresholding
        for ii,row in enumerate(y_est):
            for jj,value in enumerate(row):
                if (y_est[ii][jj])>0.02: #max=0.115, min=0.0009
                    y_est[ii][jj]=0

        y_est=y_est.reshape((192,640,1))

        #Define ROS messages
        h=Header()
        h.stamp=raw_rgb_img.header.stamp
        #h.stamp=rospy.Time.now()
        h.frame_id='camera_link'
        #Define depth image message
        depth_img=self.bridge.cv2_to_imgmsg((y_est*255).astype(np.float32),"32FC1")
        depth_img.header=h

        #Define rgb message
        rgb_img=self.bridge.cv2_to_imgmsg(rgb_img,"bgr8")
        rgb_img.header=h

        #Define camera calibration info. message
        cal_msg=CameraInfo()
        cal_msg.header=h
        cal_msg.distortion_model="plumb_bob"
        cal_msg.height=192
        cal_msg.width=640

        #FROM FIELDSAFE MULTISENSE CAMERA
        #Distortion coefficients
        cal_msg.D=[0.0030753163155168295, 0.002497022273018956, 0.0003005412872880697, 0.001575434347614646, -0.003454494522884488, 0.0, 0.0, 0.0]
        #Intrinsic Camera Matrix
        cal_msg.K=[ 555.9204711914062, 0.0, 498.1905517578125, 0.0, 556.6275634765625, 252.35089111328125, 0.0, 0.0, 1.0]
        cal_msg.R=[0.9999634027481079, -0.000500216381624341, 0.00853759702295065, 0.0005011018947698176, 0.9999998807907104, -0.00010158627264900133, -0.00853754486888647, 0.00010586075950413942, 0.9999635219573975]
        #Projection Matrix
        cal_msg.P=[580.6427001953125, 0.0, 512.0, 0.0, 0.0, 580.6427001953125, 254.5, 0.0, 0.0, 0.0, 1.0, 0.0]


        #Publish messages
        self.pub.publish(depth_img)
        self.rgb_pub.publish(rgb_img)
        self.rgb_pub_cal.publish(cal_msg)

def main():
    dp=DepthPub()
    rospy.init_node('depth_publisher', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting Down')
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()