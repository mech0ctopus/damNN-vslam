#!/usr/bin/env python
from __future__ import print_function

import roslib
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("/damnn/image/topic",Image,queue_size=1)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/Multisense/depth",Image,self.callback)

  def callback(self,msg_depth):
    try:
        # The depth image is a single-channel float32 image
        # the values is the distance in mm in z axis
        cv_image = self.bridge.imgmsg_to_cv2(msg_depth, "32FC1")
        # Convert the depth image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
        # Normalize the depth image to fall between 0 (black) and 1 (white)
        # http://docs.ros.org/electric/api/rosbag_video/html/bag__to__video_8cpp_source.html lines 95-125
        cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
        # Resize to the desired size
        #cv_image_resized = cv2.resize(cv_image_norm, self.desired_shape, interpolation = cv2.INTER_CUBIC)
        self.depthimg = cv_image_norm
        cv2.imshow("Image from my node", self.depthimg)
        cv2.waitKey(1)
    except CvBridgeError as e:
        print(e)


    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(self.depthimg, "32FC1"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)