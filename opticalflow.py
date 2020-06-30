# -*- coding: utf-8 -*-
"""
https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
"""

import numpy as np
import cv2 as cv

def get_optical_flow(frame1,frame2):
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    
    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY) 
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    return bgr

if __name__ == '__main__':
    frame1 = cv.imread(r"C:\Users\craig\Documents\GitHub\damNN-vslam\data\val\X\2011_09_30_drive_0018_sync_0000000135.png")
    frame2 = cv.imread(r"C:\Users\craig\Documents\GitHub\damNN-vslam\data\val\X\2011_09_30_drive_0018_sync_0000000136.png")
    test=get_optical_flow(frame1, frame2)