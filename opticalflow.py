# -*- coding: utf-8 -*-
"""
https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
"""

import numpy as np
import cv2 as cv
from os.path import basename
from glob import glob

def get_optical_flow(img_path1,img_path2,vis=False, save=False):
    frame1 = cv.imread(img_path1) #Time=t-1
    frame2 = cv.imread(img_path2) #Time=t
    prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)

    next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY) 
    flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    if vis:
        mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,1] = 255
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
        bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
    
        if save:
            frame2_name=basename(img_path2).split('.')[0] #Time=t
            cv.imwrite(rf'data/train/flow/{frame2_name}_flow.png',bgr)
        return bgr
    else:
        hsv[...,0]=cv.normalize(flow[:,:,0],None,0,255,cv.NORM_MINMAX)
        hsv[...,1]=cv.normalize(flow[:,:,1],None,0,255,cv.NORM_MINMAX)
        
        if save:
            frame2_name=basename(img_path2).split('.')[0] #Time=t
            cv.imwrite(rf'data/train/flow/{frame2_name}_flow.png',hsv)
        return hsv
        
def get_optical_flows(folderpath, vis, save=False):
    img_paths=glob(folderpath+'*.PNG')
    num_paths=len(img_paths)
    for i in range(1,len(img_paths)):
        get_optical_flow(img_paths[i-1], img_paths[i], vis=vis, save=save)
        print(str(i)+'/'+str(num_paths))

if __name__ == '__main__':   
    test_folder=r"C:\Users\craig\Documents\GitHub\damNN-vslam\data\train\X\\"
    get_optical_flows(test_folder,vis=False,save=True)