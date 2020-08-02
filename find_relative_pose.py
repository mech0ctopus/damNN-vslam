# -*- coding: utf-8 -*-
"""
Find Relative Pose change given two RGB images.

TODO:Troubleshoot pose estimates with ground truth.

WARNING: Experimental - This code is not full tested or validated yet.

https://robotics.stackexchange.com/questions/14456/determine-the-relative-camera-pose-given-two-rgb-camera-frames-in-opencv-python
"""
import cv2
import numpy as np

def feature_detection(img_path):
    '''Detects features and builds feature descriptors.'''
    img = cv2.imread(img_path,0)
    # Initiate detector
    brisk=cv2.BRISK_create();
    # find the keypoints and descriptors
    kp, des = brisk.detectAndCompute(img,None)
    return kp, des

def get_feature_matches(kp1,des1,kp2,des2):
    '''Find descriptor matches between two images.'''
    #Try matching image descriptors
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_points = []
    good_matches=[]
    for m1, m2 in raw_matches:
        if m1.distance < 0.85 * m2.distance:
            good_points.append((m1.trainIdx, m1.queryIdx))
            good_matches.append([m1])

    if len(good_points) > 8:
        kp1 = np.float32([kp1[i].pt for (_, i) in good_points])
        kp2 = np.float32([kp2[i].pt for (i, _) in good_points])
    
    return kp1, kp2

def get_essential_matrix(kp1, kp2):
    '''Calculate essential matrix from keypoints.'''
    E, mask = cv2.findEssentialMat(kp1, kp2, focal=1.0, pp=(0., 0.), method=cv2.RANSAC, prob=0.999, threshold=3.0)
    return E

def get_relative_pose(img1_path, img2_path):
    '''Calculate relative camera pose change from two images.'''  
    #Feature Detection
    kp1, des1=feature_detection(img1_path)
    kp2, des2=feature_detection(img2_path)
    #Feature Matching
    kp1, kp2 = get_feature_matches(kp1,des1,kp2,des2)
    #Find essential Matrix (requires camera instrics). Encodes tf between images
    E=get_essential_matrix(kp1,kp2)
    #Use recoverPose and relative translation to scale
    points, R, t, mask = cv2.recoverPose(E, kp1, kp2)
    return R, t

if __name__=='__main__':
    img2_path=r"C:\Users\craig\Documents\GitHub\damNN-vslam\data\val\X\2011_09_30_drive_0018_sync_0000000134.png"
    img1_path=r"C:\Users\craig\Documents\GitHub\damNN-vslam\data\val\X\2011_09_30_drive_0018_sync_0000000135.png"

    R,t=get_relative_pose(img1_path, img2_path)
    print(R)
    print(t)
