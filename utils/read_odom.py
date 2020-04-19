# -*- coding: utf-8 -*-
"""
Read odom data as (12,) numpy array.
"""
import numpy as np

def read_odom(sequence_id,desired_frame): 
    '''Return odom data for sequence_id at specific frame.'''
    from math import atan2, sqrt
    odom_ids={"2011_10_03_drive_0027":"00",
              "2011_10_03_drive_0042":"01",
              "2011_10_03_drive_0034":"02",
              "2011_09_26_drive_0067":"03",
              "2011_09_30_drive_0016":"04",
              "2011_09_30_drive_0018":"05",
              "2011_09_30_drive_0020":"06",
              "2011_09_30_drive_0027":"07",
              "2011_09_30_drive_0028":"08",
              "2011_09_30_drive_0033":"09",
              "2011_09_30_drive_0034":"10"}
    
    frame_ids={"00": (0,4540),
                "01": (0,1100),
                "02": (0,4660),
                "03": (0,800),
                "04": (0,270),
                "05": (0,2760),
                "06": (0,1100),
                "07": (0,1100),
                "08": (1100,5170),
                "09": (0,1590),
                "10": (0,1200)}
    
    folderpath=r"G:\Documents\KITTI\raw_data_odometry\poses\\"
    odom_id=odom_ids[sequence_id]
    filepath=folderpath+odom_id+'.txt'
    
    start_frame=frame_ids[odom_id][0]
    end_frame=frame_ids[odom_id][1]
    
    frames=np.arange(start_frame,end_frame+1)

    #Read file
    with open(filepath,"r") as f:
        for frame in frames:
            current_data=f.readline()
            if frame==desired_frame:
                #Read first line of data and split
                current_data=current_data.strip('\n')
                current_data=current_data.split(' ')
                current_data=np.array(current_data,dtype=np.float64)
                #data[frame]=np.array(current_data,dtype=np.float16)
                break
            
    #Get roll, pitch, yaw from data
    r11=current_data[0]
    #r12=current_data[1]
    #r13=current_data[2]
    
    r21=current_data[4]
    #r22=current_data[5]
    #r23=current_data[6]
    
    r31=current_data[8]
    r32=current_data[9]
    r33=current_data[10]
    
    #Get translations from data
    dx,dy,dz=current_data[3], current_data[7], current_data[11]
    
    #http://planning.cs.uiuc.edu/node103.html
    #gamma
    roll=atan2(r32,r33)
    
    #beta
    pitch=atan2(-r31,sqrt((r32**2)+(r33**2)))
    
    #alpha
    yaw=atan2(r21,r11)

    #Set result
    current_data=np.array([roll, pitch, yaw, dx, dy, dz])
    
    return current_data

if __name__=='__main__':
    test_data=read_odom(sequence_id="2011_09_30_drive_0018",desired_frame=135)
    test_data=test_data.reshape((2,3))
    print(test_data)
    print(test_data.shape)

        
