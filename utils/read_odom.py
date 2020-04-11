# -*- coding: utf-8 -*-
"""
Read odom data as (12,) numpy array.
"""
import numpy as np

def read_odom(sequence_id,desired_frame): 
    '''Return odom data for sequence_id at specific frame.'''
    odom_ids={"2011_10_03_drive_0027":"00",
              "2011_10_03_drive_0042":"01",
              "2011_10_03_drive_0034":"02",
              "2011_09_26_drive_0067":"03",
              "2011_09_30_drive_0016":"04",
              "2011_09_30_drive_0018c":"05",
              "2011_09_30_drive_0020":"06",
              "2011_09_30_drive_0027":"07",
              "2011_09_30_drive_0028":"08",
              "2011_09_30_drive_0033":"09",
              "2011_09_30_drive_0034":"10"}
    
    frames={"00": (0,4540),
            "01": (0,110),
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
    
    start_frame=frames[odom_id][0]
    end_frame=frames[odom_id][1]
    frames=np.arange(start_frame,end_frame+1)
    data={}
    #Read file
    with open(filepath,"r") as f:
        for frame in frames:
            #Read first line of data and split
            current_data=f.readline().strip('\n')
            current_data=current_data.split(' ')
            data[frame]=np.array(current_data,dtype=np.float64)
            
    return data[desired_frame]

if __name__=='__main__':
    test_data=read_odom(sequence_id="2011_09_30_drive_0016_sync",desired_frame=35)
    print(test_data)

        
