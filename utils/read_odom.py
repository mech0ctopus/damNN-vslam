# -*- coding: utf-8 -*-
"""
Read odom data as (12,) numpy array.
"""
import numpy as np

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
    
def read_odom(sequence_id,desired_frame): 
    '''Return odom data for sequence_id at specific frame.'''
    from math import atan2, sqrt, pi
    global odom_ids
    global frame_ids
    
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
    tx,ty,tz=current_data[3], current_data[7], current_data[11]
    
    #http://planning.cs.uiuc.edu/node103.html
    #gamma
    roll=atan2(r32,r33)
    
    #beta
    pitch=atan2(-r31,sqrt((r32**2)+(r33**2)))
    
    #alpha
    yaw=atan2(r21,r11)

    #RPY Normalization
    # roll = np.interp(roll, (-pi, pi), (0,1))
    # pitch = np.interp(pitch, (-pi, pi), (0,1))
    # yaw = np.interp(yaw, (-pi, pi), (0,1))

    #Translation Normalization
    maxes=[1827.83, 66.21801, 841.5323, 3.1415883306943355, 1.5690605800920419, 3.1415386023269973]
    mins=[-389.7065, -57.58387, 2.220446e-16, -3.141569718542289, -1.5696300642331433, -3.1414104611730806]
    
    tx = np.interp(tx, (mins[0], maxes[0]), (0,1))
    ty = np.interp(ty, (mins[1], maxes[1]), (0,1))
    tz = np.interp(tz, (mins[2], maxes[2]), (0,1))
    roll = np.interp(roll, (mins[3], maxes[3]), (0,1))
    pitch = np.interp(pitch, (mins[4], maxes[4]), (0,1))
    yaw = np.interp(yaw, (mins[5], maxes[5]), (0,1))
    
    #Set result
    current_data=np.array([roll, pitch, yaw, tx, ty, tz])
    
    return current_data

def get_max(sequence_id):
    global frame_ids
    global odom_ids
    from math import atan2, sqrt
    
    folderpath=r"G:\Documents\KITTI\raw_data_odometry\poses\\"
    odom_id=odom_ids[sequence_id]
    filepath=folderpath+odom_id+'.txt'
    
    start_frame=frame_ids[odom_id][0]
    end_frame=frame_ids[odom_id][1]
    
    frames=np.arange(start_frame,end_frame+1)
    max_x,max_y,max_z = None, None, None
    max_r,max_p,max_yaw = None, None, None
    
    #Read file
    with open(filepath,"r") as f:
        for frame in frames:
            current_data=f.readline()
            #Read first line of data and split
            current_data=current_data.strip('\n')
            current_data=current_data.split(' ')
            current_data=np.array(current_data,dtype=np.float64)
            
            current_x=current_data[3]
            current_y=current_data[7]
            current_z=current_data[11]
            
            r11=current_data[0]
            r21=current_data[4]
            r31=current_data[8]
            r32=current_data[9]
            r33=current_data[10]
            
            current_roll=atan2(r32,r33)
            current_pitch=atan2(-r31,sqrt((r32**2)+(r33**2)))
            current_yaw=atan2(r21,r11)
            
            if max_x==None or current_x>max_x:
                max_x=current_x
            if max_y==None or current_y>max_y:
                max_y=current_y
            if max_z==None or current_x>max_z:
                max_z=current_z
            if max_r==None or current_roll>max_r:
                max_r=current_roll
            if max_p==None or current_pitch>max_p:
                max_p=current_pitch
            if max_yaw==None or current_yaw>max_yaw:
                max_yaw=current_yaw
                
    return max_x,max_y,max_z,max_r,max_p,max_yaw

def get_maxes():
    global odom_ids
    idxs=list(odom_ids.keys())
    max_x,max_y,max_z,max_r,max_p,max_yaw = get_max(idxs[0])
    for idx in idxs:
        current_max_x,current_max_y,current_max_z,current_max_r,current_max_p,current_max_yaw = get_max(idx)
        if current_max_x>max_x:
            max_x=current_max_x
        if current_max_y>max_y:
            max_y=current_max_y
        if current_max_x>max_z:
            max_z=current_max_z
        if current_max_r>max_r:
            max_r=current_max_r
        if current_max_p>max_p:
            max_p=current_max_p
        if current_max_yaw>max_yaw:
            max_yaw=current_max_yaw
    return max_x,max_y,max_z,max_r,max_p,max_yaw

def get_min(sequence_id):
    global frame_ids
    global odom_ids
    from math import atan2, sqrt
    
    folderpath=r"G:\Documents\KITTI\raw_data_odometry\poses\\"
    odom_id=odom_ids[sequence_id]
    filepath=folderpath+odom_id+'.txt'
    
    start_frame=frame_ids[odom_id][0]
    end_frame=frame_ids[odom_id][1]
    
    frames=np.arange(start_frame,end_frame+1)
    min_x,min_y,min_z = None, None, None
    min_r,min_p,min_yaw = None, None, None
    
    #Read file
    with open(filepath,"r") as f:
        for frame in frames:
            current_data=f.readline()
            #Read first line of data and split
            current_data=current_data.strip('\n')
            current_data=current_data.split(' ')
            current_data=np.array(current_data,dtype=np.float64)
            
            current_x=current_data[3]
            current_y=current_data[7]
            current_z=current_data[11]
            
            r11=current_data[0]
            r21=current_data[4]
            r31=current_data[8]
            r32=current_data[9]
            r33=current_data[10]
            
            current_roll=atan2(r32,r33)
            current_pitch=atan2(-r31,sqrt((r32**2)+(r33**2)))
            current_yaw=atan2(r21,r11)
            
            if min_x==None or current_x<min_x:
                min_x=current_x
            if min_y==None or current_y<min_y:
                min_y=current_y
            if min_z==None or current_x<min_z:
                min_z=current_z
            if min_r==None or current_roll<min_r:
                min_r=current_roll
            if min_p==None or current_pitch<min_p:
                min_p=current_pitch
            if min_yaw==None or current_yaw<min_yaw:
                min_yaw=current_yaw
                
    return min_x,min_y,min_z,min_r,min_p,min_yaw

def get_mins():
    global odom_ids
    idxs=list(odom_ids.keys())
    min_x,min_y,min_z,min_r,min_p,min_yaw =  get_min(idxs[0])
    for idx in idxs:
        current_min_x,current_min_y,current_min_z,current_min_r,current_min_p,current_min_yaw = get_min(idx)
        if current_min_x<min_x:
            min_x=current_min_x
        if current_min_y<min_y:
            min_y=current_min_y
        if current_min_x<min_z:
            min_z=current_min_z
        if current_min_r<min_r:
            min_r=current_min_r
        if current_min_p<min_p:
            min_p=current_min_p
        if current_min_yaw<min_yaw:
            min_yaw=current_min_yaw
    return min_x,min_y,min_z,min_r,min_p,min_yaw

def denormalize(odom):
    #Translation Normalization
    maxes=[1827.83, 66.21801, 841.5323, 3.1415883306943355, 1.5690605800920419, 3.1415386023269973]
    mins=[-389.7065, -57.58387, 2.220446e-16, -3.141569718542289, -1.5696300642331433, -3.1414104611730806]
    tx, ty, tz, roll, pitch, yaw = odom
    tx = np.interp(tx, (0,1), (mins[0], maxes[0]))
    ty = np.interp(ty, (0,1), (mins[1], maxes[1]))
    tz = np.interp(tz, (0,1), (mins[2], maxes[2]))
    roll = np.interp(roll, (0,1), (mins[3], maxes[3]))
    pitch = np.interp(pitch, (0,1), (mins[4], maxes[4]))
    yaw = np.interp(yaw, (0,1), (mins[5], maxes[5]))
    
    denormalized_odom=list(tx, ty, tz, roll, pitch, yaw)
    
    return denormalized_odom
    
if __name__=='__main__':
    test_data=read_odom(sequence_id="2011_09_30_drive_0018",desired_frame=135)
    test_data=test_data.reshape((2,3))

    print('Maxes')
    x,y,z,r,p,yaw=get_maxes()
    print(x,y,z,r,p,yaw)
    
    print('Mins')
    x,y,z,r,p,yaw=get_mins()
    print(x,y,z,r,p,yaw)
    
    

        
