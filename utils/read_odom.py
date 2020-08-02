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
    
    folderpath=r"C:\Users\craig\Documents\GitHub\damNN-vslam\raw_data_odometry\poses\\"
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
            
    #Create Rotation Matrix
    R = [current_data[0], current_data[1], current_data[2], 
         current_data[4], current_data[5], current_data[6], 
         current_data[8], current_data[9], current_data[10]]
    R = np.array(R).reshape(3,3)
    
    #Get translations from data
    tx,ty,tz=current_data[3], current_data[7], current_data[11]
    
    #Get roll, pitch yaw from rotation matrix
    #https://www.learnopencv.com/rotation-matrix-to-euler-angles/
    roll, pitch, yaw = rotationMatrixToEulerAngles(R)

    #Set result
    current_data=np.array([roll, pitch, yaw, tx, ty, tz])
    
    return current_data

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    from math import atan2, sqrt
    #assert(isRotationMatrix(R))
    
    sy = sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if  not singular :
        x = atan2(R[2,1] , R[2,2])
        y = atan2(-R[2,0], sy)
        z = atan2(R[1,0], R[0,0])
    else :
        print("SINGULAR SINGULAR SINGULAR")
        input("SINGULAR Press Enter to Continue...")
        x = atan2(-R[1,2], R[1,1])
        y = atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

def get_max_diff(sequence_id):
    global frame_ids
    global odom_ids
    from math import atan2, sqrt
    
    folderpath=r"G:\Documents\KITTI\raw_data_odometry\poses\\"
    odom_id=odom_ids[sequence_id]
    filepath=folderpath+odom_id+'.txt'
    
    start_frame=frame_ids[odom_id][0]
    end_frame=frame_ids[odom_id][1]
    
    frames=np.arange(start_frame,end_frame+1)
    max_x_diff,max_y_diff,max_z_diff = None, None, None
    max_r_diff,max_p_diff,max_yaw_diff = None, None, None
    
    #Read file
    with open(filepath,"r") as f:
        for idx,frame in enumerate(frames):
            current_data=f.readline()
            #Read first line of data and split
            current_data=current_data.strip('\n')
            current_data=current_data.split(' ')
            current_data=np.array(current_data,dtype=np.float64)
            
            current_x=current_data[3]
            current_y=current_data[7]
            current_z=current_data[11]
            
            #Create Rotation Matrix
            current_R = [current_data[0], current_data[1], current_data[2], 
                         current_data[4], current_data[5], current_data[6], 
                         current_data[8], current_data[9], current_data[10]]
            current_R = np.array(current_R).reshape(3,3)
    
            #Get roll, pitch yaw from rotation matrix
            #https://www.learnopencv.com/rotation-matrix-to-euler-angles/
            current_roll, current_pitch, current_yaw = rotationMatrixToEulerAngles(current_R)
            if idx>0:
                current_x_diff=current_x-prev_x
                current_y_diff=current_y-prev_y
                current_z_diff=current_z-prev_z
                current_r_diff=current_roll-prev_roll
                current_p_diff=current_pitch-prev_pitch
                current_yaw_diff=current_yaw-prev_yaw
                
                #Adjust angular displacement to deal with swinging through +/- pi
                # if np.abs(current_r_diff) > np.pi:
                #     if current_r_diff < 0:
                #         current_r_diff = -1 * (2*np.pi + current_r_diff)
                #     else :
                #         current_r_diff = (2*np.pi - current_r_diff)
                        
                # if np.abs(current_p_diff) > np.pi:
                #     if current_p_diff < 0:
                #         current_p_diff = -1 * (2*np.pi + current_p_diff)
                #     else :
                #         current_p_diff = (2*np.pi - current_p_diff)
                        
                # if np.abs(current_yaw_diff) > np.pi:
                #     if current_yaw_diff < 0:
                #         current_yaw_diff = -1 * (2*np.pi + current_yaw_diff)
                #     else :
                #         current_yaw_diff = (2*np.pi - current_yaw_diff)
                        
                if max_x_diff==None or current_x_diff>max_x_diff:
                    max_x_diff=current_x_diff
                if max_y_diff==None or current_y_diff>max_y_diff:
                    max_y_diff=current_y_diff
                if max_z_diff==None or current_x_diff>max_z_diff:
                    max_z_diff=current_z_diff
                if max_r_diff==None or current_r_diff>max_r_diff:
                    max_r_diff=current_r_diff
                if max_p_diff==None or current_p_diff>max_p_diff:
                    max_p_diff=current_p_diff
                if max_yaw_diff==None or current_yaw_diff>max_yaw_diff:
                    max_yaw_diff=current_yaw_diff
                 
            prev_roll, prev_pitch, prev_yaw = current_roll, current_pitch, current_yaw
            prev_x, prev_y, prev_z=current_x, current_y, current_z
            
    return max_x_diff,max_y_diff,max_z_diff,max_r_diff,max_p_diff,max_yaw_diff

def get_max_diffs():
    global odom_ids
    idxs=list(odom_ids.keys())
    max_x_diff,max_y_diff,max_z_diff,max_r_diff,max_p_diff,max_yaw_diff = get_max_diff(idxs[0])
    for idx in idxs:
        current_x_diff,current_y_diff,current_z_diff,current_r_diff,current_p_diff,current_yaw_diff = get_max_diff(idx)
        if current_x_diff>max_x_diff:
            max_x_diff=current_x_diff
        if current_y_diff>max_y_diff:
            max_y_diff=current_y_diff
        if current_x_diff>max_z_diff:
            max_z_diff=current_z_diff
        if current_r_diff>max_r_diff:
            max_r_diff=current_r_diff
        if current_p_diff>max_p_diff:
            max_p_diff=current_p_diff
        if current_yaw_diff>max_yaw_diff:
            max_yaw_diff=current_yaw_diff
    return max_r_diff,max_p_diff,max_yaw_diff,max_x_diff,max_y_diff,max_z_diff

def get_min_diff(sequence_id):
    global frame_ids
    global odom_ids
    from math import atan2, sqrt
    
    folderpath=r"G:\Documents\KITTI\raw_data_odometry\poses\\"
    odom_id=odom_ids[sequence_id]
    filepath=folderpath+odom_id+'.txt'
    
    start_frame=frame_ids[odom_id][0]
    end_frame=frame_ids[odom_id][1]
    
    frames=np.arange(start_frame,end_frame+1)
    min_x_diff,min_y_diff,min_z_diff = None, None, None
    min_r_diff,min_p_diff,min_yaw_diff = None, None, None
    
    #Read file
    with open(filepath,"r") as f:
        for idx,frame in enumerate(frames):
            current_data=f.readline()
            #Read first line of data and split
            current_data=current_data.strip('\n')
            current_data=current_data.split(' ')
            current_data=np.array(current_data,dtype=np.float64)
            
            current_x=current_data[3]
            current_y=current_data[7]
            current_z=current_data[11]

            #Create Rotation Matrix
            current_R = [current_data[0], current_data[1], current_data[2], 
                         current_data[4], current_data[5], current_data[6], 
                         current_data[8], current_data[9], current_data[10]]
            current_R = np.array(current_R).reshape(3,3)
            
            #Get roll, pitch yaw from rotation matrix
            #https://www.learnopencv.com/rotation-matrix-to-euler-angles/
            current_roll, current_pitch, current_yaw = rotationMatrixToEulerAngles(current_R)
            if idx>0:
                current_x_diff=current_x-prev_x
                current_y_diff=current_y-prev_y
                current_z_diff=current_z-prev_z
                current_r_diff=current_roll-prev_roll
                current_p_diff=current_pitch-prev_pitch
                current_yaw_diff=current_yaw-prev_yaw

                #Adjust angular displacement to deal with swinging through +/- pi
                # if np.abs(current_r_diff) > np.pi:
                #     if current_r_diff < 0:
                #         current_r_diff = -1 * (2*np.pi + current_r_diff)
                #     else :
                #         current_r_diff = (2*np.pi - current_r_diff)
                        
                # if np.abs(current_p_diff) > np.pi:
                #     if current_p_diff < 0:
                #         current_p_diff = -1 * (2*np.pi + current_p_diff)
                #     else :
                #         current_p_diff = (2*np.pi - current_p_diff)
                        
                # if np.abs(current_yaw_diff) > np.pi:
                #     if current_yaw_diff < 0:
                #         current_yaw_diff = -1 * (2*np.pi + current_yaw_diff)
                #     else :
                #         current_yaw_diff = (2*np.pi - current_yaw_diff)
                        
                if min_x_diff==None or current_x_diff<min_x_diff:
                    min_x_diff=current_x_diff
                if min_y_diff==None or current_y_diff<min_y_diff:
                    min_y_diff=current_y_diff
                if min_z_diff==None or current_x_diff>min_z_diff:
                    min_z_diff=current_z_diff
                if min_r_diff==None or current_r_diff<min_r_diff:
                    min_r_diff=current_r_diff
                if min_p_diff==None or current_p_diff<min_p_diff:
                    min_p_diff=current_p_diff
                if min_yaw_diff==None or current_yaw_diff<min_yaw_diff:
                    min_yaw_diff=current_yaw_diff
              
            prev_roll, prev_pitch, prev_yaw = current_roll, current_pitch, current_yaw
            prev_x, prev_y, prev_z=current_x, current_y, current_z
            
    return min_x_diff,min_y_diff,min_z_diff,min_r_diff,min_p_diff,min_yaw_diff

def get_min_diffs():
    global odom_ids
    idxs=list(odom_ids.keys())
    min_x_diff,min_y_diff,min_z_diff,min_r_diff,min_p_diff,min_yaw_diff = get_min_diff(idxs[0])
    for idx in idxs:
        current_x_diff,current_y_diff,current_z_diff,current_r_diff,current_p_diff,current_yaw_diff = get_min_diff(idx)
        if current_x_diff<min_x_diff:
            min_x_diff=current_x_diff
        if current_y_diff<min_y_diff:
            min_y_diff=current_y_diff
        if current_x_diff<min_z_diff:
            min_z_diff=current_z_diff
        if current_r_diff<min_r_diff:
            min_r_diff=current_r_diff
        if current_p_diff<min_p_diff:
            min_p_diff=current_p_diff
        if current_yaw_diff<min_yaw_diff:
            min_yaw_diff=current_yaw_diff
    return min_r_diff,min_p_diff,min_yaw_diff,min_x_diff,min_y_diff,min_z_diff

def normalize(odom):
    '''Normalize RPYXYZ'''
    max_diffs=[6.282581652723159, 0.07379722218974893, 6.282196556258313, 2.425999999999931, 0.2665500000000005, 1.5289750000000002]
    min_diffs=[-6.2823649683145195, -0.08280581919847907, -6.282372246787359, -1.70900000000006, -0.19907300000000028, -0.010032000000000707]
    roll, pitch, yaw, tx, ty, tz = odom[0],odom[1],odom[2],odom[3],odom[4],odom[5]
    roll = np.interp(roll, (min_diffs[0], max_diffs[0]), (0,1))
    pitch = np.interp(pitch, (min_diffs[1], max_diffs[1]), (0,1))
    yaw = np.interp(yaw, (min_diffs[2], max_diffs[2]), (0,1))
    
    tx = np.interp(tx, (min_diffs[3], max_diffs[3]), (0,1))
    ty = np.interp(ty, (min_diffs[4], max_diffs[4]), (0,1))
    tz = np.interp(tz, (min_diffs[5], max_diffs[5]), (0,1))
    
    normalized_odom=np.array([roll, pitch, yaw, tx, ty, tz])
    
    return normalized_odom

def denormalize(odom):
    '''Denormalize RPYXYZ'''
    max_diffs=[6.282581652723159, 0.07379722218974893, 6.282196556258313, 2.425999999999931, 0.2665500000000005, 1.5289750000000002]
    min_diffs=[-6.2823649683145195, -0.08280581919847907, -6.282372246787359, -1.70900000000006, -0.19907300000000028, -0.010032000000000707]
    roll, pitch, yaw, tx, ty, tz = odom[0],odom[1],odom[2],odom[3],odom[4],odom[5]

    roll = np.interp(roll, (0,1), (min_diffs[0], max_diffs[0]))
    pitch = np.interp(pitch, (0,1), (min_diffs[1], max_diffs[1]))
    yaw = np.interp(yaw, (0,1), (min_diffs[2], max_diffs[2]))
    
    tx = np.interp(tx, (0,1), (min_diffs[3], max_diffs[3]))
    ty = np.interp(ty, (0,1), (min_diffs[4], max_diffs[4]))
    tz = np.interp(tz, (0,1), (min_diffs[5], max_diffs[5]))
    
    denormalized_odom=np.array([roll, pitch, yaw, tx, ty, tz])
    
    return denormalized_odom
    
if __name__=='__main__':
    test_data=read_odom(sequence_id="2011_09_30_drive_0018",desired_frame=135)
    test2_data=read_odom(sequence_id="2011_09_30_drive_0018",desired_frame=134)
    # test_data=test_data.reshape((2,3))
    print('Actual RPYXYZ:')
    print(test_data-test2_data)

    # print('Max Diffs')
    # r,p,yaw,x,y,z=get_max_diffs()
    # print(r,p,yaw,x,y,z)
    
    # print('Min Diffs')
    # r,p,yaw,x,y,z=get_min_diffs()
    # print(r,p,yaw,x,y,z)