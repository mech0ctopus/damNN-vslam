function [this_pose, last_pose] = get_pose(odom_data,offset_idx)
    if offset_idx == 1 
        this_pose.r11 = odom_data.r11(1);
        this_pose.r12 = odom_data.r12(1);
        this_pose.r13 = odom_data.r13(1);
        this_pose.tx = odom_data.tx(1);
        
        this_pose.r21 = odom_data.r21(1);
        this_pose.r22 = odom_data.r22(1);
        this_pose.r23 = odom_data.r23(1);
        this_pose.ty = odom_data.ty(1);
        
        this_pose.r31 = odom_data.r31(1);
        this_pose.r32 = odom_data.r32(1);
        this_pose.r33 = odom_data.r33(1);
        this_pose.tz = odom_data.tz(1);
        
        last_pose = this_pose;
    else
        this_pose.r11 = odom_data.r11(offset_idx);
        this_pose.r12 = odom_data.r12(offset_idx);
        this_pose.r13 = odom_data.r13(offset_idx);
        this_pose.tx = odom_data.tx(offset_idx);
        
        this_pose.r21 = odom_data.r21(offset_idx);
        this_pose.r22 = odom_data.r22(offset_idx);
        this_pose.r23 = odom_data.r23(offset_idx);
        this_pose.ty = odom_data.ty(offset_idx);
        
        this_pose.r31 = odom_data.r31(offset_idx);
        this_pose.r32 = odom_data.r32(offset_idx);
        this_pose.r33 = odom_data.r33(offset_idx);
        this_pose.tz = odom_data.tz(offset_idx);
    
        last_pose.r11 = odom_data.r11(offset_idx-1);
        last_pose.r12 = odom_data.r12(offset_idx-1);
        last_pose.r13 = odom_data.r13(offset_idx-1);
        last_pose.tx = odom_data.tx(offset_idx-1);
        
        last_pose.r21 = odom_data.r21(offset_idx-1);
        last_pose.r22 = odom_data.r22(offset_idx-1);
        last_pose.r23 = odom_data.r23(offset_idx-1);
        last_pose.ty = odom_data.ty(offset_idx-1);
        
        last_pose.r31 = odom_data.r31(offset_idx-1);
        last_pose.r32 = odom_data.r32(offset_idx-1);
        last_pose.r33 = odom_data.r33(offset_idx-1);
        last_pose.tz = odom_data.tz(offset_idx-1);
    end 
end