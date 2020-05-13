function [transform] = compute_transform(x,y,z,this_pose)

    H_Transform = [this_pose.r11 this_pose.r12 this_pose.r13 1;
                   this_pose.r21 this_pose.r22 this_pose.r23 1;
                   this_pose.r31 this_pose.r32 this_pose.r33 1;
                   0                    0             0      1];
    p0 = [this_pose.tx this_pose.ty this_pose.tz 1];
    
    transform = p0*H_Transform;
    
end
