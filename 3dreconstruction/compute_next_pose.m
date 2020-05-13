 function [h_global] = compute_pose_delta(last_pose,this_pose,offset_idx)
    persistent cf0
    if isempty(cf0)
        cf0 = [last_pose.r11 last_pose.r12 last_pose.r13 last_pose.tx;
               last_pose.r21 last_pose.r22 last_pose.r23 last_pose.ty;
               last_pose.r31 last_pose.r32 last_pose.r33 last_pose.tz;
               0             0             0             1];
           
    end
    H_Transform = [this_pose.r11 this_pose.r12 this_pose.r13 1;
                   this_pose.r21 this_pose.r22 this_pose.r23 1;
                   this_pose.r31 this_pose.r32 this_pose.r33 1;
                   0                    0             0      1];
    p0 = [(this_pose.tx  - last_pose.tx) (this_pose.ty  - last_pose.ty) (this_pose.tz  - last_pose.tz) 1];
    
    transform = p0*H_Transform;
    
    cf0 = transform;
    
    h_global = cf0;
end

