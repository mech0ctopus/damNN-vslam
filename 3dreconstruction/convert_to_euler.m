function [x,y,z,roll,pitch,yaw] = convert_to_euler(r11,r12,r13,tx,r21,r22,r23,ty,r31,r32,r33,tz)
    %convert_to_euler.m converts 12 parameter pose data to euler (rpy,xyz)
    %   
    x = tx;
    y = ty;
    z = tz;
    
    roll = atan2(r32,r33);
    pitch = atan2(-r31,sqrt((r32.^2) + (r33.^2)));
    yaw = atan2(r21,r11);
    
end