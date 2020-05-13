%%
headWidth = 8;
headLength = 8;
LineLength = 0.08;
odom_data = get_odom_data('./poses/07.txt');
odom_data_sift = get_odom_data('./poses/sift_07.txt');
odom_data_damnn = get_odom_data('./poses/odom07pred2.txt');

[x,y,z,u,v,w] = convert_to_euler(odom_data.r11(:),odom_data.r12(:),odom_data.r13(:), ...
                                 odom_data.tx(:),odom_data.r21(:), odom_data.r22(:), ...
                                 odom_data.r23(:), odom_data.ty(:),odom_data.r31(:), ...
                                 odom_data.r32(:), odom_data.r33(:), odom_data.tz(:));

odom_data = odom_data_sift;
[xp,yp,zp,up,vp,wp] = convert_to_euler(odom_data.r11(:),odom_data.r12(:),odom_data.r13(:), ...
                                 odom_data.tx(:),odom_data.r21(:), odom_data.r22(:), ...
                                 odom_data.r23(:), odom_data.ty(:),odom_data.r31(:), ...
                                 odom_data.r32(:), odom_data.r33(:), odom_data.tz(:));

odom_data = odom_data_damnn;
[xd,yd,zd,ud,vd,wd] = convert_to_euler(odom_data.r11(:),odom_data.r12(:),odom_data.r13(:), ...
                                 odom_data.tx(:),odom_data.r21(:), odom_data.r22(:), ...
                                 odom_data.r23(:), odom_data.ty(:),odom_data.r31(:), ...
                                 odom_data.r32(:), odom_data.r33(:), odom_data.tz(:));
x = x(1:length(xd));
y = y(1:length(xd));
z = z(1:length(xd));
u = u(1:length(xd));
v = v(1:length(xd));
w = w(1:length(xd));

xp = xp(1:length(xd));
yp = yp(1:length(xd));
zp = zp(1:length(xd));
up = up(1:length(xd));
vp = vp(1:length(xd));
wp = wp(1:length(xd));
figure;


 hq = quiver3(x,y,z,u,v,w);           %get the handle of quiver
 hold on;
 hq2 = quiver3(xp,y,zp,up,vp,wp);           %get the handle of quiver
  hold on;
 hq3 = quiver3(xd,yd,zd,ud,vd,wd);           %get the handle of quiver