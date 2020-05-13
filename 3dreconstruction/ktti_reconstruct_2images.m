%% ktti_reconstruct_2images.m
% parses files in depth prediction path
% merges rgb sample
% runs ICP with ground truth odometry
% just 2 images to show point cloud registration concept

clear all
clear compute_next_pose
clc

path = './poses/07.txt';

odom_data = get_odom_data(path);

myFiles = dir(fullfile('./depth_predict','*.png')); %gets all wav files in struct

k = 148;
baseFileName = myFiles(k).name;
fullFileName = fullfile('./depth_predict', baseFileName);

depthimg = imread(fullFileName);
depth = rgb2gray(depthimg);
depth = double(depth);

img = imread(fullfile('./rgb',baseFileName));

% Find out how many rows and columns are in the image.
  [rows, columns, numberOfColorChannels] = size(depth);
  
  % Use meshgrid() to get arrays of all possible x and y coordinates in the image.
  [x, y] = meshgrid(1:columns, 1:rows);
  x = x(:);
  y = y(:);
  z = depth(:);
  pts = [z(:),x(:),y(:)];
  
  ptCloudRef = pointCloud(pts,'Color',reshape(img,[],3));
  
k = k + 1;
baseFileName = myFiles(k).name;
fullFileName = fullfile('./depth_predict', baseFileName);

depthimg = imread(fullFileName);
depth = rgb2gray(depthimg);
depth = double(depth);

z = depth(:);

pts = [z(:),x(:),y(:)];

img = imread(fullfile('./rgb',baseFileName));

ptCloudCurrent = pointCloud(pts,'Color',reshape(img,[],3));

gridSize = 0.003;
fixed = pcdownsample(ptCloudRef, 'gridAverage', gridSize);
moving = pcdownsample(ptCloudCurrent, 'gridAverage', gridSize);
%pcshowpair(moving,fixed,'VerticalAxis','Y','VerticalAxisDir','Down');


    [x,y,z,r,p,yaw] = convert_to_euler(odom_data.r11(k),odom_data.r12(k),odom_data.r13(k), ...
                                 odom_data.tx(k),odom_data.r21(k), odom_data.r22(k), ...
                                 odom_data.r23(k), odom_data.ty(k),odom_data.r31(k), ...
                                 odom_data.r32(k), odom_data.r33(k), odom_data.tz(k));
    
    vo_tform_rot = [cos(yaw) sin(yaw) 0; ...
                    -sin(yaw) cos(yaw) 0;
                    0                 0                1]
                    
    vo_tform_trans = [odom_data.tx(k); odom_data.ty(k); odom_data.tz(k)]
    vo_tform = rigid3d(vo_tform_rot,vo_tform_trans');
    tform = pcregistericp(moving,fixed,'Extrapolate',true,'Metric','pointToPlane','InlierRatio',0.1,'Tolerance',[0.00001,0.00005],"InitialTransform",vo_tform);
ptCloudAligned = pctransform(ptCloudCurrent,tform);

mergeSize = 0.001;
ptCloudScene = pcmerge(ptCloudRef, ptCloudAligned, mergeSize);

pcshow(ptCloudScene);
