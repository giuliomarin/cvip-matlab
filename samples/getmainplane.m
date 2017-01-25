% Load image
disparity = imread32f('/Users/giulio/Desktop/problem_files/Scan_sir30C_Birdhouse1_u009_frame11_class16.png');

% Parameters
b = 50;
fx = 450;
fy = 450;
[cy, cx] = size(disparity);
cy = cy / 2;
cx = cx / 2;
bf = b * fx;
K = [fx 0 cx; 0 fy cy; 0 0 1];

% Convert to point cloud
depth = disparity2depth(disparity, bf);
xyz = to3d(depth, K);
ptCloud = pointCloud(xyz);
pcwrite(ptCloud, 'original.ply');

% Fit plane
maxDist = 20;
[model, inlierIndices,outlierIndices] = pcfitplane(ptCloud, maxDist);
ptCloudColor = pointCloud(xyz, 'Color', uint8(repmat([255 0 0], size(xyz, 1), 1)));
plane = select(ptCloudColor,inlierIndices);
pcwrite(plane, 'plane.ply');