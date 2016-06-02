% Generate synthetic view given a top view image

% Giulio Marin
%
% giulio.marin@me.com
% 2016/02/01

%% Preliminaries

% close all

addpath(genpath('../'));

%% Parameters

topView = imread('images/topview.jpg');
z0 = 5; % distance of the camera from the ground in the top view
REAL_DEPTH = 0;

% Intrinsic parameters
cols = size(topView, 2);
rows = size(topView, 1);
fx = cols / 2;
fy = fx;
cx = cols / 2;
cy = rows / 2;

% Extrinsic parameters
tx = 1;
ty = 8;
tz = 1;
rotAxis = [1, 0, 0];
rotAngle = 60; % degrees

% Create matrices
K = [fx,  0, cx; 0, fy, cy; 0,  0,  1];
R = rodrigues(rotAxis/norm(rotAxis) * rotAngle/180*pi);
t = [tx; ty; tz];

%% Generate synthetic view

[uu, vv] = meshgrid(1:cols, 1:rows);
if REAL_DEPTH
    % Use real depth
    zz = imread32f('images/depth.png');
else
    % Generate fake depth map
    zz = ones(size(topView,1), size(topView,2)) * z0;
end
uvz = [uu(:) .* zz(:), vv(:) .* zz(:), zz(:)]';

% Point cloud top view
P0 = K \ uvz;
colors = double([reshape(topView(:,:,1), [], 1), reshape(topView(:,:,2), [], 1), reshape(topView(:,:,3), [], 1)]);

% Point cloud synthetic view after rototranslation
P1 = R' * (P0 - repmat(t, 1, size(P0, 2)));

% Project to the camera
p1 = K * (P1 ./ repmat(P1(3, :), 3, 1));

% Keep only points inside the camera
p1Valid = [p1' colors];
p1Valid(p1(1,:) < 1 | p1(1,:) > cols | p1(2,:) < 1 | p1(2,:) > rows | (P1(3,:) <= 0), :) = [];
synthView = uint8(zeros(rows, cols, 3));
for i = 1: size(p1Valid,1)
    synthView(floor(p1Valid(i,2)), floor(p1Valid(i,1)), :) = p1Valid(i,[4,5,6]);
    synthView(floor(p1Valid(i,2)), ceil(p1Valid(i,1)), :) = p1Valid(i,[4,5,6]);
    synthView(ceil(p1Valid(i,2)), floor(p1Valid(i,1)), :) = p1Valid(i,[4,5,6]);
    synthView(ceil(p1Valid(i,2)), ceil(p1Valid(i,1)), :) = p1Valid(i,[4,5,6]);
end

%% Show images

% Top view
figure(1)
imagesc(topView)
title('Top view')
axis image

% Synthetic view
figure(2)
imagesc(synthView)
title('Synthetic view')
axis image
