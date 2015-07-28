% Compute the disparity map given two rectified images.

% Giulio Marin
%
% giulio.marin@me.com
% 2015/07/28

%% Preliminaries

close all

addpath(genpath('../'));

pathLeftImg = 'images/left.png';
pathRightImg = 'images/right.png';

%% Load images

leftImg = imread(pathLeftImg);
rightImg = imread(pathRightImg);

%% Compute disparity

bm = stereobm;
bm.winSize = [9 9];
bm.subpixel = 1;

[disparity, cost] = bm.compute(leftImg, rightImg);

%% Show results

figure;
subplot(2,2,1)
imagesc(leftImg)
title('Left image')
axis image

subplot(2,2,2);
imagesc(rightImg)
title('Right image')
axis image

subplot(2,2,3);
imagesc(disparity)
colorbar
colormap jet
title('Disparity map')
axis image

%% Show cost

subplot(2,2,4);
imagesc(min(cost, [], 3))
colorbar
title('Minimum cost')
axis image