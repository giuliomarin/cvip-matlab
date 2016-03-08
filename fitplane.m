function [a,b,c,d] = fitplane(xyz)
% [a,b,c,d] = FITPLANE( filePath )
%
% Fit a plane on a 3D point cloud (x,y,z).
%
% INPUT
%
% xyz: 3D array of point cloud coordinates x,y,z (N x 3)
%
% OUTPUT
%
% [a,b,c,d]: coefficients of the plane that is at minimum distance
%            from all the input points. Coefficients are such that
%            ax+by+cz+d=0 where [x,y,z] is a point in xyz.

% Giulio Marin
%
% giulio.marin@me.com
% 2014/03/20

%% Script

% Compute center
C = mean(xyz);

% Compute only the valid columns of U to be faster
[U, ~, ~] = svd((xyz - repmat(C, length(xyz), 1))','econ');

% Assign coefficients
a = U(1,3);
b = U(2,3);
c = U(3,3);
d = -C * [a b c]';