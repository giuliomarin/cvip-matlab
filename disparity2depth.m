function [ depth ] = disparity2depth( disparity, baselinefocal )
% DISPARITY2DEPTH( disparity, baselinefocal )
%
% Compute a depth map given a disparity map and the product baseline * focal length.
% It can be used also to compute the opposite, given a depth map, compute
% the disparity map.
%
% INPUT
%
% disparity:     MxN disparity map [pixel]
%
% baselinefocal: Product baseline [mm] * focal length [pixel]
%
% OUTPUT
%
% depth: MxN depth map [mm]. Points with invalid depth are NaN.

% Giulio Marin
%
% giulio.marin@me.com
% 2015/05/15

%% Compute depth.
depth = baselinefocal ./ disparity;

% Fix invalid values.
depth(depth <= 0) = NaN;
depth(isinf(depth)) = NaN;

end

