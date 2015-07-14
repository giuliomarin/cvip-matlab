function [ o1, o2, o3 ] = to3d( depth, K )
% [ o1, o2, o3 ] = TO3D( depth, K )
%
% Project a 2D depth map in the 3D space.
%
% INPUT
%
% depth:     MxN depth map. Points with invalid depth are NaN.
%
% K:         3x3 camera projection matrix. Assumes C++ convention (indices
%            start from 0).
%
% OUTPUT
%
% o1:     (MxN)x3 matrix where each row represents a points and the columns
%         are x, y and z coordinates. Points not valid have NaN coordinates.
%         If also o2 and o3 are used, o1 is a vector of x components.
% o2:     Vector of y components. (Check o1 description)
% o3:     Vector of z components. (Check o1 description)

% Giulio Marin
%
% giulio.marin@me.com
% 2015/05/15

%% Compute back projection
o1 = NaN(numel(depth), 3);

i = 1;
for r = 1:size(depth, 1)
    for c = 1:size(depth, 2)
        if ~isnan(depth(r,c))
            o1(i, :) = (K \ [c-1 ; r-1; 1])' * depth(r,c);
        end
        i = i + 1;
    end
end

%% Check output format
if nargout > 1
    xyz = o1;
    o1 = xyz(:,1);
    o2 = xyz(:,2);
    o3 = xyz(:,3);
end

