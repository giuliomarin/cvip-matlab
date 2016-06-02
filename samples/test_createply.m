% Create a point cloud and relative mesh.

% Giulio Marin
%
% giulio.marin@me.com
% 2016/01/19

%% Preliminaries

close all

addpath(genpath('../'));

pathDispImg = 'images/disp.png';

%% Load images

disparity = double(imread(pathDispImg)) + 270;

%% Create point cloud and mesh

% Convert disparity to depth
depth = 560 ./ disparity;
depth(depth <= 0 | isinf(depth)) = NaN;

% Create 3D points
points = to3d(depth, [3740 0 641; 0 3740 555; 0 0 1]);
points(isnan(points(:,3)), 3) = 0;

% Create mesh
rows = size(depth, 1);
cols = size(depth, 2);
faces = zeros((rows - 1) * (cols - 1), 4);

% First row
for c = 0 : cols - 2
    faces(c+1, 1) = c;
    faces(c+1, 2) = c + 1;
    faces(c+1, 3) = cols + c + 1;
    faces(c+1, 4) = cols + c;
end

% All the other rows
for r = 1 : rows - 2
    faces(r * (cols - 1) + 1 : (r + 1) * (cols - 1), :) = faces((r-1) * (cols - 1) + 1 : r * (cols - 1), :) + cols;
end

%% Remove bad faces
idxValidPoints = points(:,3) > 0;

% including at least one invalid point
idxFacesToRemove = ~all(idxValidPoints(faces+1)');

% or too slanted surfaces
z = points(:,3);
zFaces = z(faces+1)';
deltaZ = max(zFaces) - min(zFaces);
idxFacesToRemove = idxFacesToRemove | (deltaZ > 0.02);

faces(idxFacesToRemove, :) = [];

%% Save ply
mat2ply(points, './pointcloud.ply', faces, jet);