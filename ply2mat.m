function [ P, C ] = ply2mat( filePath )
% PLY2MAP( filePath )
%
% Converts '.ply' format file to 3D point cloud (x,y,z).
%
% INPUT
%
% filePath:     Path to the '.ply' file store
%
% OUTPUT
%
% P:	3D array of point cloud coordinates x,y,z (N x 3).
%
% C:    3D array of RGB values for each point (N x 3).

% Giulio Marin
%
% giulio.marin@me.com
% 2014/11/07

%% Script

% Open file
fid = fopen(filePath,'r');

% Get number of points in the file
n = fscanf(fid, ['ply\n'...
    'format ascii 1.0\n'...
    'element vertex %d\n'...
    'property float x\n'...
    'property float y\n'...
    'property float z\n'...   
    'property uchar red\n'...
    'property uchar green\n'...
    'property uchar blue\n'...
    'end_header']);

% Allocate output matrices
P = zeros(n,3);
C = zeros(n,3);

% Fill output matrices
for i=1:n
    [values,~] = fscanf(fid, '%f %f %f %d %d %d\n',6);
    P(i,1:3) = values(1:3,1);
    C(i,1:3) = values(4:6,1);
end

% Close file
fclose(fid);

end

