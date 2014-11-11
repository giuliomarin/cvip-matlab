function [ P, C ] = ply2mat( filePath )
% PLY2MAP( filePath )
%
% Converts '.ply' format file to 3D point cloud (x,y,z).
%
% INPUT
%
% filePath: Path to the '.ply' file store
%
% OUTPUT
%
% P: 3D array of point cloud coordinates x,y,z (N x 3).
%
% C: 3D array of RGB values for each point (N x 3).

% Giulio Marin
%
% giulio.marin@me.com
% 2014/11/07

%% Script

% Open file
fid = fopen(filePath,'r');

% Get number of points
found = false;
currLine = '';
while ~found
   currLine = fgetl(fid);
   found = ~isempty(strfind(currLine,'element vertex'));
end
nPoints = sscanf(currLine, 'element vertex %d\n');

found = false;
while ~found
   currLine = fgetl(fid);
   found = ~isempty(strfind(currLine,'end_header'));
end

% Allocate output matrices
P = zeros(nPoints,3);
C = zeros(nPoints,3);

% Fill output matrices
for i=1:nPoints
    [values,~] = fscanf(fid, '%f %f %f %d %d %d %d\n',7);
    P(i,1:3) = values(1:3,1);
    C(i,1:3) = values(4:6,1);
end

% Close file
fclose(fid);

end

