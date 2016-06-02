function mat2ply( pointCloud, filePath, faces, colorMap )
% MAT2PLY( pointCloud, filePath, faces, colorMap )
%
% Converts 3D point cloud (x,y,z) to '.ply' format.
%
% INPUT
%
% pointCloud:	3D array of point cloud coordinates x,y,z (N x 3).
%               The point cloud can have additional columns for the
%               color information:
%               - Greyscale: N x 4 matrix with the additional column
%                 containing the grayscale value of the points (0-255).
%               - RGB: N x 6 with the last three columns containing the
%               red, green and blue values of the points (0-255).
%
% filePath:     Path of the '.ply' file to store
%
% faces:        (Optional) M x 4 matrix of faces. Each line specifies the
%               index (starting from 0) of points representing the
%               vertices. Can be empty. The list of points must be
%               clockwise.
%
% colorMap:     (Optional) Name of colormap to use if a Nx3 matrix is given
%               as input to pointCloud.

% Giulio Marin
%
% giulio.marin@me.com
% 2014/03/15

%% Script

% Check input format
if (size(pointCloud,2) < 3 || ...
        size(pointCloud,2) > 6 || ...
        size(pointCloud,2) == 5)
    fprintf('Input format not valid\n')
    return
end

% Number of valid points (coordinates not null)
notValid = isnan(pointCloud(:,1)) | isnan(pointCloud(:,2)) | isnan(pointCloud(:,3));
pointCloud(notValid, :) = 0;
if nargin < 3 || isempty(faces)
    pointCloud(notValid, :) = [];
end
totNum = size(pointCloud,1);

% Color map according to z (default grayscale)
if (size(pointCloud,2) == 3 && nargin == 4)
    figure
    colors = colormap(colorMap);
    close
    colors = interp1((1:size(colorMap,1))-1,colors,linspace(1,size(colorMap,1),256)-1)*255;
    z = pointCloud(:,3);
    z = z - min(z);
    if max(z) > 0
        z = z / max(z);
    end
    z = round(z * 255) + 1;
    colorZ = colors(z,:);
    pointCloud(:,4:6) = colorZ;
end

% Write header
fid = fopen(filePath,'w');
fprintf(fid, 'ply\n');
fprintf(fid, 'format ascii 1.0\n');
fprintf(fid, 'element vertex %i\n',totNum);
fprintf(fid, 'property float x\n');
fprintf(fid, 'property float y\n');
fprintf(fid, 'property float z\n');

% Color information
if (size(pointCloud,2) > 3 || nargin == 4)
    fprintf(fid, 'property uchar red\n');
    fprintf(fid, 'property uchar green\n');
    fprintf(fid, 'property uchar blue\n');
    fprintf(fid, 'property uchar alpha\n');
end

if nargin > 2 && ~isempty(faces)
    fprintf(fid, 'element face %i\n', size(faces,1));
    fprintf(fid, 'property list uchar int vertex_index\n');
end

fprintf(fid,'end_header\n');

% Round off color columns and make sure they are uint8
if( size(pointCloud,2) > 3)
    pointCloud(:,4:end) = uint8(round(pointCloud(:,4:end)));
end

% Write data points to file
if (size(pointCloud,2) == 3) % No color
    fprintf(fid,'%.3f %.3f %.3f\n',pointCloud');
elseif (size(pointCloud,2) == 4) % Greyscale
    fprintf(fid,'%.3f %.3f %.3f %u %u %u %u\n',[pointCloud(:,1:3), pointCloud(:,4), pointCloud(:,4), pointCloud(:,4), zeros(length(pointCloud),1)]');
elseif (size(pointCloud,2) == 6)  % Color
    fprintf(fid,'%.3f %.3f %.3f %u %u %u %u\n',[pointCloud, zeros(length(pointCloud),1)]');
end

% TODO: deal with faces with 3 points.
if nargin > 2 && ~isempty(faces)
    faces = [4 * ones(size(faces,1),1), faces];
    fprintf(fid,'%i %i %i %i %i\n',faces');
end

% Close file
fclose(fid);

end