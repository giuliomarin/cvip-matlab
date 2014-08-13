function export_ply( pointCloud, filePath, colorMap )
% EXPORT_PLY( pointCloud, filePath, colorMap )
%
% Converts 3D point cloud (x,y,z) to '.ply' format.
%
% pointCloud:	3D array of point cloud coordinates x,y,z (N x 3).
%               The point cloud can have additional columns for the
%               color information:
%               - Greyscale: N x 4 matrix with the additional column
%                 containing the grayscale value of the points (0-256).
%               - RGB: N x 6 with the last three columns containing the
%               red, green and blue values of the points (0-256).
%
% filePath:     Path of the '.ply' file to store
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

% Number of valid points (i.e. z coordinate not null)
idx = ~isnan(pointCloud(:,3));
totNum = sum(idx);

% Color map according to z (default grayscale)
if (size(pointCloud,2) == 3 && nargin == 3)
    figure
    colors = colormap(colorMap);
    close
    colors = interp1(1:size(colorMap,1),colors,linspace(1,size(colorMap,1),256))*256;
    z = pointCloud(:,3);
    z = z - min(z);
    z = z / max(z);
    z = round(z * 255) + 1;
    colorZ = colors(z,:);
    pointCloud(:,4:6) = colorZ;
end

% Write header
fid = fopen(filePath,'w');
fprintf(fid,'ply\n');
fprintf(fid,'format ascii 1.0\n');
fprintf(fid,'element vertex %i\n',totNum);
fprintf(fid,'property float x\n');
fprintf(fid,'property float y\n');
fprintf(fid,'property float z\n');

% Color information
if (size(pointCloud,2) > 3 || nargin == 3)
    fprintf(fid, 'property uchar red\n');
    fprintf(fid, 'property uchar green\n');
    fprintf(fid, 'property uchar blue\n');
end

fprintf(fid,'end_header\n');

% Round off color columns and make sure they are uint8
if( size(pointCloud,2) > 3)
    pointCloud(:,4:end) = uint8(round(pointCloud(:,4:end)));
end

% Write data points to file
if (size(pointCloud,2) == 3) % No color
    fprintf(fid,'%.5f %.5f %.5f\n',pointCloud(idx,:)');
elseif (size(pointCloud,2) == 4) % Greyscale
    fprintf(fid,'%.5f %.5f %.5f %u %u %u\n',[pointCloud(idx,1:3), pointCloud(idx,4), pointCloud(idx,4), pointCloud(idx,4)]');
elseif (size(pointCloud,2) == 6)  % Color
    fprintf(fid,'%.5f %.5f %.5f %u %u %u\n',pointCloud(idx,:)');
end

fclose(fid);

end