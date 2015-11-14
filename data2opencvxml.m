function data2opencvxml( data, filename )
% DATA2OPENCVXML( mat, filename )
%
% Save data according to OpenCV XML format. Input data is a struct with
% multiple fields that can be strings, numbers or matrices.
%
% Example:
%
% data = struct;
% data.vector = 1:10;
% data.matrix: rand(3);
% data.text = 'This is a sting';
% data.number = 10;
%
% INPUT
%
% data:     struct with multiple fields (string, number, matrix).
%
% filename: path to the file to be saved.

% Giulio Marin
%
% giulio.marin@me.com
% 2015/10/12

%% Store matrices

% Write header
fid = fopen(filename,'w');
fprintf(fid, '%s\n', '<?xml version="1.0" encoding="UTF-8"?>');
fprintf(fid, '%s\n', '<opencv_storage>');

% Get fields
fields = fieldnames(data);
for i = 1:numel(fields)
    currData = data.(fields{i});
    if ischar(currData) % String
       fprintf(fid, '%s', writeString(currData, fields{i}));
    elseif isscalar(currData) % Number
        fprintf(fid, '%s', writeNumber(currData, fields{i}));
    else % Matrix
        fprintf(fid, '%s', writeMatrix(currData, fields{i}));
    end

end

% Close file
fprintf(fid, '%s', '</opencv_storage>');
fclose(fid);

end

%% Auxiliary functions

function str = writeNumber(mat, name)
    str = sprintf('<%s>', name);
    str = [str sprintf('%f ', mat)];
    str = [str sprintf('</%s>\n', name)];
end

function str = writeString(mat, name)
    str = sprintf('<%s>', name);
    str = [str sprintf('%s ', mat)];
    str = [str sprintf('</%s>\n', name)];
end

function str = writeMatrix(mat, name)
    str = sprintf('<%s %s\n', name, 'type_id="opencv-matrix">');
    str = [str sprintf('\t<rows>%d</rows>\n', size(mat, 1))];
    str = [str sprintf('\t<cols>%d</cols>\n', size(mat, 2))];
    str = [str sprintf('\t<dt>d</dt>\n')];
    str = [str sprintf('\t<data>')];
    str = [str sprintf('%f ', reshape(mat',1,{}))];
    str = [str sprintf('</data>\n')];
    str = [str sprintf('</%s>\n', name)];
end