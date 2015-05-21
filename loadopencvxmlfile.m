function [ content ] = loadopencvxmlfile( filepath )
% LOADOPENCVXMLFILE( filepath )
%
% Load data in the xml file stored with OpenCV.
%
% INPUT
%
% filePath:     Path to the OpenCV xml file.
%
% OUTPUT
%
% content:     Structure with data in the file.
%
% DEPENDENCIES
%
% parsexml.m

% Giulio Marin
%
% giulio.marin@me.com
% 2015/05/15

%% Parse file
param = parsexml(filepath);

for moduleIdx = 2:2:size(param.Children, 2) - 1
        name = param.Children(moduleIdx).Name;
        if size(param.Children(moduleIdx).Children, 2) >= 8
            % Is a Mat
            rows = str2double(param.Children(moduleIdx).Children(2).Children.Data);
            cols = str2double(param.Children(moduleIdx).Children(4).Children.Data);
            dataRaw = param.Children(moduleIdx).Children(8).Children.Data;
            data = reshape(strread(dataRaw, '%f'), [cols, rows])';
        else
            % Is a single value
            % If not a string, convert to double
            data = param.Children(moduleIdx).Children.Data;
            if ~strcmp(data(1), '"')
                data = str2double(param.Children(moduleIdx).Children.Data);
            end
        end
        
        % Add new entry
        content.(name) = data;
end

end

