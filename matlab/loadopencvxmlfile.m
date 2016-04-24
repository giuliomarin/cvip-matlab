function [ content ] = loadopencvxmlfile( filepath )
% [content] = LOADOPENCVXMLFILE( filepath )
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
content = parseChildren(param);

end

% Recursively get nodes
function content = parseChildren(node)
for moduleIdx = 1:size(node.Children, 2)
    if numel(node.Children(moduleIdx).Children) == 0
        % Is an empty node
        
        continue
    elseif numel(node.Children(moduleIdx).Children) == 1
        % Is a single value or a sequence
        
        data = node.Children(moduleIdx).Children.Data;
        % If not a string, convert to double
        if ~strcmp(data(1), '"')
            data = str2num(node.Children(moduleIdx).Children.Data);
        end
        
        % Add new entry
        content.(node.Children(moduleIdx).Name) = data;
    elseif numel(node.Children(moduleIdx).Children) == 8 && (str2double(node.Children(moduleIdx).Children(2).Children.Data) > 0) && (str2double(node.Children(moduleIdx).Children(4).Children.Data) > 0)
        % Is a matrix
        
        rows = str2double(node.Children(moduleIdx).Children(2).Children.Data);
        cols = str2double(node.Children(moduleIdx).Children(4).Children.Data);
        dataRaw = node.Children(moduleIdx).Children(8).Children.Data;
        data = reshape(strread(dataRaw, '%f'), [cols, rows])';
        
        % Add new entry
        content.(node.Children(moduleIdx).Name) = data;
    else
        contentChildren = parseChildren(node.Children(moduleIdx));
        content.(node.Children(moduleIdx).Name) = contentChildren;
    end
end
end
