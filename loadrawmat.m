function [ m ] = loadrawmat( path, size, type )
% LOADRAWMAT( path, size, type )
%
% Load a matrix stored in binary format.
%
% INPUT
%
% path: path to the matrix
%
% size: [rows cols] of the matrix
%
% type: matrix element format. E.g. 'float', 'int', 'double', etc.

% Giulio Marin
%
% giulio.marin@me.com
% 2013/08/12

if ~exist(path, 'file')
    error(['File: ' path  ' not present.'])
end

id = fopen(path);
m = fread(id,(prod(size)),type);
fclose(id);

m = reshape(m,size(2),size(1))';

end