function img = imread32f(filename)
% img = IMREAD32F(filename)
%
% Load a float PNG image saved as 4 channel image.
%
% INPUT
%
% filename: path to the png image.
%
% OUTPUT
%
% img: image loaded with float precision (single).

% Giulio Marin
%
% giulio.marin@me.com
% 2015/04/29


% Original data are in RGB and alpha channels (4 channels of 8 bytes)
[RGB, ~, A] = imread(filename);
RGB = uint32(RGB);
A = uint32(A);

% Create 'single' precision matrix
origData = RGB(:,:,3); % B
origData = bitor(origData, bitshift(RGB(:,:,2), 8)); % G
origData = bitor(origData, bitshift(RGB(:,:,1), 16)); % R
origData = bitor(origData, bitshift(A, 24)); % A
origData = typecast(origData(:), 'single');
img = double(reshape(origData, size(RGB, 1), size(RGB, 2)));
end
