function imwrite32f(img, filename)
% IMWRITE32F(img, filename)
%
% Save a float matrix to a PNG image. Use 4 channel to store the
% information.
%
% INPUT
%
% img: image to store with float precision (single).
%
% filename: path to the png image.

% Giulio Marin
%
% giulio.marin@me.com
% 2015/04/29


% Get size of the image
[rows, cols] = size(img);

% Cast data to 'uint32' (32 bits)
imgVec = typecast(single(img(:)), 'uint32');
imgCast = reshape(imgVec, rows, cols);

% Save data in 4 channels (BGR+A for compatibility with OpenCV)
maskChannel = repmat(uint32(255), rows, cols);
RGB = zeros(rows, cols, 3, 'uint8'); 
RGB(:,:,1) = bitshift(bitand(imgCast, bitshift(maskChannel, 16)), -16); % R
RGB(:,:,2) = bitshift(bitand(imgCast, bitshift(maskChannel, 8)), -8); % G
RGB(:,:,3) = bitand(imgCast, maskChannel); % B
A = uint8(bitshift(bitand(imgCast, bitshift(maskChannel, 24)), -24)); % A

% Save image to file 
imwrite(RGB, filename,'Alpha', A);
end
