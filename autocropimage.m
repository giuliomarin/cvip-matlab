function out = autocropimage(src)
% dst = AUTOCROPIMAGE(src)
%
% Assuming the top left pixel is the background color to remove, output an
% image with the region containing that value removed. It returns the image
% containing the biggest rectable of valid pixels and prints the range of
% valid columns and rows.
%
% INPUT
%
% src: image to crop with 1 or more channels.
%
% OUTPUT
%
% dst: image cropped to valid pixels only.

% Giulio Marin
%
% giulio.marin@me.com
% 2015/10/30


% Mask of pixels equal to the top left value
mask = mean(src,3);
mask = mask == mask(1,1);

% Get range of valid pixels
validRows = find(any(~mask,2));
validCols = find(any(~mask,1));

% Crop image
out = src(min(validRows):max(validRows),min(validCols):max(validCols), :);

% Print valid range
fprintf('valid rows=%d:%d\nvalid cols=%d:%d\n', min(validRows), max(validRows), min(validCols), max(validCols));
