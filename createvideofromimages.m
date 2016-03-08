function createvideofromimages( img, filename, fps, compressed )
% CREATEVIDEOFROMIMAGES( img, filename, fps, compressed )
%
% Create a video sequence of the images given in input.
%
% INPUT
%
% img:        MxNx1xK or MxNx3xK matrix of images with 1 or 3 channels.
%
% filename:   File name of the video sequence (without extension).
%
% fps:        Number of frames per second (30 by default).
%
% compressed: Boolean value, true if the video has to be compressed (false by default).

% Giulio Marin
% giulio.marin@me.com
%
% 2015/06/23

%% Check input

% FPS
if nargin < 3
    fps = 30;
end

% Compression
if nargin < 4
    compressed = false;
end

if compressed
    profile = 'Motion JPEG AVI';
else
    if size(img, 3) == 3
        profile = 'Uncompressed AVI';
    elseif size(img, 3) == 1
        profile = 'Grayscale AVI';
    end
end

%% Set properties
writerObj = VideoWriter(filename, profile);
writerObj.FrameRate = fps;

%% Write images
open(writerObj);
writeVideo(writerObj, img);
close(writerObj);

end