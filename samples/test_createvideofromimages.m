% Create a video sequence from a set of images.

% Giulio Marin
%
% giulio.marin@me.com
% 2015/08/10

%% Preliminaries

addpath(genpath('../'));

%% Create images

img = imread('images/left.png');
imgsRgb = zeros(size(img, 1), size(img, 2), 3, 31, 'uint8');
imgsRgb(:,:,:,1) = img;

for n = 2:size(imgsRgb,4)
    imgsRgb(:,:,:,n) = circshift(img, round((n - 1) * size(img, 2) / 30), 2);
end

imgsGray = uint8(mean(imgsRgb, 3));

%% Create video

% Color not compressed
createvideofromimages(imgsRgb, 'images/video_rgb', 30, false);

% Color compressed
createvideofromimages(imgsRgb, 'images/video_rgb_compressed', 30, true);

% Grayscale not compressed
createvideofromimages(imgsGray, 'images/video_gray', 30, false);

% Grayscale compressed
createvideofromimages(imgsGray, 'images/video_gray_compressed', 30, true);