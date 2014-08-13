%% Otsu's algorithm for the case of three gray levels
%
% Assume that an image with three mainly blocks of different gray levels is
% provided as input. After the addition of some noise, the real algorithm
% is tested and the results are shown in the figures.
%
% For more information on Otsu's method visit
% <http://www.labbookpages.co.uk/software/imgProc/otsuThreshold.html>
%
% Notes: Due to the nature of the image and the pre-processing filter,
% there are some artifacts at the edges of the different segments.

% Giulio Marin
%
% giulio.marin@me.com
% 2013/01/12
%% Pre-processing

close all; clear; clc;

% Read the image
im = double(imread('test.png'));
im = im./max(im(:));
figure; imshow(im,'Border','tight'); title('Original Image')

% Add noise to the image
imn = imnoise(im,'gaussian',0,0.1);
h = fspecial('gaussian',30,5);
imn = imfilter(imn,h,'replicate');
figure; imshow(imn,'Border','tight'); title('Blurred Image')

% Compute the histogram
Prob = imhist(imn);
Prob = Prob./max(Prob);

%% Algorithm

iter = 1;
for t1 = 0:253
    for t2 = (t1+1):254
        
        % Divide the probability vector
        p1 = Prob((0:t1)+1);
        p2 = Prob((t1+1:t2)+1);
        p3 = Prob((t2+1:255)+1);
        
        % Calculate the weights
        w1 = sum(p1);
        w2 = sum(p2);
        w3 = sum(p3);
        
        if(w1==0 || w2==0 || w3==0)
            continue
        end
        
        % Compute the expectation for fixed threshold t1 and t2
        m1 = (0:t1)     *  p1	/ w1;
        m2 = (t1+1:t2)  *  p2	/ w2;
        m3 = (t2+1:255) *  p3   / w3;
        
        % Compute the variance for fixed threshold t1 and t2
        var1 = ((0:t1) - m1).^2     *  p1	/ w1;
        var2 = ((t1+1:t2) - m2).^2  *  p2	/ w2;
        var3 = ((t2+1:255) - m3).^2 *  p3   / w3;
        
        % Compute the weighted sum of whithin-class variance for fixed
        % threshold t1 and t2
        thr(:,iter) = [w1*var1 + w2*var2 + w3*var3; t1; t2];
        iter = iter + 1;
    end
end

% Find optimal threshold for minimal whithin-class variance
thr_opt = thr(2:3,find(thr(1,:) == min(thr(1,:)),1))/255;

% Perform segmentation
ims = zeros(size(im));
ims(imn <= thr_opt(1)) = 0;
ims((imn > thr_opt(1)) & (imn <= thr_opt(2))) = 1;
ims(imn > thr_opt(2)) = 2;
figure; imshow(ims/max(ims(:)),'Border','tight'); title('Segmented Image')

%% Show thresholds on the histogram

figure; plot(0:255,Prob); hold on;
plot([thr_opt(1) thr_opt(1)]*255,[0 1],'r','LineWidth',2);
plot([thr_opt(2) thr_opt(2)]*255,[0 1], 'g','LineWidth',2);
title('Histogram and thresholds')