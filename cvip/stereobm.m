classdef stereobm < handle
    % STEREOBM
    %
    % Compute block matching given a couple of rectified images and produce the
    % disparity map.
    %
    % PROPERTIES
    %
    % minDisparity:   Minimum disparity.
    % maxDisparity:   Maximum disparity.
    % winSize:        Window size [width height].
    %
    % upsample:       0 disabled, >0 enabled
    % subpixel:       0 disabled, 1 enabled
    % leftRightCheck: 0 disabled, 1 enabled
    %
    % badMatch:       Structure with masks of bad match.
    %
    % METHODS
    %
    % compute(left, right):     compute block matching for input stereo images.
    
    % Giulio Marin
    %
    % giulio.marin@me.com
    % 2015/05/15
    
    properties
        minDisparity = 0;
        maxDisparity = 100
        winSize = [0 0]
        
        % Refinement
        upsample = 0
        subpixel = 0
        leftRightCheck = 0
        
        % Additional data
        badMatch % A mask for each metric
    end
    
    methods
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute disparity map between left and right.
        function [disparity, costVolume] = compute(self, left, right)
            
            % Convert to 1 channel image.
            if size(left, 3) > 1
                left = rgb2gray(left);
            end
            if size(right, 3) > 1
                right = rgb2gray(right);
            end
            
            % Convert images.
            left = double(left);
            right = double(right);
            
            % Upsample images
            if self.upsample > 0
                self.upsample = round(self.upsample);
                self.minDisparity = self.minDisparity * self.upsample;
                self.maxDisparity = self.maxDisparity * self.upsample;
                self.winSize = self.winSize * self.upsample;
                left = imresize(left, self.upsample);
                right = imresize(right, self.upsample);
            end
            
            % Cost volume.
            costVolume = zeros(size(left, 1), size(left, 2), self.maxDisparity - self.minDisparity + 1);
            
            for dispIdx = self.minDisparity : self.maxDisparity
                
                % Compute matching cost.
                circShiftFrameSlave = circshift(right, [0, dispIdx]);
                singlePixelCost = abs(circShiftFrameSlave - left);
                
                % Aggregate matching cost.
                costVolume(:, :, dispIdx - self.minDisparity + 1) = ...
                    conv2(singlePixelCost, ones(self.winSize)/(prod(self.winSize)), 'same');
                
            end
            
            % Compute disparity = index of the minimum cost.
            [minCost, idx] = min(costVolume, [], 3);
            disparity = self.minDisparity + idx - 1;
            
            % Left-right check
            if self.leftRightCheck
                computeLeftRightCheck(self, disparity, minCost);
            end
            
            % Subpixel.
            if self.subpixel
                disparity = computeSubpixel(self, disparity, costVolume);
            end
            
            % Downsample images
            if self.upsample > 0
                disparity = imresize(disparity, 1 / self.upsample, 'nearest') / self.upsample;
            end
        end
    end
    
    methods (Access = private)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute subpixel refinement.
        function [disparityOut] = computeSubpixel(self, disparityIn, costVolume)
            
            % Allocate output.
            disparityOut = disparityIn;
            
            % Indices of minimum cost.
            idx = disparityIn + 1 - self.minDisparity;
            
            % Use parabolic interpolation of the cost
            for r = 1 : size(costVolume,1)
                for c = self.maxDisparity + 1 : size(costVolume,2)
                    if isnan(disparityIn(r,c)) || idx(r,c) == 1 || idx(r,c) == size(costVolume,3)
                        continue
                    end
                    prev = costVolume(r,c,idx(r,c) - 1);
                    curr = costVolume(r,c,idx(r,c));
                    next = costVolume(r,c,idx(r,c) + 1);
                    disparityOut(r,c) = disparityIn(r,c) + (prev - next) ./ (prev + next - 2*curr);
                end
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Compute left-right check.
        function [self] = computeLeftRightCheck(self, disparity, minCost)
            
            % Allocate output.
            self.badMatch.leftRight = false(size(disparity));
            self.badMatch.leftRight2Matches = false(size(disparity));
            
            for r = 1:size(disparity,1)
                % First row is the disparity, second row is the cost.
                minIdxCost = inf*ones(2,size(disparity,2));
                for c = 1:size(disparity,2)
                    if isnan(disparity(r,c))
                        continue
                    end
                    % Column in right image.
                    cRight = c - disparity(r,c);
                    if cRight < 1
                        continue
                    end
                    if minCost(r,c) < minIdxCost(2,cRight)
                        % One is already present.
                        if ~isinf(minIdxCost(2,cRight))
                            if abs(minIdxCost(1,cRight) - c) > 1
                                self.badMatch.leftRight(r,minIdxCost(1,cRight)) = 1;
                            end
                        end
                        minIdxCost(1,cRight) = c;
                        minIdxCost(2,cRight) = minCost(r,c);
                    elseif minCost(r,c) > minIdxCost(2,cRight)
                        if ~isinf(minIdxCost(2,cRight))
                            % There was a previous match that has lower
                            % cost. Uniqueness assumption violated.
                            self.badMatch.leftRight2Matches(r,c) = 1;
                        end
                    end
                end
            end
        end
    end
end
