function drawCameraPoses(rvec, tvec, flag)

if (size(tvec,2) ~= 3 || size(tvec,1) <= 0)
    disp('tvec must contain 3 columns and greater than 0 row') ;
    exit ;
end

if (size(rvec) ~= size(tvec))
    disp('rvec must be at the same size of the tvec.') ;
end

if ~flag
    %raw rvec tvec
    for i = 1:size(tvec,1)
        rotM = rodrigues(rvec(i,:)) ;
        tvec(i,:) = -rotM'* tvec(i,:)' ;
        rvec(i,:) = rodrigues(rotM') ;
    end
end

%plot camera positions
figure ;
%watch out xyz order
plot3(tvec(:,1), tvec(:,2), tvec(:,3),'r+') ;
hold on ;
grid on ;

xlim([-1 1]) ;
ylim([-1 1]) ;
zlim([-1 1]) ;

%plot camera orientation
baseSymbol = 0.2 * [0,-0.4,-0.4,0, 0.4,0.4,0,-0.4,0.4, 0.4,-0.4;...
0,-0.3, 0.3,0,-0.3,0.3,0, 0.3,0.3,-0.3,-0.3;...
0, 1, 1,0, 1,1,0, 1,1, 1, 1];%[0 1 0 0 0 0;0 0 0 1 0 0;0 0 0 0 0 1] ;
for i = 1 : size(rvec,1)
   %rotM already transposed
   rotM = rodrigues(rvec(i,:)) ;
   baseK = rotM * baseSymbol + tvec(i,:)' * ones(1,11) ;
   plot3(baseK(1,:),baseK(2,:), baseK(3,:),'-b') ;
end
end