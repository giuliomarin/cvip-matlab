function mergeimages(pathIn1, pathIn2, pattern, pathOut)
for i = 1:100
    filename = [pathIn1, sprintf(pattern, i)];
    if ~exist(filename, 'file')
        break
    end
    img1 = imread(filename);
    
    filename = [pathIn2, sprintf(pattern, i)];
    if ~exist(filename, 'file')
        break
    end
    img2 = imread(filename);
    
    filename = [pathOut, sprintf(pattern, i)];
    if ~exist(pathOut, 'dir')
        mkdir(pathOut);
    end
    imwrite([img1, img2], filename);
end
fprintf('%d images written\n', i-1)
end