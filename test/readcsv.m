function [data, header] = readcsv(fileName, delimiter, dim)

if nargin < 3
    dim = 1;
end

%% Open file
fid = fopen(fileName,'r');
lines = cell(100,1);
lineIndex = 1;
nextLine = fgetl(fid);

%% Read file
while ~isequal(nextLine,-1)
    lines{lineIndex} = nextLine;
    lineIndex = lineIndex+1;
    nextLine = fgetl(fid);
end
fclose(fid);

%% Fill fields
numberOfFields = numel(textscan(lines{1},'%s', 'Delimiter', delimiter));
data = cell(lineIndex-1,numberOfFields);
for iLine = 1:lineIndex-1
    lineData = textscan(lines{iLine},'%s', 'Delimiter', delimiter);
    lineData = lineData{1};
    data(iLine,1:numel(lineData)) = lineData;
end

if nargout == 2
    if dim == 1
        header = data(:,1);
        data = data(:, 2:end);
    elseif dim == 2
        header = data(1,:);
        data = data(2:end,:);
    end
end