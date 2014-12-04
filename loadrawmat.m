function [ m ] = loadrawmat( path, size, type )

if ~exist(path, 'file')
    disp(['File:\n' path  '\nnot present.'])
    return
end

id = fopen(path);
m = fread(id,(prod(size)),type);
fclose(id);

m = reshape(m,size(1),size(2));

end