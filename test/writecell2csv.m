function writecell2csv(fileName, cellArray)
% Writes cell into a *.csv file.
% 
% WRITECELL2CSV(fileName, cell, separator, excelYear, decimal)
%
% fileName     = Name of the file to save. [ i.e. 'text.csv' ]
% cellArray    = Name of the Cell Array where the data is in


%% Useful quantities.
separator = ',';
decimal = '.';


%% Write file
datei = fopen(fileName, 'w');

for r = 1:size(cellArray, 1)
    for c = 1:size(cellArray, 2)
        
        var = eval('cellArray{r, c}');
        
        % If zero, then empty cell.
        if size(var, 1) == 0
            var = '';
        end
        
        % If numeric -> String.
        if isnumeric(var)
            var = num2str(var);
            % Conversion of decimal separator (4 Europe & South America)
            % http://commons.wikimedia.org/wiki/File:DecimalSeparator.svg
            if decimal ~= '.'
                var = strrep(var, '.', decimal);
            end
        end
        
        % If logical -> 'true' or 'false'.
        if islogical(var)
            if var == 1
                var = 'TRUE';
            else
                var = 'FALSE';
            end
        end
        
        % Quotes 4 Strings.
        %var = ['"' var '"']; %#ok<AGROW>
                
        % OUTPUT value
        fprintf(datei, '%s', var);
        
        % OUTPUT separator
        if c ~= size(cellArray, 2)
            fprintf(datei, separator);
        end
        
    end
    
    if r ~= size(cellArray, 1) % prevent a empty line at EOF
        % OUTPUT newline
        fprintf(datei, '\n');
    end
end

% Closing file
fclose(datei);
% END