function    num = readLog(fileName) 
        fid = fopen( fileName );
        cac = textscan(fid,'%f%f%f%f%f%f%f', 'Headerlines',5, 'CollectOutput',true );
        fclose( fid );
        num = cac{1};
end