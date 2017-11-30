clear all
close all
clc
%% Rewrite data 

E_vec = [];
fileName = '';

for i = 1:5 
    fileName = sprintf('EL_power_%d.txt',i );
    path = strcat('Data\Raw\data_psl_28_11\',fileName);
    comma2point_overwrite( path )
end
for i = 1:5 
    fileName = sprintf('demoTrajEL%d.txt',i );
    path = strcat('Data\Raw\data_psl_28_11\',fileName);
    comma2point_overwrite( path )
end

%% Read energy

E_vec = [];
fileName = '';

for i = 1:5 
    fileName = sprintf('EL_power_%d.txt',i );
    path = strcat('Data\Raw\data_psl_28_11\',fileName);
    E_vec = [E_vec, importdata(path)];
%     E_vec = importdata(path);
end

%% Read trace

Trace_vec = [];
fileName = '';

for i = 1:5
    fileName = sprintf('demoTrajEL%d.txt',i );
    path = strcat('Data\Raw\data_psl_28_11\', fileName);
    tr.data = readLog(path);
    Trace_vec = [Trace_vec, tr];
end

%% filter & get results

eVec = [];
maxP_filt_vec = [];
maxP_vec = [];


for i=1:5
    fileName = sprintf('EL_data_fitted%d.csv',i );
    filt_power =  david_filter( E_vec(i), Trace_vec(i).data );

    headers = {'Time','Axis 1','Axis 2','Axis 3','Axis 4','Axis 5','Axis 6','Power [Watt]'};
    toCSV = [Trace_vec(i).data filt_power] ;
    csvwrite_with_headers(fileName,toCSV,headers);
end
