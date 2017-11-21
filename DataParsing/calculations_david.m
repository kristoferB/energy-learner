clear all
close all
clc
%% Read energy
E_orig = importdata('Data\Raw\160406- KR30-Sensitivity Analises\Results\Peak power\160413 - E - peakPower-original.txt');
E_100 = importdata('Data\Raw\160406- KR30-Sensitivity Analises\Results\Peak power\160413 - E - peakPower-opt_simple.txt');

E_vec = [];
fileName = '';

for i = 1:9 
    fileName = sprintf('160413 - E - peakPower-%d.txt',i );
    path = strcat('Data\Raw\160406- KR30-Sensitivity Analises\Results\Peak power\',fileName);
    E_vec = [E_vec, importdata(path)];
end
%% Read trace

Trace_orig = readLog('orig_traj.txt');
Trace_opt_100 = readLog('opt_simple.txt');

Trace_vec = [];
fileName = '';

for i = 1:9 
    fileName = sprintf('%d.txt',i );
    path = strcat('Data\Raw\160406- KR30-Sensitivity Analises\Results\Peak power\', fileName);
    tr.data = readLog(path);
    Trace_vec = [Trace_vec, tr];
end

%% filter & get results

eVec = [];
maxP_filt_vec = [];
maxP_vec = [];


filt_power = david_filter( E_orig, Trace_orig );
headers = {'Time','Axis 1','Axis 2','Axis 3','Axis 4','Axis 5','Axis 6','Power [Watt]'};
toCSV = [Trace_orig filt_power] ;
csvwrite_with_headers('fitted_data_orig.csv',toCSV,headers);

filt_power = david_filter( E_100, Trace_opt_100 );
headers = {'Time','Axis 1','Axis 2','Axis 3','Axis 4','Axis 5','Axis 6','Power [Watt]'};
toCSV = [Trace_opt_100 filt_power] ;
csvwrite_with_headers('fitted_data_simple.csv',toCSV,headers);

for i=1:9
    fileName = sprintf('fitted_data_simple%d.csv',i );
    filt_power =  david_filter( E_vec(i), Trace_vec(i).data );

    headers = {'Time','Axis 1','Axis 2','Axis 3','Axis 4','Axis 5','Axis 6','Power [Watt]'};
    toCSV = [Trace_vec(i).data filt_power] ;
    csvwrite_with_headers(fileName,toCSV,headers);
end
