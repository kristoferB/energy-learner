clear variables
clc, close all

rawData = importdata('Data\Raw\R30\E-R30_original.csv');
I = rawData.data(:,2);
V = rawData.data(:,3);

power = V.*I; %Watt
P_cabinet = 200; %Watt

plot(power);

%% Choose data range
ns = 1.608e4; %Start
nf = 4.207e5; %End

sampleFrequency = 10000; % 10k Hz
cycleTime = (nf-ns)/sampleFrequency;

% Extract voltage and current in the cycle
I = rawData.data(ns:nf,2);
V = rawData.data(ns:nf,3);

%% Energy and power, including cabinet and regeneration
total_power = V.*I - P_cabinet; %Watt
total_energy = sum(total_power)/sampleFrequency / 1000; %kW.s (kJouls)

%energy_noCabinet = total_energy; %- (P_cabinet)*(nf-ns)*0.0001 / 1000;%  rms(P_cabinet)*(ns*0.0001)/ 1000 

%% Regeneration
regenPower  = (total_power<0).*(total_power);
consumedPower = (total_power>0).*(total_power);

regenE = sum(regenPower)*0.0001 / 1000; %kW.s (kJouls)
consumedE = sum(consumedPower)*0.0001 / 1000; %kW.s (kJouls)

%% Peak power
%peakPower = (max(consumedPower)- P_cabinet)/1000 ; %kWatt
peakPower = max(consumedPower)/1000 ; %kWatt

%%
figure(2), hold on
plot(total_power)
plot(consumedPower)
%% Filter and downsample active power

r30_sampling = 0.012; %12ms

active_power = downsample(total_power, r30_sampling*sampleFrequency);
used_power = downsample(consumedPower, r30_sampling*sampleFrequency);


figure(3)
plot(active_power,'b')
hold on

% ---- TEST DIFFERENT FILTERS
% n = 100;
% Wn = 0.1; % Frequency cutoff
% b = fir1(n,Wn);
% filt_power = conv(active_power,b);
% % plot(filt_power,'black')
% 
% n = 24;
% Wn = 0.1; % Frequency cutoff
% b = fir1(n,Wn);
% filt_power = conv(active_power,b);
% % plot(filt_power,'yellow')
% 
% n = 48;
% Wn = 0.2; % Frequency cutoff
% b = fir1(n,Wn);
% filt_power = conv(active_power,b);
% % plot(filt_power,'magenta')

n = 12;
Wn = 0.2; % Frequency cutoff
b = fir1(n,Wn);
filt_power = conv(active_power,b);
filt_usedPower = conv(used_power,b);

filt_power = filt_power(1:end-100);
filt_usedPower = filt_usedPower(1:end-100);

figure(3)
plot(filt_power,'r')

% -- Display Filter --
% figure(4)
% freqz(b,1,512)

%% READ TXT trajectorieis

filename = 'Data\Original\R30\orig_traj.txt';
delimiterIn = ' ';
headerlinesIn = 5;
A = importdata(filename,delimiterIn,headerlinesIn);
A.data = A.data(1:end-5 , :);

%%

% save('eOrig_R30.mat','total_energy','energy_noCabinet','consumedE','regenE','cycleTime','peakPower','filt_power');
headers = {'Time','Axis 1','Axis 2','Axis 3','Axis 4','Axis 5','Axis 6','Power [Watt]','Consumed Power [Watt]'};
toCSV = [A.data filt_power filt_usedPower];
csvwrite_with_headers('fitted_data.csv',toCSV,headers);

