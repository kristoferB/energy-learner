clear variables
clc, close all

rawData = importdata('160429 - E - R30_original.csv');
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
figure(2)
plot(total_power)
%% Filter and downsample active power

%close all
figure(3)
active_power = downsample(total_power,0.012*10000);
plot(active_power,'b')
hold on

% ---- TEST DIFFERENT FILTERS
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
% plot(filt_power,'green')
% -----

% filt_power = conv(active_power,b);
plot(filt_power,'r')

% -- Display Filter --
%figure(4)
%freqz(b,1,512)
%%

save('eOrig_R30.mat','total_energy','energy_noCabinet','consumedE','regenE','cycleTime','peakPower','filt_power');
temp = [filt_power*2 filt_power];
csvwrite('test.csv',temp);