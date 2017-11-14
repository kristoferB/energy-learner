clear all;
clc;
close all;

%% 
rawData = importdata('160429 - E - R10_opt_secondTry.csv');
I = rawData.data(:,2);
V = rawData.data(:,3);
power = V.*I; %Watts

plot(power);

%% Cabinet

%I_c = I(1:9.625e4,1);
%V_c = V(1:9.625e4,1);

P_cabinet = 200;% I_c.*V_c;

%% Choose data range


ns = 2.79e4;
nf = 6.732e5;

cycleTime = (nf-ns)/10000;


I = rawData.data(ns:nf,2);
V = rawData.data(ns:nf,3);

%% Energy and power, including cabinet and regeneration
total_power = V.*I - P_cabinet; %Watts

total_energy = sum(total_power)*0.0001 / 1000; %kW.s (kJouls)

energy_noCabinet = total_energy %- (P_cabinet)*(nf-ns)*0.0001 / 1000;%  rms(P_cabinet)*(ns*0.0001)/ 1000 

%% Regeneration

regenPower  = (total_power<0).*(total_power);
consumedPower = (total_power>0).*(total_power);

regenE = sum(regenPower)*0.0001 / 1000; %kW.s (kJouls)
consumedE = sum(consumedPower)*0.0001 / 1000; %kW.s (kJouls)
%% Peak power

%peakPower = (max(consumedPower)- P_cabinet)/1000 ; %kWatts
peakPower = max(consumedPower)/1000 ; %kWatts
%%
figure(1)
plot(total_power);

%% Filter and downsample active power

close all
active_power = downsample(total_power,0.012*10000);
plot(active_power,'b')
hold on

n = 100;
Wn = 0.1;
b = fir1(n,Wn);


filt_power = conv(active_power,b)
plot(filt_power,'r')




%% Save the results
save('eOpt_R10.mat','total_energy','energy_noCabinet','consumedE','regenE','cycleTime','peakPower','filt_power');
