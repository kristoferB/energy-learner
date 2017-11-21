function [P] = david_filter( energy, trace)
%Filter: synchs the energy log with motion
 
  %% Get the peak power before filtering
    
    % Get data
    d = energy.data;

    % Remove mean
    d = d - ones(length(d),1)*mean(d);

    % Sum over three phases I*V
    P = sum(d(:,2:4).*d(:,5:7),2);

    %% Filter
    
    % Filter window size
    N = 500;

    P = conv(P, ones(N,1),'valid')/N;

    % Filter trace data
    d = trace;
   % d(sum(isnan(d),2)>0,:) = [];

    x = d(:,2:7);

    dx = [zeros(1,6);diff(x)];

    P = downsample(P,10000*0.012);

    % Synch data
    f = xcorr(P,sum(dx.^2,2));
    [~,f] = max(f);

    P(1:f-length(P)) = [];
    P = P(1:length(x));
%% 
    % Remove cabinet
    P_0 = min(P);
    P = P - P_0;
    

%% plot
%     figure()
%     plot(P)
%     hold on
%     plot(sum(dx.^2,2)*1000,'r')
%     legend('Active power','Acceleration')



end

