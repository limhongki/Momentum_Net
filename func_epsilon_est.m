function epsilon = func_epsilon_est(IMout, IMout_old, Irecon, Irecon_old, ratio_aggr, num_sampl, L) 

rnx = [];
rny = [];
for iter_aggr = 1:ratio_aggr
    rnx = [rnx, randperm(L)];
    rny = [rny, randperm(L)];
end

rnx_idx = rnx(1:num_sampl);
rny_idx = rny(1:num_sampl);
   
epsilon_est = zeros(1,num_sampl);
for j = 1:num_sampl
    epsilon_est(j) = ...
        norm( col(IMout(:,:,rnx_idx(j))) - col(IMout_old(:,:,rny_idx(j))), 2 ) - ...
        norm( col(Irecon(:,:,rnx_idx(j))) - col(Irecon_old(:,:,rny_idx(j))), 2 );   %original assumption
%         norm( col(IMout(:,:,rnx_idx(j))) - col(IMout_old(:,:,rny_idx(j))), 2 ) / ...
%         norm( col(Irecon(:,:,rnx_idx(j))) - col(Irecon_old(:,:,rny_idx(j))), 2 ) - 1;   %new assumption
end
    
epsilon = max(max(epsilon_est), 0);  %smallest epsilon value is 0