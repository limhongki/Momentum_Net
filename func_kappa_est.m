function kappa = func_kappa_est(IMout, Irecon, ratio_aggr, num_sampl, L) 

rnx = [];
rny = [];
for iter_aggr = 1:ratio_aggr
    rnx = [rnx, randperm(L)];
    rny = [rny, randperm(L)];
end

rnx_idx = rnx(1:num_sampl);
rny_idx = rny(1:num_sampl);

kappa_ls = zeros(1,num_sampl);
for j = 1:num_sampl
    kappa_ls(j) = ...
        norm( col(IMout(:,:,rnx_idx(j))) - col(IMout(:,:,rny_idx(j))), 2 ) / ...
        norm( col(Irecon(:,:,rnx_idx(j))) - col(Irecon(:,:,rny_idx(j))), 2 ); 
end

kappa = max(kappa_ls);