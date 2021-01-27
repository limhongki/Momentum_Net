function [etime,RMSEtest] = func_test_momentum(W,D,alpha,K,num,ld,rho)

load './data/testing_data(ctrecon,phantom).mat'

gen_sys;

%% Learn image mapping autoencoder and apply reconstruction modules
Ltest = size(Itrue_test,3);
RMSEtest = zeros(num,Ltest,'single');

%Learn autoencoder for "num" different recon layers
disp(['Testing begins']);

w_u_var = true;    %variation of w_u: "true" or "false"?
if w_u_var == true
    tau = 1;
else
    w_u = 1;
end
delta = 1-eps;
Irecon_test_ls = Irecon_test; 
tic
for kp = 1:num
    
    Wb = W(:,:,:,kp);
    Db = D(:,:,:,kp);
    alphab = alpha(:,kp);
    
    %Apply trained autoencoder (a non-zero mean vector for Gaussian prior)
    %Note: odd-sized filters only!
    IMout_test = cellfun( @(A) autoEncCNNrot( A, Wb, Db, alphab, K ) + A,  num2cell(Irecon_test, [1 2]), 'UniformOutput', false');
    IMout_test = cell2mat(IMout_test);
    
    if kp > 1
        I0_test_old = I0_test;
    end
    
    I0_test = func_mapping( IMout_test, Irecon_test, rho );
    
    if kp == 1 
        Irecon_test_old = Irecon_test;
    else
        Irecon_test_old = Irecon_test_ls(:,:,:,kp-1);
    end
    
    if w_u_var == true        
        tau_old = tau;
        tau = ( 1 + sqrt(1 + 4*tau^2) ) / 2;
        w_u = (tau_old - 1)/tau;
    end
    
    Irecon_test = func_bpg_m_diffld_v0_x0(A, Irecon_test, Irecon_test_old, w_test, sino_test, Mx_test, I0_test, ig.mask, ld, w_u, delta);
    Irecon_test_ls = cat(4,Irecon_test_ls,Irecon_test);
    
    %Record performances
    
    RMSEtest(kp,:) = reshape(cell2mat( cellfun( @(A,B)(my_rmse(A,B,ig)), ...
        num2cell(Irecon_test*mm2HU, [1 2]), num2cell(Itrue_test*mm2HU, [1 2]), 'UniformOutput', false') ),[1,Ltest]);
    
    %Save the training and testing datasets to train the autoencoder in the next layer
    
    disp(['Layer #', num2str(kp),' RMSE:', num2str(RMSEtest(kp,end))]);
    etime(kp) = toc;
end