%% MOMENTUM-NET
clear all; close all; clc;

%% Construct paired training data
% pcodes_init 
which_gpu = 1;
% gen_data
load('./data/training_data(ctrecon,phantom).mat');
load('./data/testing_data(ctrecon,phantom).mat');

data_name = 'phantom_momentum';

cache_path = 'mypcodes/cache/';
filename = strcat(cache_path,'Training_data_',data_name,'_Layer',num2str(0),'.mat');
save(filename,'Irecon','Itrue','-v7');

filename = strcat(cache_path,'Testing_data_',data_name,'_Layer',num2str(0),'.mat');
save(filename,'Irecon_test','Itrue_test','-v7');


gen_sys;

L = size(Itrue,3);
Ltest = size(Itrue_test,3);
Lbatch = 20;       %size of mini batch
num_epoch = 300;   %number of epochs

num = 100;         %number of layers
R = 7^2;           %size of filter
K = 7^2;            %number of filters
Rpad = floor((sqrt(R)-1)/2);    %padding size
delta = 1-eps;
rho = 0.5;
resnet = 1;

optim = 'l2';
lr_enc = 1e-3; 
lr_dec = 1e-3;
lr_threshold = 1e-1;

w_u_var = true;    %variation of w_u: "true" or "false"?
if w_u_var == true
    tau = 1;
else
    w_u = 1;
end

%reg param (lambda) selection based on condition numbers
ld_exp = 13e6;  %ld value obtained from CAOL experiments

%%%%Condition number based selection: full FOV
% Mx_exp = Mx_ffov_test(:,:,2);  %diagonal majorizer used in CAOL experiments
% cn = (max(Mx_exp(:)) + ld_exp) ./ (min(Mx_exp(:)) + ld_exp);  %desired condition number
% ldvec = func_ld_est_corr( Mx_ffov, cn, true(ig.dim), L );
% ldvec_test = func_ld_est_corr( Mx_ffov_test, cn, true(ig.dim), Ltest );
% clear Mx_exp Mx_ffov Mx_ffov_test

%%%%Condition number based selection: circular FOV
% Mx_exp = Mx_test(:,:,2);  %diagonal majorizer used in CAOL experiments
% cn = (max(Mx_exp(ig.mask)) + ld_exp) ./ (min(Mx_exp(ig.mask)) + ld_exp);  %desired condition number
% ldvec = func_ld_est_corr( Mx, cn, ig.mask, L );
% ldvec_test = func_ld_est_corr( Mx_test, cn, ig.mask, Ltest );
% clear Mx_exp Mx_ffov Mx_ffov_test

%%%%Upper bounded condition number based selection: better results than the methods above
Mx_exp = Mx_test(:,:,2);  %diagonal majorizer used in CAOL experiments
cn = (max(Mx_exp(ig.mask)) - min(Mx_exp(ig.mask))) ./ ld_exp + 1;     %desired condition number (?)
ldvec = func_ld_est( Mx, cn, ig.mask, L );
ldvec_test = func_ld_est( Mx_test, cn, ig.mask, Ltest );
clear Mx_exp Mx_ffov Mx_ffov_test

ratio_aggr = 1;     %ratio to aggregate data selection index
num_sampl = 100;    %number of taining samples need for kappa/epsilon estimation

%% Learn image mapping autoencoder and apply reconstruction modules
W = zeros(sqrt(R),sqrt(R),K,num,'single');       %encoding filters
D = zeros(sqrt(R),sqrt(R),K,num,'single');       %decoding filters
alpha = zeros(K,num,'single');                   %thresholding values
loss = zeros(num_epoch,num,'single');            %MSE during training
PSNR = zeros(num,L,'single');                    %PSNR for training               
PSNR_IMout = zeros(num,L,'single');              %PSNR for training               
PSNRtest = zeros(num,Ltest,'single');            %PSNR for testing
PSNRtest_IMout = zeros(num,Ltest,'single');            %PSNR for testing
RMSEtest = zeros(num,Ltest,'single'); 
maxval = max(Itrue(:)); 

%initial PSNR for training and testing
PSNR0 = reshape(cell2mat( cellfun( @(A,B)(my_psnr(A,B,maxval,ig)), ...
    num2cell(Irecon, [1 2]), num2cell(Itrue, [1 2]), 'UniformOutput', false') ),[1,L]);
PSNR0test = reshape(cell2mat( cellfun( @(A,B)(my_psnr(A,B,maxval,ig)), ...
    num2cell(Irecon_test, [1 2]), num2cell(Itrue_test, [1 2]), 'UniformOutput', false') ),[1,Ltest]);
RMSE0test = reshape(cell2mat( cellfun( @(A,B)(my_rmse(A,B,ig)), ...
    num2cell(Irecon_test, [1 2]), num2cell(Itrue_test, [1 2]), 'UniformOutput', false') ),[1,Ltest]);

%Learn autoencoder for "num" different recon layers
disp(['Training begins. (L=', num2str(L), ', Lbatch=', num2str(Lbatch), ')']);
Irecon_ls = Irecon; 
Irecon_test_ls = Irecon_test; 
kappa = []; 
epsilon = [];
Delta_list = [];
Delta_test_list = [];

for kp = 1:num
    
    disp(['Layer #', num2str(kp)]);
       
    %==== Train mapping neural networks =====
    %Note: odd-sized filters only!
    output = py.mypcodes.train_iy.train(int32(kp-1),...
        pyargs('kern_size',int32(sqrt(R)), 'kern_num',int32(K), 'pad_size',int32(Rpad),...
        'lr_enc',lr_enc, 'lr_dec',lr_dec, 'lr_threshold',lr_threshold,...
        'num_epoch',int32(num_epoch), 'Lbatch',int32(Lbatch),...
        'optim', optim, 'data_name', data_name, 'which_gpu', which_gpu));
    
    %Save trained autoencoders
    Wb = single(py.array.array('d',py.numpy.nditer(output{'Wb'},pyargs('order','F'))));
    Wb = reshape(Wb,[K,sqrt(R),sqrt(R)]);
    Wb = permute(Wb,[2 3 1]);

    Db = single(py.array.array('d',py.numpy.nditer(output{'Db'},pyargs('order','F'))));
    Db = reshape(Db,[K,sqrt(R),sqrt(R)]);
    Db = permute(Db,[2 3 1]);

    alphab = single(py.array.array('d',py.numpy.nditer(output{'alphab'},pyargs('order','F'))));
    alphab = permute(alphab,[2 1]);

    filename = strcat(cache_path,['Learned_D_W_alpha_data_',data_name,'_Layer',num2str(kp),'.mat']);
    save(filename,'Wb','Db','alphab');

    lossb = cell2mat(cell(output{'loss_epoch'}));

    W(:,:,:,kp) = Wb; 
    D(:,:,:,kp) = Db;
    alpha(:,kp) = alphab;
    loss(:,kp) = lossb;
    
    %Display & record performances
    fig1 = figure(1);
    semilogy(loss);
    title('Training loss'); ylabel('Training loss (log scale)'); xlabel('Epochs');
    saveas(fig1, './results/loss.png');

    %==== Apply trained neural networks =====
    %Note: odd-sized filters only!
    if kp > 1
        IMout_old = IMout;  %for epsilon estimation
    end
    if resnet == 1
        %ResNet: Autoencoder for residual
        IMout = cellfun( @(A) autoEncCNNrot( A, Wb, Db, alphab, K ) + A,  num2cell(Irecon, [1 2]), 'UniformOutput', false');
        IMout_test = cellfun( @(A) autoEncCNNrot( A, Wb, Db, alphab, K ) + A,  num2cell(Irecon_test, [1 2]), 'UniformOutput', false');
    else
        %Autoencoder
        IMout = cellfun( @(A) autoEncCNNrot( A, Wb, Db, alphab, K ),  num2cell(Irecon, [1 2]), 'UniformOutput', false');
        IMout_test = cellfun( @(A) autoEncCNNrot( A, Wb, Db, alphab, K ),  num2cell(Irecon_test, [1 2]), 'UniformOutput', false');
    end
    IMout = cell2mat(IMout);
    IMout_test = cell2mat(IMout_test);
    
    %==== Mapping module =====
    if kp > 1
        I0_old = I0;
        I0_test_old = I0_test;
    end
    I0 = func_mapping( IMout, Irecon, rho );
    I0_test = func_mapping( IMout_test, Irecon_test, rho );
    
    if kp == 1 
        Irecon_old = Irecon;
        Irecon_test_old = Irecon_test;
    else
        Irecon_old = Irecon_ls(:,:,:,kp-1);
        Irecon_test_old = Irecon_test_ls(:,:,:,kp-1);
    end

    %kappa estimation
    kappa = cat(1, kappa, func_kappa_est(IMout, Irecon, ratio_aggr, num_sampl, L)); 
    
    %epsilon estimation
    if kp > 1
        epsilon = cat(1, epsilon, func_epsilon_est(IMout, IMout_old, Irecon, Irecon_old, ratio_aggr, num_sampl, L)); 
    end
    
    %rho estimation
    if kp > 1
        Delta = func_Delta_est( Irecon, I0_old, I0, L );
        Delta_test = func_Delta_est( Irecon_test, I0_test_old, I0_test, Ltest );
        Delta_list = cat(1, Delta_list, Delta);
        Delta_test_list = cat(1, Delta_test_list, Delta_test);
    end
    
    %Update momentum coefficient    
    if w_u_var == true        
        tau_old = tau;
        tau = ( 1 + sqrt(1 + 4*tau^2) ) / 2;
        w_u = (tau_old - 1)/tau;
    end
    
    %==== Reconstruction module (extrapolation included) =====
    Irecon = func_bpg_m_diffld_v0_x0(A, Irecon, Irecon_old, w, sino_train, Mx, I0, ig.mask, ldvec, w_u, delta);
    Irecon_test = func_bpg_m_diffld_v0_x0(A, Irecon_test, Irecon_test_old, w_test, sino_test, Mx_test, I0_test, ig.mask, ldvec_test, w_u, delta);
    Irecon_ls = cat(4,Irecon_ls,Irecon);
    Irecon_test_ls = cat(4,Irecon_test_ls,Irecon_test);
    
    %Display & record performances
    PSNR(kp,:) = reshape(cell2mat( cellfun( @(A,B)(my_psnr(A,B,maxval,ig)), ...
        num2cell(Irecon, [1 2]), num2cell(Itrue, [1 2]), 'UniformOutput', false') ),[1,L]);
    
    PSNR_IMout(kp,:) = reshape(cell2mat( cellfun( @(A,B)(my_psnr(A,B,maxval,ig)), ...
        num2cell(IMout, [1 2]), num2cell(Itrue, [1 2]), 'UniformOutput', false') ),[1,L]);
    
    PSNRtest(kp,:) = reshape(cell2mat( cellfun( @(A,B)(my_psnr(A,B,maxval,ig)), ...
        num2cell(Irecon_test, [1 2]), num2cell(Itrue_test, [1 2]), 'UniformOutput', false') ),[1,Ltest]);
    
    PSNRtest_IMout(kp,:) = reshape(cell2mat( cellfun( @(A,B)(my_psnr(A,B,maxval,ig)), ...
        num2cell(IMout_test, [1 2]), num2cell(Itrue_test, [1 2]), 'UniformOutput', false') ),[1,Ltest]);
    
    RMSEtest(kp,:) = reshape(cell2mat( cellfun( @(A,B)(my_rmse(A,B,ig)), ...
        num2cell(Irecon_test*mm2HU, [1 2]), num2cell(Itrue_test*mm2HU, [1 2]), 'UniformOutput', false') ),[1,Ltest]);
        
    %Save the training and testing datasets to train the autoencoder in the next layer    
    filename = strcat(cache_path,['Training_data_',data_name,'_Layer',num2str(kp),'.mat']);
    save(filename,'Irecon','Itrue','-v7');
    
    filename = strcat(cache_path,['Testing_data_',data_name,'_Layer',num2str(kp),'.mat']);
    save(filename,'Irecon_test','Itrue_test','-v7');
     
    disp(['Layer #', num2str(kp),' PSNR:', num2str(PSNRtest(kp,2)),' RMSE:', num2str(RMSEtest(kp,2))]);
    
    %Record & dispaly performances
    fig2 = figure(2);
    plot(PSNRtest_IMout,'--');
    hold on
    set(gca,'ColorOrderIndex',1)
    plot(PSNRtest);
    hold on
    plot(linspace(1,kp,kp),mean(PSNRtest(1:kp,:),2),'k', linspace(1,kp,kp),mean(PSNRtest_IMout(1:kp,:),2),'--k')
    legend('Test PSNR: IMout (test#1)', 'Test PSNR: IMout (test#2)', ...
        'Test PSNR: Irecon (test#1)', 'Test PSNR: Irecon (test#2)',...
        'Mean Test PSNR: Irecon', 'Mean Test PSNR: IMout', 'location','best');
    title(sprintf('Peak mean PSNR is %.2f dB',max(mean(PSNRtest(1:kp,:),2))))
    hold off
    saveas(fig2, './results/PSNR_TEST.png')
    
    fig3 = figure(3);
    plot(linspace(1,kp,kp),mean(PSNR(1:kp,:),2),linspace(1,kp,kp),mean(PSNR_IMout(1:kp,:),2));
    legend('Mean Train PSNR: Irecon','Mean Train PSNR: IMout','location','best');
    saveas(fig3, './results/PSNR_TRAIN.png');
    
    fig4 = figure(4);
    plot(kappa); title('\kappa values');
    saveas(fig4, './results/kappa.png'); 
    
    fig5 = figure(5);
    plot(epsilon); title('\epsilon values');
    saveas(fig5, './results/epsilon.png');
    
    if kp > 1
        fig6 = figure(6);
        errorbar(2:kp, mean(Delta_list,2), std(Delta_list,0,2));
        title('\Delta values');
        saveas(fig6, './results/Delta.png');
    end
    
    figure(7);
    imshow([Irecon_test(:,:,1), Irecon_test(:,:,2); IMout_test(:,:,1), IMout_test(:,:,2)]*mm2HU, [800 1200]); drawnow;
    title(sprintf('Layer: %g, PSNR (test#1): %g, RMSE (test#1): %g, PSNR (test#2): %g, RMSE (test#2): %g', ...
        kp, PSNRtest(kp,1), RMSEtest(kp,1), PSNRtest(kp,2), RMSEtest(kp,2))); drawnow;
    
end


%% Save and display
if resnet == 1
    name = ['Momentum_v0,Delta(resnet,rho',num2str(rho), ')_optim_', optim, '_R_', num2str(R), '_K_',num2str(K), '_cn_', num2str(cn), '_layer_',num2str(num), '_epoch_',num2str(num_epoch), '_batch_',num2str(Lbatch), '_lr_',num2str(lr_dec),'_',num2str(lr_enc),'_',num2str(lr_threshold),'.mat'];
else
    name = ['Momentum_v0,Delta(rho',num2str(rho), ')_optim_', optim, '_R_', num2str(R), '_K_',num2str(K), '_cn_', num2str(cn), '_layer_',num2str(num), '_epoch_',num2str(num_epoch),'_batch_',num2str(Lbatch), '_lr_',num2str(lr_dec),'_',num2str(lr_enc),'_',num2str(lr_threshold),'.mat'];
end
clear P output A down error cache_path I0 Ddisp Wdisp var sqrtK Rpad xtrue_hi ye xtrue xfbp Mdiag Mx Irecon_old Irecon IMout zi wi sino_true sino sino_train w Mx_test Irecon_test_old Irecon_test IMout_test sino_test w_test ig_big Db Wb i kp Itrue Delta Delta_test
% save(['./results/',name], '-v7.3') 
clear Irecon_ls IMout_ls
save(name, '-v7.3') 

% Legend = cell(num,1);
% figure; hold on;
% for kp = 1:num
%     semilogy(1:num_epoch, loss(:,kp));
%     Legend{kp}=strcat('Layer #', num2str(kp));
% end
% ylabel('Training loss (log scale)');
% xlabel('Epochs');
% legend(Legend);

Wdisp = W(:,:,:,end);
Ddisp = D(:,:,:,end);
for k = 1:K
    Wdisp(:,:,k) = Wdisp(:,:,k) - min(min(Wdisp(:,:,k)));
    Ddisp(:,:,k) = Ddisp(:,:,k) - min(min(Ddisp(:,:,k)));
    if max(max(Wdisp(:,:,k))) > 0
        Wdisp(:,:,k) = Wdisp(:,:,k)/max(max(Wdisp(:,:,k)));
    end    
    if max(max(Ddisp(:,:,k))) > 0
        Ddisp(:,:,k) = Ddisp(:,:,k)/max(max(Ddisp(:,:,k)));
    end
end

sqrtK = ceil(sqrt(K));
figure;
subplot(121)
im(['notick'],'row',sqrtK,'col',sqrtK,permute(Wdisp,[2,1,3]),'cbar');
title('Encoding filter (final estimates)');
    
subplot(122)
im(['notick'],'row',sqrtK,'col',sqrtK,permute(Ddisp,[2,1,3]),'cbar');
title('Decoding filter (final estimates)');