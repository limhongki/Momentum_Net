clc
clear

load Test_Momentum_resnet_rho_0.5_optim_l2_R_49_K_49_cn_21.3851_layer_100_epoch_300_lr_0.001_0.001_0.1.mat
etime_momentum = etime;
RMSE_momentum = RMSEtest;

load Test_Momentum_resnet_rho_0.5_optim_l2_R_49_K_49_cn_168.6415_layer_100_epoch_300_lr_0.001_0.001_0.1.mat
etime_momentum_cn168 = etime;
RMSE_momentum_cn168 = RMSEtest;

load Test_Momentum_scnn_resnet_R_9_rho_0.5_optim_l2_cn_168.6415_layer_100_epoch_300_lr_0.001_0.001_0.1.mat
etime_momentum_scnn_r3 = etime;
RMSE_momentum_scnn_r3 = RMSEtest;

load Test_Momentum_scnn_resnet_R_49_rho_0.5_optim_l2_cn_168.6415_layer_100_epoch_300_lr_0.001_0.001_0.1.mat
etime_momentum_scnn_r7 = etime;
RMSE_momentum_scnn_r7 = RMSEtest;

load Test_BCD_resnet_rho_1_layer_50_niter_3_optim_l2_R_49_K_49_cn_168.6415_layer_50_epoch_300_lr_0.001_0.001_0.1.mat
etime_bcd_3_iter = etime;
RMSE_bcd_3_iter = RMSEtest;

load Test_BCD_resnet_rho_1_layer_30_niter_10_optim_l2_R_49_K_49_cn_168.6415_layer_30_epoch_300_lr_0.001_0.001_0.1.mat
etime_bcd_10_iter = etime;
RMSE_bcd_10_iter = RMSEtest;

load Test_TNRD_resnet_rho_1_layer_100_niter_1_optim_l2_R_49_K_49_cn_168.6415_layer_100_epoch_300_lr_0.001_0.001_0.1.mat
etime_tnrd = etime;
RMSE_tnrd = RMSEtest;

load Test_ADMM_Net_resnet_rho_1_layer_30_niter_10_optim_l2_R_49_K_49_cn_168.6415_layer_30_epoch_200_lr_0.001_0.001_0.1.mat
etime_admm_10_iter = etime;
RMSE_admm_10_iter = RMSEtest;

load Test_ADMM_Net_resnet_rho_1_layer_50_niter_3_optim_l2_R_49_K_49_cn_168.6415_layer_50_epoch_200_lr_0.001_0.001_0.1.mat
etime_admm_3_iter = etime;
RMSE_admm_3_iter = RMSEtest;

load Test_PDS_Net_resnet_layer_100_optim_l2_R_49_K_49_cn_168.6415_layer_100_epoch_200_lr_0.001_0.001_0.1.mat
etime_pds = etime;
RMSE_pds = RMSEtest;


figure;
plot(etime_momentum_cn168, mean(RMSE_momentum_cn168,2), '-x')
hold on
plot(etime_momentum_scnn_r3, mean(RMSE_momentum_scnn_r3,2), '-d')
plot(etime_momentum_scnn_r7, mean(RMSE_momentum_scnn_r7,2), '-*')
plot(etime_bcd_10_iter, mean(RMSE_bcd_10_iter,2), '-v')
plot(etime_bcd_3_iter, mean(RMSE_bcd_3_iter,2), '-o')
plot(etime_tnrd, mean(RMSE_tnrd,2), '-^')
plot(etime_admm_10_iter, mean(RMSE_admm_10_iter,2), '->')
plot(etime_admm_3_iter, mean(RMSE_admm_3_iter,2), '-s')
plot(etime_pds, mean(RMSE_pds,2), '-p')

% legend('Momentum-Net (ResNet), rho=0.5', 'Momentum-Net-SCNN (ResNet)', 'BCD-Net (ResNet), fixed inner iter (10), rho=1', 'BCD-Net (ResNet), fixed inner iter (3), rho=1', 'TNRD (ResNet), rho=1')
legend('Momentum-Net , rho=0.5', 'Momentum-Net-SCNN (R=9)','Momentum-Net-SCNN (R=49)', 'BCD-Net (inner iter 10)', 'BCD-Net (inner iter 3)' , 'TNRD, rho=1', 'ADMM-Net, fixed inner iter (10)', 'ADMM-Net, fixed inner iter (3)', 'PDS-Net')

xlabel('Elapsed Time'); ylabel('RMSE'); set(gca,'fontsize',30)

etime_momentum(40)
% RMSE_momentum_mean = mean(RMSE_momentum,2);
% RMSE_momentum_mean(39:41)
etime_bcd_10_iter(21)
% RMSE_bcd_10_mean = mean(RMSE_bcd_10_iter,2);
% RMSE_bcd_10_mean(20:22)
etime_bcd_3_iter(36)
% RMSE_bcd_3_mean = mean(RMSE_bcd_3_iter,2);
% RMSE_bcd_3_mean(35:37)
etime_tnrd(78)
% RMSE_tnrd_mean = mean(RMSE_tnrd,2);
% RMSE_tnrd_mean(77:79)
(etime_momentum(40)-etime_tnrd(78))/etime_tnrd(78)*100
(etime_momentum(40)-etime_bcd_3_iter(36))/etime_bcd_3_iter(36)*100
