def train(layer_idx, kern_size=3, kern_num=81, pad_size=1, lr_enc=1e-4, lr_dec=1e-4, lr_threshold=1e-4, alpha_init=1e-15, num_epoch=5,
          Lbatch=1, optim='l2', data_name='ge'):
    import os as os
    os.environ["MKL_NUM_THREADS"] = "4"
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parameter import Parameter
    import numpy as np
    from torch.autograd import Variable
    from torch.utils.data import Dataset as dset
    from torch.utils.data import DataLoader
    from scipy.io import loadmat
    #from PIL import Image
    #import matplotlib.pyplot as plt
    #import time

    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.chdir('/home/incfk8/Dropbox/Imaging/CT/Momentum-Net/')
    print('cwd is:' + os.getcwd())

    class mydataset(dset):
        def __init__(self, folderpath_img, test):
            super(mydataset, self).__init__()
            if test == True:
                append = '_test'
            else:
                append = ''
            self.I_true_all = loadmat(folderpath_img)["Itrue" + append].transpose(2, 0, 1).astype(float)
            self.I_noisy_all = loadmat(folderpath_img)["Irecon" + append].transpose(2, 0, 1).astype(float)

        def __len__(self):
            return len(self.I_true_all)  # number of samples

        def __getitem__(self, index):

            I_true = np.expand_dims(self.I_true_all[index], axis=0)
            I_noisy = np.expand_dims(self.I_noisy_all[index], axis=0)

            return (I_true, I_noisy)

    def overwrite_init(network, option='random'):
        ##option is a string, either random or dct
        if option == 'random':
            # network.encoder.weight = nn.init.orthogonal_(network.encoder.weight)
            # network.decoder.weight = nn.init.orthogonal_(network.decoder.weight)
            # network.decoder.weight = nn.Parameter(network.encoder.weight.transpose(1,0).transpose(3,4))
            pass
        elif option == 'previous_layer':
            W0 = loadmat('mypcodes/cache/Learned_D_W_alpha_data_' + data_name +'_Layer' + str(layer_idx) + '.mat')['Wb']
            D0 = loadmat('mypcodes/cache/Learned_D_W_alpha_data_' + data_name +'_Layer' + str(layer_idx) + '.mat')['Db']
            alpha0 = loadmat('mypcodes/cache/Learned_D_W_alpha_data_' + data_name + '_Layer' + str(layer_idx) + '.mat')['alphab'].squeeze()

            W0 = np.expand_dims(W0, axis=0)  # add singleton dimension
            D0 = np.expand_dims(D0, axis=0)
            alpha0 = alpha0[np.newaxis,:,np.newaxis,np.newaxis]
            
            W0 = W0.transpose(3, 0, 1, 2)  ##permute dimension to fit with cnn weight dimension format
            D0 = D0.transpose(0, 3, 1, 2)

            network.encoder.weight = nn.Parameter(torch.from_numpy(W0))
            network.decoder.weight = nn.Parameter(torch.from_numpy(D0))
            network.soft_thresholding.alpha = nn.Parameter(torch.from_numpy(alpha0))  ##check

    train_dataset = mydataset('mypcodes/cache/Training_data_' + data_name + '_Layer' + str(layer_idx) + '.mat', test=False)
    train_loader = DataLoader(train_dataset, batch_size=Lbatch, shuffle=True)
    dtype = torch.cuda.FloatTensor

    class mysoftshrink(nn.Module):
        def __init__(self, K, initial_threshold=-5):
            super(mysoftshrink, self).__init__()
            self.K = K
            self.initial_threshold = initial_threshold
            self.alpha = Parameter(initial_threshold * torch.ones(1, K, 1, 1))

        def forward(self, input):
            out = ( input.abs() > (torch.exp(self.alpha)) ).type(dtype) * (
                        input.abs() - (torch.exp(self.alpha)) ).type(dtype) * input.sign()
            return out

    class autoEncoder(nn.Module):
        def __init__(self):
            super(autoEncoder, self).__init__()
            self.encoder = nn.Conv2d(1, kern_num, kern_size, padding=pad_size, bias=False)
            self.soft_thresholding = mysoftshrink(kern_num, alpha_init)
            self.decoder = nn.Conv2d(kern_num, 1, kern_size, padding=pad_size, bias=False)

        def forward(self, x):
            u = self.decoder(self.soft_thresholding(self.encoder(x))) + x
            return u

    torch.manual_seed(100)

    net = autoEncoder()
    overwrite_init(net, 'random')
    if layer_idx != 0:
        overwrite_init(net, 'previous_layer')
    net.type(dtype)
    net = nn.DataParallel(net)

    if optim == 'l1':
        criterion = torch.nn.L1Loss()
    elif optim == 'l2':
        criterion = torch.nn.MSELoss()
    elif optim == 'l1.5':
        criterion = torch.nn.SmoothL1Loss()

    optimizer = torch.optim.Adam([
        {'params': net.module.encoder.parameters(), 'lr': lr_enc},
        {'params': net.module.decoder.parameters(), 'lr': lr_dec},
        {'params': net.module.soft_thresholding.parameters(), 'lr': lr_threshold}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    loss_history = []
    loss_history_epoch = []
    update_ratio_W_hitory = []
    update_ratio_alpha_hitory = []
    update_ratio_D_hitory = []
    update_ratio_W_hitory_epoch = []
    update_ratio_D_hitory_epoch = []
    update_ratio_alpha_hitory_epoch = []

    old_W = torch.cuda.FloatTensor(net.module.encoder.weight.shape)
    old_alpha = torch.cuda.FloatTensor(net.module.soft_thresholding.alpha.shape)
    old_D = torch.cuda.FloatTensor(net.module.decoder.weight.shape)

    for epoch in range(num_epoch):

        scheduler.step()
        for idx, data in enumerate(train_loader, 0):
            net.train()
            I_true_bat, I_noisy_bat = data
            I_true_bat_var, I_noisy_bat_var = Variable(I_true_bat.type(dtype)), Variable(I_noisy_bat.type(dtype))
            IM_denoised = net(I_noisy_bat_var)
            loss = criterion(IM_denoised, I_true_bat_var)

            loss_history.append(loss.data.item()) # Pytorch v1.0

            # loss_history.append(loss.data)  # Pytorch v0.3.1

            old_W.copy_(net.module.encoder.weight.data)
            old_alpha.copy_(net.module.soft_thresholding.alpha.data)
            old_D.copy_(net.module.decoder.weight.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            alpha_epoch = net.module.soft_thresholding.alpha.data.cpu().numpy()

            update_ratio_W = torch.norm(old_W - net.module.encoder.weight.data) / torch.norm(old_W)
            update_ratio_D = torch.norm(old_D - net.module.decoder.weight.data) / torch.norm(old_D)
            update_ratio_alpha = torch.norm(old_alpha - net.module.soft_thresholding.alpha.data) / torch.norm(old_alpha)

            update_ratio_W_hitory.append(update_ratio_W.item()) # Pytorch v1.0
            update_ratio_alpha_hitory.append(update_ratio_D.item())
            update_ratio_D_hitory.append(update_ratio_alpha.item())

            # update_ratio_W_hitory.append(update_ratio_W)  # Pytorch v0.3.1
            # update_ratio_alpha_hitory.append(update_ratio_D)
            # update_ratio_D_hitory.append(update_ratio_alpha)

        print('Current epoch: %d || Loss: %E || Update ratio W:%3E, alpha:%3E, D:%3E, alpha max: %g , alpha min: %g' % (
                (epoch + 1), np.mean(loss_history), np.mean(update_ratio_W_hitory), np.mean(update_ratio_alpha_hitory),
                np.mean(update_ratio_D_hitory),np.amax(alpha_epoch), np.amin(alpha_epoch)))

        update_ratio_W_hitory_epoch.append(np.mean(update_ratio_W_hitory))
        update_ratio_D_hitory_epoch.append(np.mean(update_ratio_D_hitory))
        update_ratio_alpha_hitory_epoch.append(np.mean(update_ratio_alpha_hitory))
        loss_history_epoch.append(np.mean(loss_history))


    # model_name = 'layer' + str(layer_idx) + '_autoencoder_optim_' + optim + '_R_' + str(kern_size) + '_K_' + str(kern_num) + '_epoch_' + str(num_epoch) + '_batch_' + str(Lbatch) + '_lr_' + str(
    #     lr_enc) + str(lr_dec) + str(lr_threshold)

    # x_axis = np.arange(0, num_epoch)
    #
    # fig1 = plt.figure(figsize=(40, 20))
    # plt.semilogy(x_axis, update_ratio_W_hitory_epoch)
    # plt.semilogy(x_axis, update_ratio_D_hitory_epoch)
    # plt.semilogy(x_axis, update_ratio_alpha_hitory_epoch)
    # plt.ylim([1e-5, 1e-1])
    # plt.legend(('W ratio', 'D ratio', 'alpha ratio'), fontsize='xx-large')
    # plt.title('Parameter Update Ratio')
    # fig1.savefig('mypcodes/result/update_ratio_history_' + model_name + '.png', dpi=fig1.dpi)
    #
    # fig2 = plt.figure(figsize=(40, 20))
    # plt.semilogy(x_axis, loss_history_epoch)
    # plt.ylim([1e-4, 1e-1])
    # plt.legend(('Training Loss'), fontsize='xx-large')
    # plt.title('Loss History')
    # fig2.savefig('mypcodes/result/loss_history_' + model_name + '.png', dpi=fig2.dpi)

    return {'Wb': net.module.encoder.weight.data.cpu().numpy(), 'Db': net.module.decoder.weight.data.cpu().numpy(), \
            'alphab': net.module.soft_thresholding.alpha.data.cpu().numpy(), 'loss_epoch': loss_history_epoch}

