def train(layer_idx, kern_size=3, kern_num=81, pad_size=1, lr_enc=1e-4, lr_dec=1e-4, lr_threshold=1e-4, alpha_init=1e-15, num_epoch=5,
          Lbatch=1, optim='l2', data_name='sphere', which_gpu=0):
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
    import scipy
    # from conv_sn_chen import conv_spectral_norm
    #from PIL import Image
    #import matplotlib.pyplot as plt
    #import time

    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)
    # os.chdir('/home/incfk8/Dropbox/Imaging/CT/Momentum-Net/')
    os.chdir('/n/escanaba/v/hongki/Documents/Momentum-Net/')
    print('cwd is:' + os.getcwd())

    class mydataset(dset):
        def __init__(self, folderpath_img, test):
            super(mydataset, self).__init__()
            if test == True:
                append = '_test'
            else:
                append = ''
            self.I_true_all = loadmat(folderpath_img)["Itrue" + append].transpose(2, 1, 0).astype(float)
            self.I_noisy_all = loadmat(folderpath_img)["Irecon" + append].transpose(2, 1, 0).astype(float)

        def __len__(self):
            return len(self.I_true_all)  # number of samples

        def __getitem__(self, index):

            I_true = np.expand_dims(self.I_true_all[index], axis=0)
            I_noisy = np.expand_dims(self.I_noisy_all[index], axis=0)

            return (I_true, I_noisy)

    train_dataset = mydataset('mypcodes/cache/Training_data_' + data_name + '_Layer' + str(layer_idx) + '.mat', test=False)
    train_loader = DataLoader(train_dataset, batch_size=Lbatch, shuffle=True)
    train_loader2 = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataset = mydataset('mypcodes/cache/Testing_data_' + data_name + '_Layer' + str(layer_idx) + '.mat', test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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

    class DnCNN(nn.Module):
        def __init__(self, channels, num_of_layers=17, lip=0.0, no_bn=True,
                     adaptive=False):
            super(DnCNN, self).__init__()
            kernel_size = 3
            padding = 1
            features = 64
            if lip > 0.0:
                sigmas = [pow(lip, 1.0/num_of_layers) for _ in range(num_of_layers)]
            else:
                sigmas = [0.0 for _ in range(num_of_layers)]
 
            if adaptive:
                # sigmas = [5.0, 2.0, 0.68, 0.46, 0.31]
                # sigmas = [5.0, 1.0, 0.584, 0.342]
                sigmas = [5.0, 2.0, 1.0, 0.681, 0.464, 0.316]
                assert len(sigmas) == num_of_layers, "Length of SN list uncompatible with num of layers."
 
            def conv_layer(cin, cout, sigma):
                conv = nn.Conv2d(in_channels=cin,
                                 out_channels=cout,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 bias=False)
                if sigma > 0.0:
                   # pdb.set_trace()
                    return conv_spectral_norm(conv, sigma=sigma)
                else:
                    return conv
 
            def bn_layer(n_features, sigma=1.0):
                bn = nn.BatchNorm2d(n_features)
                if sigma > 0.0:
                    return bn_spectral_norm(bn, sigma=sigma)
                else:
                    return bn
 
            layers = []
            layers.append(conv_layer(channels, features, sigmas[0]))
            layers.append(nn.ReLU(inplace=True))
            print("conv_1 with SN {}".format(sigmas[0]))
 
            for i in range(1, num_of_layers-1):
                layers.append(conv_layer(features, features, sigmas[i])) # conv layer
                print("conv_{} with SN {}".format(i+1, sigmas[i]))
                if not no_bn:
                    layers.append(bn_layer(features, 0.0)) # bn layer
                layers.append(nn.ReLU(inplace=True))
 
            layers.append(conv_layer(features, channels, sigmas[-1]))
            print("conv_{} with SN {}".format(num_of_layers, sigmas[-1]))
            self.dncnn = nn.Sequential(*layers)
 
        def forward(self, x):
            # out = -self.dncnn(x) + x
            out = self.dncnn(x)
            out = x - out # + x
            return out

    class deepEPINet(nn.Module):
        def __init__(self,num_feature=64):
            super(deepEPINet, self).__init__()
            self.conv1=nn.Conv2d(1, num_feature, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
            self.conv2=nn.Conv2d(num_feature, num_feature, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
            self.conv3=nn.Conv2d(num_feature, num_feature, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
            self.conv4=nn.Conv2d(num_feature, num_feature, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
            self.conv5=nn.Conv2d(num_feature, num_feature, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
            self.conv6=nn.Conv2d(num_feature, 1, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)

        def forward(self, x):
            out=self.conv1(x)
            out=F.relu(out)
            out=self.conv2(out)
            out=F.relu(out)
            out=self.conv3(out)
            out=F.relu(out)
            out=self.conv4(out)
            out=F.relu(out)
            out=self.conv5(out)
            out=F.relu(out)
            out=self.conv6(out)
        
            out = out+x #residue node
            return out

    class deepEPINet_dilated(nn.Module):
        def __init__(self,num_feature=64):
            super(deepEPINet_delated, self).__init__()
            self.conv1=nn.Conv2d(1, num_feature, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
            self.conv2=nn.Conv2d(num_feature, num_feature, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)
            self.conv3=nn.Conv2d(num_feature, num_feature, 3, stride=1, padding=2, dilation=2, groups=1, bias=True)
            self.conv4=nn.Conv2d(num_feature, num_feature, 3, stride=1, padding=4, dilation=4, groups=1, bias=True)
            self.conv5=nn.Conv2d(num_feature, num_feature, 3, stride=1, padding=8, dilation=8, groups=1, bias=True)
            self.conv6=nn.Conv2d(num_feature, 1, 3, stride=1, padding=1, dilation=1, groups=1, bias=True)

        def forward(self, x):
            out=self.conv1(x)
            out=F.relu(out)
            out=self.conv2(out)
            out=F.relu(out)
            out=self.conv3(out)
            out=F.relu(out)
            out=self.conv4(out)
            out=F.relu(out)
            out=self.conv5(out)
            out=F.relu(out)
            out=self.conv6(out)
        
            out = out+x #residue node
            return out

    torch.manual_seed(100)

    # net = autoEncoder()
    net = DnCNN(1,num_of_layers=6, lip=0.0, no_bn=True).cuda()
    # net = deepEPINet()
    # net = deepEPINet_dilated()

    net.type(dtype)
    net = nn.DataParallel(net)
    # overwrite_init(net, 'random')
    if layer_idx != 0:
        checkpoint_file = 'mypcodes/model/model_weights_'+data_name+ '_layer' + str(layer_idx) + '.pt'
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint['net'])
        

    if optim == 'l1':
        criterion = torch.nn.L1Loss()
    elif optim == 'l2':
        criterion = torch.nn.MSELoss()
    elif optim == 'l1.5':
        criterion = torch.nn.SmoothL1Loss()

    optimizer=torch.optim.Adam(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

    loss_history = []
    loss_history_epoch = []

    for epoch in range(num_epoch):
        for idx, data in enumerate(train_loader, 0):
            net.train()
            I_true_bat, I_noisy_bat = data
            I_true_bat = I_true_bat.to('cuda').float()
            I_noisy_bat = I_noisy_bat.to('cuda').float()
            IM_denoised = net(I_noisy_bat)
            loss = criterion(IM_denoised, I_true_bat)

            loss_history.append(loss.data.item()) # Pytorch v1.0

            # loss_history.append(loss.data)  # Pytorch v0.3.1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print('Current epoch: %d || Loss: %E' % (
                (epoch + 1), np.mean(loss_history) ))

        loss_history_epoch.append(np.mean(loss_history))

    net.eval()

    for idx, data in enumerate(train_loader2, 0):
        I_true_bat, I_noisy_bat = data
        I_true_bat = I_true_bat.to('cuda').float()
        I_noisy_bat = I_noisy_bat.to('cuda').float()
        IMout = net(I_noisy_bat)
        IMout = IMout.permute(3, 2, 1, 0)
        IMout = IMout.data.cpu().numpy()
        scipy.io.savemat(
            'mypcodes/cache/IMout_' + data_name + '_Layer' + str(layer_idx + 1) + '_image_' + str(idx) + '.mat',
            mdict={'IMout': IMout})

    for idx, data in enumerate(test_loader, 0):
        I_true_test_bat, I_noisy_test_bat = data
        I_true_test_bat = I_true_test_bat.to('cuda').float()
        I_noisy_test_bat = I_noisy_test_bat.to('cuda').float()
        IMout_test = net(I_noisy_test_bat)
        IMout_test = IMout_test.permute(3, 2, 1, 0)
        IMout_test = IMout_test.data.cpu().numpy()
        scipy.io.savemat(
            'mypcodes/cache/IMout_test_' + data_name + '_Layer' + str(layer_idx + 1) + '_image_' + str(idx) + '.mat',
            mdict={'IMout_test': IMout_test})

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

    torch.save(
        {
            'net': net.state_dict()
        },
        f='mypcodes/model/model_weights_'+data_name+ '_layer' + str(layer_idx + 1) + '.pt'
    )

    return {'loss_epoch': loss_history_epoch}

