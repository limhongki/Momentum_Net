def train(layer_idx, kern_size=3, kern_num=81, pad_size=1, lr_enc=1e-4, lr_dec=1e-4, lr_threshold=1e-4, num_epoch=5, Lbatch=10, optim='l1',data_name='abc',which_gpu=0):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.nn.parameter import Parameter
    import numpy as np
    from torch.autograd import Variable
    from torch.utils.data import Dataset as dset
    from torch.utils.data import DataLoader
    from scipy.io import loadmat
    from PIL import Image
    #import matplotlib.pyplot as plt
    #from logger import Logger
    import os as os

    #from vis_utils import visualize_grid
    torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)
    # os.chdir('/home/incfk8/Dropbox/Imaging/CT/Momentum-Net/')
    os.chdir('/n/escanaba/v/hongki/Documents/Momentum-Net/')
    print ('cwd is:'+ os.getcwd())        
    class mydataset(dset):
        def __init__(self,folderpath_img,test):
            super(mydataset,self).__init__() 
            if test==True:
                append='_test'
            else:
                append=''
            self.I_true_all=loadmat(folderpath_img)["Itrue"+append].transpose(2,0,1).astype(float) #import 512,512,60 data array from mat and permute diemnsion 
                                                               #to 60,512,512, self.data is numpy array with size(64,512,512)   
            self.I_noisy_all=loadmat(folderpath_img)["Irecon"+append].transpose(2,0,1).astype(float)
                
        def __len__(self):
            return len(self.I_true_all) #number of samples
        def __getitem__(self,index):

            I_true = np.expand_dims(self.I_true_all[index],axis=0)  ##each sample has shape 1,512,512
            I_noisy = np.expand_dims(self.I_noisy_all[index],axis=0) ##each sample has shape 1,512,512

            return (I_true,I_noisy)

    def overwrite_init(network,option='random'):
        ##option is a string, either random or dct
        if option=='random':
            pass   ## do nothing, the initialization is the pytorch default random initialization

        elif option == 'previous_layer':
            # W0 = loadmat('mypcodes/cache/Learned_D_W_alpha_Layer'+str(layer_idx)+'.mat')['Wb']
            # D0 = loadmat('mypcodes/cache/Learned_D_W_alpha_Layer'+str(layer_idx)+'.mat')['Db']
            # alpha0 = loadmat('mypcodes/cache/Learned_D_W_alpha_Layer'+str(layer_idx)+'.mat')['alphab'].squeeze()
            
            W0 = loadmat('mypcodes/cache/Learned_D_W_alpha_data_' + data_name +'_Layer' + str(layer_idx) + '.mat')['Wb']
            D0 = loadmat('mypcodes/cache/Learned_D_W_alpha_data_' + data_name +'_Layer' + str(layer_idx) + '.mat')['Db']
            alpha0 = loadmat('mypcodes/cache/Learned_D_W_alpha_data_' + data_name + '_Layer' + str(layer_idx) + '.mat')['alphab'].squeeze()

            W0 = np.expand_dims(W0,axis=0)   #add singleton dimension
            D0 = np.expand_dims(D0,axis=0)
            alpha0 = alpha0[np.newaxis,:,np.newaxis,np.newaxis]
            
            W0=W0.transpose(3,0,1,2)   ##permute dimension to fit with cnn weight dimension format
            D0=D0.transpose(0,3,1,2)
            
            network.encoder.weight = nn.Parameter(torch.from_numpy(W0))
            network.decoder.weight = nn.Parameter(torch.from_numpy(D0))
            network.NL.alpha = nn.Parameter(torch.from_numpy(alpha0)) ##check
            
            

    # train_dataset = mydataset('mypcodes/cache/Training_data_Layer'+str(layer_idx)+'.mat',test=False) #trainingdata from phantom([4:29,31:64]) in matlab
    # train_loader = DataLoader(train_dataset, batch_size=Lbatch, shuffle=True)
    train_dataset = mydataset('mypcodes/cache/Training_data_' + data_name + '_Layer' + str(layer_idx) + '.mat', test=False)
    train_loader = DataLoader(train_dataset, batch_size=Lbatch, shuffle=True)
    
    test_dataset = mydataset('mypcodes/cache/Testing_data_' + data_name + '_Layer' + str(layer_idx) + '.mat', test=True)##testdata  phantom([1:3,30]) in matlab
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    class mysoftshrink(nn.Module):
        def __init__(self, K, initial_threshold=200):
            super(mysoftshrink, self).__init__()
            self.K = K
            self.initial_threshold = initial_threshold
            self.alpha = Parameter(initial_threshold*torch.ones(1,K,1,1)) ##to be broadcasted as N,K,H,W in the forward pass


        def forward(self, input):
            return (input.abs() > torch.exp(self.alpha)).type(dtype) * (input.abs() - torch.exp(self.alpha)) * input.sign()


    class autoEncoder(nn.Module):
        def __init__(self):
            super(autoEncoder, self).__init__()
            self.encoder = nn.Conv2d(1, kern_num, kern_size, padding=pad_size, bias=False) #note the kernal size is even,
                                                                                                   #so the output will not preserve the inoput size, 
            self.NL = mysoftshrink(kern_num, -5) #in this case, output is N,C,513X513

            self.decoder = nn.Conv2d(kern_num, 1, kern_size, padding=pad_size, bias=False)


        def forward(self,x):
            
            z = self.encoder(x)
            z_afterNL = self.NL(z)
            out = self.decoder(z_afterNL)
            out = out + x
            return out

    #dtype = torch.FloatTensor ##run on cpu
    dtype = torch.cuda.FloatTensor 
    torch.manual_seed(100) ##set the random seed so when construct the 

    net = autoEncoder()
    overwrite_init(net,'random') ##initialize filter to random/dct based on the argument
    if layer_idx != 0:
        overwrite_init(net,'previous_layer') ##load the learned network parameters from last layer as initialzation  
    net.type(dtype)
    net = nn.DataParallel(net)

    if optim == 'l1':
        criterion = torch.nn.L1Loss()
    elif optim == 'l2':
        criterion = torch.nn.MSELoss()
    elif optim == 'l1.5':
        criterion = torch.nn.SmoothL1Loss()

    criterion2 = torch.nn.MSELoss()
    #optimizer = torch.optim.Adam(net.module.parameters(), lr=1e-3)
    optimizer=torch.optim.Adam([
               {'params': net.module.encoder.parameters(),'lr':lr_enc},
               {'params': net.module.decoder.parameters(),'lr':lr_dec},
               {'params': net.module.NL.parameters(),'lr':lr_threshold}
              ])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    loss_history=[]
    loss_history_epoch=[]
    test_loss_history=[]
    update_ratio_W_hitory=[]
    update_ratio_alpha_hitory=[]
    update_ratio_D_hitory=[]

    old_W=torch.cuda.FloatTensor(net.module.encoder.weight.shape)
    old_alpha=torch.cuda.FloatTensor(net.module.NL.alpha.shape)
    old_D=torch.cuda.FloatTensor(net.module.decoder.weight.shape)

    for epoch in range(num_epoch):

        scheduler.step()
        for idx,data in enumerate(train_loader,0):
            net.train()
            I_true_bat,I_noisy_bat = data
            I_true_bat_var,I_noisy_bat_var = Variable(I_true_bat.type(dtype)),Variable(I_noisy_bat.type(dtype))
            IM_denoised = net(I_noisy_bat_var)
            loss = criterion(IM_denoised,I_true_bat_var) #the criterion calculate average MSE for each pixel, to get average MSE summed over one batch, multiply by 512*512
            loss_history.append(loss.data.item())


            old_W.copy_(net.module.encoder.weight.data)
            old_alpha.copy_(net.module.NL.alpha.data)
            old_D.copy_(net.module.decoder.weight.data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_ratio_W_hitory.append(
                torch.norm(old_W - net.module.encoder.weight.data) / torch.norm(old_W))
            update_ratio_alpha_hitory.append(
                torch.norm(old_alpha - net.module.NL.alpha.data) / torch.norm(old_alpha))
            update_ratio_D_hitory.append(
                torch.norm(old_D - net.module.decoder.weight.data) / torch.norm(old_D))

        print('Current epoch number: %d||| Loss: %E ||| Update ratio W:%3E, alpha:%3E, D:%3E' % ((epoch + 1), loss, update_ratio_W_hitory[-1], update_ratio_alpha_hitory[-1], update_ratio_D_hitory[-1]))
        loss_history_epoch.append(loss.data.item())

        if np.mod(epoch,10)==0:
            net.eval()

            I_true_test_bat,I_noisy_test_bat=next(iter(test_loader)) #load all 4 test samples
            I_true_test_bat_var,I_noisy_test_bat_var = Variable(I_true_test_bat.type(dtype)),Variable(I_noisy_test_bat.type(dtype))
            IM_denoised_test = net(I_noisy_test_bat_var)
            test_loss = criterion2(IM_denoised_test,I_true_test_bat_var)
            test_loss_history.append(test_loss.data)
            print("Validation Loss:%E "%test_loss.data)

    model_name = 'optim_' + optim + '_R_' + str(kern_size) + '_K_' + str(kern_num) + '_lr_enc_' + str(lr_enc) \
                 + '_lr_dec_' + str(lr_dec)+ '_lr_threshold_' + str(lr_threshold) + '_batch_' + str(Lbatch) + '_epoch_'\
                 + str(num_epoch)


    # a = np.arange(0, num_epoch/10)
    # b = np.arange(0, num_epoch, 1 / (train_dataset.I_true_all.shape[0] / Lbatch))
    # 
    # fig1 = plt.figure(figsize=(40, 20))
    # plt.semilogy(b, update_ratio_W_hitory)
    # plt.semilogy(b, update_ratio_D_hitory)
    # plt.semilogy(b, update_ratio_alpha_hitory)
    # plt.legend(('W ratio', 'D ratio', 'alpha ratio'), fontsize='xx-large')
    # plt.title('Parameter Update Ratio')
    # fig1.savefig('mypcodes/result/update_ratio_history_' + model_name + '.png', dpi=fig1.dpi)
    # 
    # fig2 = plt.figure(figsize=(40, 20))
    # plt.semilogy(a*10+10, test_loss_history)
    # plt.semilogy(b, loss_history)
    # plt.legend(('Testing Loss', 'Training Loss'), fontsize='xx-large')
    # plt.title('Loss History')
    # fig2.savefig('mypcodes/result/loss_history_' + model_name + '.png', dpi=fig2.dpi)

    return {'Wb':net.module.encoder.weight.data.cpu().numpy(), 'Db':net.module.decoder.weight.data.cpu().numpy(), \
            'alphab':net.module.NL.alpha.data.cpu().numpy(), 'loss_epoch':loss_history_epoch} ##note the loss_epoch is the loss at the end of each epoch

