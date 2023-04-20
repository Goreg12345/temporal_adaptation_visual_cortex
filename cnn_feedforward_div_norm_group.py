import torch
import os
import torch.nn as nn
from models.module_div_norm_group import module_div_norm_group
import torch.nn.functional as F


class cnn_feedforward_div_norm_group(nn.Module):

    def __init__(self, t_steps):
        super(cnn_feedforward_div_norm_group, self).__init__()

        # training variables
        self.t_steps = t_steps

        # activation functions, pooling and dropout layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()

        # placeholders
        init = torch.zeros(4)

        # conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.sconv1 = module_div_norm_group(32, 24, 24, init[0], True, init[0], True, init[0], True)
        
        # conv2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.sconv2 = module_div_norm_group(32, 8, 8, init[1], True, init[1], True, init[1], True)

        # conv3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.sconv3 = module_div_norm_group(32, 2, 2, init[2], True, init[2], True, init[2], True)
 
        # fc 1
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        # self.sfc1 = module_div_norm_group('none', 'none', 1024, init[3], True, init[3], True, init[3], True)

        # decoder
        # self.decoder = nn.Linear(in_features=1024*self.t_steps, out_features=10)
        self.decoder = nn.Linear(in_features=1024, out_features=10)         # only saves the output from the last timestep to train

    def reset_parameters(self, K_init, train_K, alpha_init, train_alpha, sigma_init, train_sigma):
        # conv 1
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)    

        self.sconv1.K = nn.Parameter(K_init[0], requires_grad=False)
        self.sconv1.alpha = nn.Parameter(alpha_init[0], requires_grad=train_alpha)
        self.sconv1.sigma = nn.Parameter(sigma_init[0], requires_grad=train_sigma)

        # conv 2
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)       

        self.sconv2.K = nn.Parameter(K_init[1], requires_grad=False)
        self.sconv2.alpha = nn.Parameter(alpha_init[1], requires_grad=train_alpha)
        self.sconv2.sigma = nn.Parameter(sigma_init[1], requires_grad=train_sigma)

        # conv 3
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)     

        self.sconv3.K = nn.Parameter(K_init[2], requires_grad=False)
        self.sconv3.alpha = nn.Parameter(alpha_init[2], requires_grad=train_alpha)
        self.sconv3.sigma = nn.Parameter(sigma_init[2], requires_grad=train_sigma)

        # fc1
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)        

        # self.sfc1.K = nn.Parameter(K_init[3], requires_grad=train_K)
        # self.sfc1.alpha = nn.Parameter(alpha_init[3], requires_grad=train_alpha)
        # self.sfc1.sigma = nn.Parameter(sigma_init[3], requires_grad=train_sigma)

        # decoder
        torch.nn.init.xavier_normal_(self.decoder.weight)
        torch.nn.init.zeros_(self.decoder.bias)

    def forward(self, input):

        """ Feedforward sweep. 
        
        Activations are saved in nestled dictionairies: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}},
        where the number indicates the layer
        
        """

        # initiate activations
        actvsc1 = {}
        actvsc2 = {}
        actvsc3 = {}
        actvsfc1 = {}

        actvs = {}
        actvs[0] = actvsc1
        actvs[1] = actvsc2
        actvs[2] = actvsc3
        actvs[3] = actvsfc1

        # initiate feedback signal
        Gc1 = {}
        Gc2 = {}
        Gc3 = {}
        # Gfc1 = {}
        
        G = {}
        G[0] = Gc1
        G[1] = Gc2
        G[2] = Gc3
        # G[3] = Gfc1

        # initiate multiplicative feedback signal
        Fc1 = {}
        Fc2 = {}
        Fc3 = {}
        # Ffc1 = {}
        
        F = {}
        F[0] = Fc1
        F[1] = Fc2
        F[2] = Fc3
        # F[3] = Ffc1

        # conv1
        x = self.conv1(input[0])
        actvs[0][0] = self.relu(x)
        G[0][0] = torch.zeros(x.shape)
        F[0][0] = torch.zeros(x.shape)
        x = self.pool(x)
        
        # conv2
        x = self.conv2(x)
        actvs[1][0] = self.relu(x)
        G[1][0] = torch.zeros(x.shape)
        F[1][0] = torch.zeros(x.shape)
        x = self.pool(x)

        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        actvs[2][0] = x
        G[2][0] = torch.zeros(x.shape)
        F[2][0] = torch.zeros(x.shape)
        
        # dropout
        x = self.dropout(x)

        # flatten output
        x = x.view(x.size(0), -1)

        # fc1
        x = self.fc1(x)
        actvs[3][0] = x
        # G[3][0] = torch.zeros(x.shape)
        # F[3][0] = torch.zeros(x.shape)

        if self.t_steps > 0:
            for t in range(self.t_steps-1):

                # conv1
                x = self.conv1(input[t+1])  
                x, F_updt, G_updt = self.sconv1(x, F[0][t], G[0][t])
                F[0][t+1] = F_updt
                G[0][t+1] = G_updt
                actvs[0][t+1] = x
                x = self.pool(actvs[0][t+1])
                
                # conv2
                x = self.conv2(x) 
                x, F_updt, G_updt = self.sconv2(x, F[1][t], G[1][t]) 
                F[1][t+1] = F_updt
                G[1][t+1] = G_updt
                actvs[1][t+1] = x
                x = self.pool(actvs[1][t+1])

                # conv3
                x = self.conv3(x)
                x, F_updt, G_updt = self.sconv3(x, F[2][t], G[2][t])
                F[2][t+1] = F_updt
                G[2][t+1] = G_updt
                actvs[2][t+1] = x

                # dropout
                x = self.dropout(actvs[2][t+1]) 

                # reshape
                x = x.view(x.size(0), -1)

                # fc1
                x = self.fc1(x)  
                # x, F_updt, G_updt = self.sfc1(x, F[3][t], G[3][t])
                # F[3][t+1] = F_updt
                # G[3][t+1] = G_updt
                actvs[3][t+1] = x

        actvs[4] = self.decoder(actvs[3][self.t_steps-1])

        return actvs
