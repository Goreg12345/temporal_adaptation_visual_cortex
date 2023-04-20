import torch
import os
import torch.nn as nn
from models.module_exp_decay import module_exp_decay
import torch.nn.functional as F

class cnn_feedforward_exp_decay(nn.Module):

    def __init__(self, t_steps):
        super(cnn_feedforward_exp_decay, self).__init__()

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
        self.sconv1 = module_exp_decay(32, 24, 24, init[0], True, init[0], True)
        
        # conv2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.sconv2 = module_exp_decay(32, 8, 8, init[1], True, init[1], True)

        # conv3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.sconv3 = module_exp_decay(32, 2, 2, init[2], True, init[2], True)
 
        # fc 1
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.sfc1 = module_exp_decay('none', 'none', 1024, init[3], True, init[3], True)

        # decoder
        # self.decoder = nn.Linear(in_features=1024*self.t_steps, out_features=10)
        self.decoder = nn.Linear(in_features=1024, out_features=10)         # only saves the output from the last timestep to train

    def reset_parameters(self, alpha_init, train_alpha, beta_init, train_beta):

        # conv 1
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)    

        self.sconv1.alpha = nn.Parameter(alpha_init[0], requires_grad=train_alpha)
        self.sconv1.beta = nn.Parameter(beta_init[0], requires_grad=train_beta)

        # conv 2
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)       

        self.sconv2.alpha = nn.Parameter(alpha_init[1], requires_grad=train_alpha)
        self.sconv2.beta = nn.Parameter(beta_init[1], requires_grad=train_beta)

        # conv 3
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)     

        self.sconv3.alpha = nn.Parameter(alpha_init[2], requires_grad=train_alpha)
        self.sconv3.beta = nn.Parameter(beta_init[2], requires_grad=train_beta)

        # fc1
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)        

        self.sfc1.alpha = nn.Parameter(alpha_init[3], requires_grad=train_alpha)
        self.sfc1.beta = nn.Parameter(beta_init[3], requires_grad=train_beta)

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

        # initiate suppression states
        sc1 = {}
        sc2 = {}
        sc3 = {}
        sfc1 = {}
        
        s = {}
        s[0] = sc1
        s[1] = sc2
        s[2] = sc3
        s[3] = sfc1

        # conv1
        x = self.conv1(input[0])
        actvs[0][0] = self.relu(x)
        s[0][0] = torch.zeros(x.shape)
        x = self.pool(x)
        
        # conv2
        x = self.conv2(x)
        actvs[1][0] = self.relu(x)
        s[1][0] = torch.zeros(x.shape)
        x = self.pool(x)

        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        actvs[2][0] = x
        s[2][0] = torch.zeros(x.shape)
        
        # dropout
        x = self.dropout(x)

        # # reshape
        # if current_batchsiz > 1:  
        #     x = x.view(x.size(0), -1)
        # else:
        #     x = torch.flatten(x)

        # flatten output
        x = x.view(x.size(0), -1)

        # fc1
        x = self.fc1(x)
        actvs[3][0] = x
        s[3][0] = torch.zeros(x.shape)
        # actvs[4] = actvs[3][0]

        if self.t_steps > 0:
            for t in range(self.t_steps-1):

                # conv1
                x = self.conv1(input[t+1])               
                s_updt, s_beta_updt = self.sconv1(actvs[0][t], s[0][t])
                s[0][t+1] = s_updt
                actvs[0][t+1] = self.relu(torch.subtract(x, s_beta_updt))
                x = self.pool(actvs[0][t+1])
                
                # conv2
                x = self.conv2(x)
                s_updt, s_beta_updt = self.sconv2(actvs[1][t], s[1][t])
                s[1][t+1] = s_updt
                actvs[1][t+1] = self.relu(torch.subtract(x, s_beta_updt))
                x = self.pool(actvs[1][t+1])

                # conv3
                x = self.conv3(x)
                s_updt, s_beta_updt = self.sconv3(actvs[2][t], s[2][t])
                s[2][t+1] = s_updt
                actvs[2][t+1] = self.relu(torch.subtract(x, s_beta_updt))

                # dropout
                x = self.dropout(actvs[2][t+1]) 

                # # reshape
                # if current_batchsiz > 1:  
                #     x = x.view(x.size(0), -1)
                # else:
                #     x = torch.flatten(x)

                x = x.view(x.size(0), -1)

                # fc1
                x = self.fc1(x)
                s_updt, s_beta_updt = self.sfc1(actvs[3][t], s[3][t])
                s[3][t+1] = s_updt
                actvs[3][t+1] = torch.subtract(x, s_beta_updt)
                # actvs[4] = torch.cat((actvs[4],actvs[3][t+1]))


        # only decode last timestep
        actvs[4] = self.decoder(actvs[3][self.t_steps-1])
        # actvs[4] = torch.cat((actvs[4],self.decoder(actvs[3][t+1])))

        return actvs
