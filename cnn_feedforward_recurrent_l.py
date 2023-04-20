import torch
# print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import os
# print(os.getcwd())

import torch.nn as nn
import torch.nn.functional as F


class cnn_feedforward_recurrent_l(nn.Module):

    def __init__(self, t_steps=3):
        super(cnn_feedforward_recurrent_l, self).__init__()

        # training variables
        self.t_steps = t_steps

        # activation functions, pooling and dropout layers
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout()

        # conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.rlconv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)

        # conv2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.rlconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)

        # conv3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.rlconv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
 
        # fc 1
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        self.rlfc1 = nn.Linear(in_features=1024, out_features=1024)

        # decoder
        # self.decoder = nn.Linear(in_features=1024*self.t_steps, out_features=10)
        self.decoder = nn.Linear(in_features=1024, out_features=10)         # only saves the output from the last timestep to train

    def reset_parameters(self):
        
        # conv 1
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)     
    
        torch.nn.init.xavier_normal_(self.rlconv1.weight)
        torch.nn.init.zeros_(self.rlconv1.bias) 

        # conv 2
        torch.nn.init.xavier_normal_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)       
    
        torch.nn.init.xavier_normal_(self.rlconv2.weight)
        torch.nn.init.zeros_(self.rlconv2.bias) 

        # conv 3
        torch.nn.init.xavier_normal_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)         
    
        torch.nn.init.xavier_normal_(self.rlconv3.weight)
        torch.nn.init.zeros_(self.rlconv3.bias)

        # fc1
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)        
    
        torch.nn.init.xavier_normal_(self.rlfc1.weight)
        torch.nn.init.zeros_(self.rlfc1.bias)  

        # decoder
        torch.nn.init.xavier_normal_(self.decoder.weight)
        torch.nn.init.zeros_(self.decoder.bias)  


    def forward(self, input):

        """ Feedforward sweep. 
        
        Activations are saved in nestled dictionairies: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}},
        where the number indicates the layer
        
        """

        # determine batchsize
        # current_batchsiz = input[0].shape[0]

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

        # conv1
        x = self.conv1(input[0])
        actvs[0][0] = self.relu(x)
        x = self.pool(x)
        
        # conv2
        x = self.conv2(x)
        actvs[1][0] = self.relu(x)
        x = self.pool(x)

        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        actvs[2][0] = x
        
        # dropout
        x = self.dropout(x)

        # if current_batchsiz > 1:  
        #     x = x.view(x.size(0), -1)
        # else:
        #     x = torch.flatten(x)

        # flatten output
        x = x.view(x.size(0), -1)

        # fc1
        x = self.fc1(x)
        actvs[3][0] = x
        # actvs[4] = actvs[3][0]

        if self.t_steps > 0:
            for t in range(self.t_steps-1):

                # conv1
                x = self.conv1(input[t+1])   
                x_r = self.rlconv1(actvs[0][t])            
                actvs[0][t+1] = self.relu(torch.add(x, x_r))
                x = self.pool(actvs[0][t+1])
                
                # conv2
                x = self.conv2(x)
                x_r = self.rlconv2(actvs[1][t])            
                actvs[1][t+1] = self.relu(torch.add(x, x_r))
                x = self.pool(actvs[1][t+1])

                # conv3
                x = self.conv3(x)
                x_r = self.rlconv3(actvs[2][t])            
                actvs[2][t+1] = self.relu(torch.add(x, x_r))

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
                x_r = self.rlfc1(actvs[3][t])
                actvs[3][t+1] = torch.add(x, x_r)
                actvs[3][t+1] = x

        # only decode last timestep
        actvs[4] = self.decoder(actvs[3][t+1])
        # actvs[4] = torch.cat((actvs[4],self.decoder(actvs[3][t+1])))

        return actvs


# self.rconv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, groups=32) # depth-wise convolution

