import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

class module_div_norm_group(nn.Module):
    def __init__(self, channels, height, width, K_init, train_K, alpha_init, train_alpha, sigma_init, train_sigma):
        super().__init__()

        self.epsilon = 1.e-8

        self.height = height
        self.width = width
        self.channels = channels

        self.K = nn.Parameter(K_init, requires_grad=train_K)
        self.alpha = nn.Parameter(alpha_init, requires_grad=train_alpha)
        self.sigma = nn.Parameter(sigma_init, requires_grad=train_sigma)

    def forward(self, x, F_previous, G_previous):
        """ x is the current input computed for the linear response, F is the multiplicative feedback
        and G is the feedback signal. """

        L = torch.relu(x)                                                                   # linear response
        if torch.max(G_previous) > self.K: 
            # print('Max G: ', torch.max(G_previous))                                                 # rescale if feedback signal exceeds maximal attainable response
            G_previous = G_previous/torch.max(G_previous) # range [0,1]
            G_previous = G_previous*self.K                # range [0,K]
        F = torch.subtract(self.K, G_previous)
        F = torch.sqrt(F+self.epsilon)
        F = F/self.sigma                                                                    # multiplicative feedback
        R = torch.relu(torch.mul(L, F))                                                     # response
                                                         # response
        # compute G by adding activity of surrounding units
        R_pad = torch.nn.functional.pad(R, (1, 1, 1, 1))
        R_sum = R_pad[:, :, 1+1:self.height+1+1, 1:self.width+1] + \
                    R_pad[:, :, 1-1:self.height+1-1, 1:self.width+1] + \
                        R_pad[:, :, 1:self.height+1, 1+1:self.width+1+1] + \
                            R_pad[:, :, 1:self.height+1, 1-1:self.width+1-1] + \
                                R_pad[:, :, 1+1:self.height+1+1, 1+1:self.width+1+1] + \
                                    R_pad[:, :, 1-1:self.height+1-1, 1-1:self.width+1-1] + \
                                        R_pad[:, :, 1+1:self.height+1+1, 1-1:self.width+1-1] + \
                                            R_pad[:, :, 1-1:self.height+1-1, 1+1:self.width+1+1] + \
                                                R_pad[:, :, 1:self.height+1, 1:self.width+1]
        G = torch.add(torch.mul(1 - self.alpha, G_previous), torch.mul(self.alpha, R_sum))      # update feedback signal

        # try:
        #     # idx = 20
        #     # print('\nx: ', x[idx, 0, 0, 0])
        #     # print('L: ', L[idx, 0, 0, 0])   
        #     # print('G: ', G[idx, 0, 0, 0])
        #     # print('F: ', F[idx, 0, 0, 0])  
        #     # print('R: ', R[idx, 0, 0, 0])  
        #     # print(torch.isnan(L).any())  
        #     # print(torch.isnan(G).any())  
        #     # print(torch.isnan(F).any())  
        #     # print(torch.isnan(R).any())  
        #     # if torch.isnan(L).any() == True:
        #     #     print('(L) NaN values detected')
        #     # if torch.isnan(G).any() == True:
        #     #     print('(G_norm) NaN values detected')
        #     # # if torch.isnan(G_previous_max).any() == True:
        #     # #     print('(G_previous max) NaN values detected')
        #     # if torch.isnan(F).any() == True:
        #     #     print('(F) NaN values detected')
        #     # if torch.isnan(R).any() == True:
        #     #     print('(R) NaN values detected')
        # except:
        #     print('FC')

        return R, F, G

# ####################  plot timecourse for example unit

# # create input
# batchsiz = 1
# channels = 1
# height = 3
# width = 3

# # t = torch.arange(0.001, 10, 0.001)
# t = torch.arange(0, 10, 1)
# t_steps = len(t)

# r = torch.zeros(t_steps, batchsiz, channels, height, width)
# G = torch.zeros(t_steps, batchsiz, channels, height, width)
# F = torch.zeros(t_steps, batchsiz, channels, height, width)

# x = torch.zeros(t_steps, batchsiz, channels, height, width)
# x[0:6] = 2
# # x[201:1100, :, :, :] = 1
# # x[1801:1900, :, :, :] = 1

# # x[3501:4000, :, :, :] = 1
# # x[4301:4800, :, :, :] = 1

# # x[6501:7000, :, :, :] = 0.05
# # x[7501:8000, :, :, :] = 0.2
# # x[8501:9000, :, :, :] = 0.5

# # initiate figure
# fig, axs = plt.subplots(3, 1, figsize=(8, 4))
# axs[0].set_title('Heeger et al. (1992, 1993)')

# # initiate network
# module = module_div_norm_full = module_div_norm_group(height=height,
#                                                         width=width, 
#                                                         channels=channels,
#                                                         K_init=torch.Tensor([1]),
#                                                         train_K=True, 
#                                                         alpha_init=torch.Tensor([0.001]), 
#                                                         train_alpha=True, 
#                                                         sigma_init=torch.Tensor([0.8]), 
#                                                         train_sigma=True)

# # compute
# # for tt in range(1, t_steps):
# for tt in range(1):
#     r[tt, :, :, :, :], F[tt, :, :, :, :], G[tt, :, :, :, :] = module.forward(x[tt, :, :, :, :], F[tt-1, :, :, :, :], G[tt-1, :, :, :, :])

# # plot
# axs[0].plot(t, x[:, 0, 0, 0, 0].detach().numpy(), color='grey', label='stimulus', lw=0.5)
# axs[1].plot(t, G[:, 0, 0, 0, 0].detach().numpy(), color='crimson', alpha=0.5, linestyle='--', label='suppression')
# axs[2].plot(t, r[:, 0, 0, 0, 0].detach().numpy(), color='dodgerblue', label='response')

# axs[0].legend(fontsize=8, frameon=False)
# axs[1].legend(fontsize=8, frameon=False)
# axs[2].legend(fontsize=8, frameon=False)
# plt.tight_layout()
# plt.savefig('visualizations/toy_Heeger1992_group')
# # plt.show()



# R = torch.relu(torch.mul(L, F))                                                     # response

# R = torch.zeros(1, batchsiz, channels, height, width)
# print(R.shape)
# R[0, 0, 0, 0, 0] = 1
# R[0, 0, 0, 0, 1] = 2
# R[0, 0, 0, 0, 2] = 3
# R[0, 0, 0, 1, 0] = 4
# R[0, 0, 0, 1, 1] = 5
# R[0, 0, 0, 1, 2] = 6
# R[0, 0, 0, 2, 0] = 7
# R[0, 0, 0, 2, 1] = 8
# R[0, 0, 0, 2, 2] = 9
# print(R)

# # # print(torch.isnan(F).any())
# # # sum response of neighbouring units
# R = R[0, :, :, :, :]
# R_pad = torch.nn.functional.pad(R, (1, 1, 1, 1))
# print(R_pad)
# if height != 'none':
#     R_sum = R_pad[:, :, 1+1:height+1+1, 1:width+1] + \
#                 R_pad[:, :, 1-1:height+1-1, 1:width+1] + \
#                     R_pad[:, :, 1:height+1, 1+1:width+1+1] + \
#                         R_pad[:, :, 1:height+1, 1-1:width+1-1] + \
#                             R_pad[:, :, 1+1:height+1+1, 1+1:width+1+1] + \
#                                 R_pad[:, :, 1-1:height+1-1, 1-1:width+1-1] + \
#                                     R_pad[:, :, 1+1:height+1+1, 1-1:width+1-1] + \
#                                         R_pad[:, :, 1-1:height+1-1, 1+1:width+1+1] + \
#                                             R_pad[:, :, 1:height+1, 1:width+1]
# print(R_sum)