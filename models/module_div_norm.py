import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

class module_div_norm(nn.Module):
    def __init__(self, height, width, channels, K_init, train_K, alpha_init, train_alpha, sigma_init, train_sigma):
        super().__init__()

        self.epsilon = 1.e-8

        self.K = nn.Parameter(K_init, requires_grad=train_K)
        self.alpha = nn.Parameter(alpha_init, requires_grad=train_alpha)
        self.sigma = nn.Parameter(sigma_init, requires_grad=train_sigma)

    def forward(self, x, F_previous, G_previous):
        """ x is the current input computed for the linear response, F is the multiplicative feedback
        and G is the feedback signal. """

        L = torch.relu(x)  # linear response
        if torch.max(G_previous) > self.K:                                                  # rescale if feedback signal exceeds maximal attainable response
            G_previous = G_previous/torch.max(G_previous)
            G_previous = G_previous*self.K
        F = torch.sqrt(torch.subtract(self.K, G_previous)+self.epsilon)/self.sigma          # multiplicative feedback
        R = torch.relu(torch.mul(L, F))                                                     # response
        
        # compute G by adding activity of unit
        G = torch.add(torch.mul(1 - self.alpha, G_previous), torch.mul(self.alpha, R))      # update feedback signal

        # # # clamp values between [0, K]
        # G_min = torch.min(G)
        # G_max = torch.max(G)
        # G = G/torch.max(G)
        # G = G*self.K
        # print('Min (norm): ', torch.min(G))
        # print('Max (norm): ', torch.max(G))
                                    
        # try:
        #     idx = 20
        #     # print('\nx: ', x[idx, 0, 0, 0])
        #     # print('L: ', L[idx, 0, 0, 0])   
        #     # print('G: ', G[idx, 0, 0, 0])
        #     # print('F: ', F[idx, 0, 0, 0])  
        #     # print('R: ', R[idx, 0, 0, 0])  
        #     # print(torch.isnan(L).any())  
        #     # print(torch.isnan(G).any())  
        #     # print(torch.isnan(F).any())  
        #     # print(torch.isnan(R).any())  
        #     if torch.isnan(L).any() == True:
        #         print('(L) NaN values detected')
        #     if torch.isnan(G).any() == True:
        #         print('(G_norm) NaN values detected')
        #     if torch.isnan(F).any() == True:
        #         print('(F) NaN values detected')
        #     if torch.isnan(R).any() == True:
        #         print('(R) NaN values detected')
        # except:
        #     print('FC')

        return R, F, G

# ####################  plot timecourse for example unit

# # create input
# batchsiz = 3
# height = 2
# width = 2
# channels = 3
# # t = torch.arange(0.001, 10, 0.001)
# t = torch.arange(0, 10, 1)
# t_steps = len(t)

# r = torch.zeros(t_steps, batchsiz, channels, height, width)
# G = torch.zeros(t_steps, batchsiz, channels, height, width)
# F = torch.zeros(t_steps, batchsiz, channels, height, width)

# x = torch.zeros(t_steps, batchsiz, channels, height, width)
# x[1:6] = 0.3
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
# module = module_div_norm_full = module_div_norm(height=height,
#                                                         width=width, 
#                                                         channels=channels,
#                                                         K_init=torch.Tensor([0.1]),
#                                                         train_K=True, 
#                                                         alpha_init=torch.Tensor([0.01]), 
#                                                         train_alpha=True, 
#                                                         sigma_init=torch.Tensor([0.5]), 
#                                                         train_sigma=True)

# # compute
# for tt in range(1, t_steps):
#     r[tt, :, :, :, :], F[tt, :, :, :, :], G[tt, :, :, :, :] = module.forward(x[tt, :, :, :, :], F[tt-1, :, :, :, :], G[tt-1, :, :, :, :])

# # plot
# axs[0].plot(t, x[:, 0, 0, 0, 0].detach().numpy(), color='grey', label='stimulus', lw=0.5)
# axs[1].plot(t, G[:, 0, 0, 0, 0].detach().numpy(), color='crimson', alpha=0.5, linestyle='--', label='suppression')
# axs[2].plot(t, r[:, 0, 0, 0, 0].detach().numpy(), color='dodgerblue', label='response')

# axs[0].legend(fontsize=8, frameon=False)
# axs[1].legend(fontsize=8, frameon=False)
# axs[2].legend(fontsize=8, frameon=False)
# plt.tight_layout()
# plt.savefig('visualizations/toy_Heeger1992')
# plt.show()