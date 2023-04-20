# %%

import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import os
from torch import optim
import torch.nn as nn
from torchsummary import summary
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import required script
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_exp_decay import cnn_feedforward_exp_decay
from models.cnn_feedforward_div_norm import cnn_feedforward_div_norm
from models.cnn_feedforward_div_norm_group import cnn_feedforward_div_norm_group
from models.cnn_feedforward_recurrent_l import cnn_feedforward_recurrent_l
from models.cnn_feedforward_recurrent_f import cnn_feedforward_recurrent_f
from utils.noiseMNIST_dataset import noiseMNIST_dataset
from utils.functions import *
import neptune.new as neptune
import json
import torch

# torch.manual_seed(0)

# start time
startTime = time.time()

###################################################################################### 
# CHANGE SETTINGS HERE

# data settings
contrast = 'lm_contrast'
mnist = False                # True if MNIST, False if fMNIST

# type of adaptation, OPTIONS: [exp. decay, div. norm., lat. recurrence and feedback]
# adapt = 'no_adaptation'
# adapt = 'exp_decay'
# adapt = 'div_norm'
# adapt = 'div_norm_group'
adapt = 'recurrence_l'
# adapt = 'recurrence_f'

# trial type
noise = 'same'
# noise = 'different'

###################################################################################### 
###################################################################################### 

# define root
dir = '/home/amber/OneDrive/code/git_nAdaptation_DNN/'

# set home directory
if mnist:
    dir_data = dir + 'datasets/DNN/MNIST/' + contrast
else:
    dir_data = dir + 'datasets/DNN/fMNIST/' + contrast

# get temporal info from json file
f = open(dir_data + '/' + 'sequenceInfo.json')
temporal_info = json.load(f)
print(temporal_info)

t_steps = temporal_info['total no. of time steps']
dur = temporal_info['stimulus durations']
start = temporal_info['stimulus start']

# set hypterparameters (pseudo-hardcoded)
numepchs = 2
batchsiz = 100
lr = 0.001
train_param = True

# track model training on neptune (CHANGE THESE VALUES)
run_init = False
random_init = 10

print(30*'--')
print('Dataset: ', dir_data)
print('Model: ', adapt)
print('Parameter trained: ', train_param)
print('Noise pattern: ', noise)
print(30*'--')

###################################################################################################################################

# training set
noise_imgs_train = torch.load(dir_data + '/' + noise + '_train_imgs')
noise_lbls_train = torch.load(dir_data + '/' + noise + '_train_lbls')
traindt = noiseMNIST_dataset(noise_imgs_train, noise_lbls_train)
print('Shape training set:'.ljust(50), noise_imgs_train.shape, ', ', noise_lbls_train.shape)

# load test set
noise_imgs_test = torch.load(dir_data + '/' + noise + '_test_imgs')
noise_lbls_test = torch.load(dir_data + '/' + noise + '_test_lbls')
testdt = noiseMNIST_dataset(noise_imgs_test, noise_lbls_test)
print('Shape test set:'.ljust(50), noise_imgs_test.shape, ', ', noise_lbls_test.shape)

# SAME - dataloader
ldrs = load_data(traindt, testdt, batch_size=batchsiz, shuffle=True, num_workers=1)
print('Shape LDRS training set:'.ljust(50), ldrs['train'])
print('Shape LDRS test set:'.ljust(50), ldrs['test'])

####################################################################################################################################

# layers feedforwar backbone
layers = ['conv1', 'conv2', 'conv3', 'fc1']

# run n random initializations
accuracies = torch.zeros(random_init)
i = 0
random_init_count = 0
while random_init_count != random_init:

    # track training
    interrupted = False

    # set counter
    print(30*'-')
    print('Random init: ', random_init_count+1)

    # initialize parameters
    if adapt == 'exp_decay': # values adapted from Vinken et al. (2019)
        alpha_init = torch.ones(len(layers))*0.96
        beta_init = torch.ones(len(layers))*0.7
        # alpha_init = torch.rand(len(layers))
        # beta_init = (-1 - 1) * torch.rand(len(layers)) + 1
        print('Alpha: ', alpha_init)
        print('Beta: ', beta_init)
    elif (adapt == 'div_norm'):
        K_init = torch.ones(len(layers))*0.6
        alpha_init = torch.ones(len(layers))*0.001
        sigma_init = torch.ones(len(layers))*1.3
    elif (adapt == 'div_norm_group'):
        K_init = torch.ones(len(layers))*0.6
        alpha_init = torch.ones(len(layers))*0.001
        sigma_init = torch.ones(len(layers))*1.3

    # initiate model
    if adapt == 'no_adaptation':
        model = cnn_feedforward(t_steps=t_steps)
    elif adapt == 'exp_decay':
        model = cnn_feedforward_exp_decay(t_steps=t_steps)
    elif adapt == 'div_norm':
        model = cnn_feedforward_div_norm(t_steps=t_steps)
    elif adapt == 'div_norm_group':
        model = cnn_feedforward_div_norm_group(t_steps=t_steps)
    elif adapt == 'recurrence_l':
        model = cnn_feedforward_recurrent_l(t_steps=t_steps)
    elif adapt == 'recurrence_f':
        model = cnn_feedforward_recurrent_f(t_steps=t_steps)

    # reinitialize network
    if (adapt == 'no_adaptation') | (adapt == 'recurrence_l') | (adapt == 'recurrence_f'):
        model.reset_parameters()
    elif adapt == 'exp_decay':
        model.reset_parameters(alpha_init, train_param, beta_init, train_param)
    elif (adapt == 'div_norm') | (adapt == 'div_norm_group'):
        model.reset_parameters(K_init, train_param, alpha_init, train_param, sigma_init, train_param)

    # loss function and optimizer
    lossfunct = nn.CrossEntropyLoss()   
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # train model
    print('Training in progress...')
    model.train()
    try:
        for epoch in range(numepchs): # images and labels
            for a, (imgs, lbls) in enumerate(ldrs['train']):

                # add inputs and labels
                ax = []
                for t in range(t_steps):
                    ax.append(var(imgs[:, t, :, :, :]))
                ay = var(lbls)   # labels

                # zero the parameter gradients
                optimizer.zero_grad()

                # compute step
                outp = model.forward(ax)
                losses = lossfunct(outp[len(outp)-1], ay)

                # backprop and optimization
                losses.backward() 
                optimizer.step()              
                
                # save losses and print progress
                if (a+1) % 50 == 0:
                    print ('Random init {} (run {}), Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                        .format(random_init_count+1, i+1, epoch+1, numepchs, a+1, len(ldrs['train']), losses.item()))
    
    except:
        print('Training interrupted...')
        interrupted = True

    # Test the model
    print('Validation in progress...')
    model.eval()
    accu = torch.zeros(len(ldrs['test']))
    for a, (imgs, lbls) in enumerate(ldrs['test']):

        # create input
        imgs_seq = []
        for t in range(t_steps):
            imgs_seq.append(imgs[:, t, : , :, :])
        
        # validate
        testoutp = model.forward(imgs_seq)
        predicy = torch.argmax(testoutp[len(testoutp)-1], dim=1)
        accu_temp = (predicy == lbls).sum().item() / float(lbls.size(0))
        
        # store accuracies
        accu[a] = accu_temp

    # print accuracy
    print('Accuracy:, ', torch.mean(accu))

    # save model if above change level and no interruptions occured
    if (torch.round(torch.mean(accu), decimals=1) > 0.1) & (interrupted == False):

        # save accuracies
        accuracies[random_init_count] = torch.mean(accu)

        # save weights for each network instance
        if mnist:
            torch.save(model.state_dict(), dir + 'weights/noise/MNIST/' + contrast + '/weights_feedforward_' + adapt + '_' + noise + '_' + str(random_init_count) + '.pth')
        else:
            torch.save(model.state_dict(), dir + 'weights/noise/fMNIST/' + contrast + '/weights_feedforward_' + adapt + '_' + noise + '_' + str(random_init_count) +  '.pth')

        # save accuracies
        if mnist:
            torch.save(accuracies, dir + 'accu/noise/MNIST/' + contrast + '/accu_feedforward_' + adapt + '_' + noise)
        else:
            torch.save(accuracies, dir + 'accu/noise/fMNIST/' + contrast + '/accu_feedforward_' + adapt + '_' + noise)

        # increment
        random_init_count+=1

    # increment count
    i+=1

print(30*'--')
print('Dataset: ', dir_data)
print('Model: ', adapt)
print('Parameter trained: ', train_param)
print('Noise pattern: ', noise)
print('Mean accuracy: ', torch.mean(accuracies), '(std ',  torch.std(accuracies), ')')
print(30*'--')

# print(model.sconv1.alpha)   
# print(model.sconv2.alpha)   
# print(model.sconv3.alpha)   
# print(model.sfc1.alpha)  

# print(model.sconv1.beta)   
# print(model.sconv2.beta)   
# print(model.sconv3.beta)   
# print(model.sfc1.beta)

# determine time it took to run script 
executionTime = (time.time() - startTime)
print('Execution time in seconds: ' + str(executionTime))
