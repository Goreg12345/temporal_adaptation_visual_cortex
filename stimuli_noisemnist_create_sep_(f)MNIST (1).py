import json
import torch
print('\n', 'GPU available: ', torch.cuda.is_available(), '\n')
import os
import numpy as np
from torch import optim
import torch.nn as nn
from torchsummary import summary
import time
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
# from PIL import Image
# import PIL
from torchvision.utils import save_image
# from PIL import Image as pimg
# import cv2

# import required script
from models.cnn_feedforward import cnn_feedforward
from models.cnn_feedforward_exp_decay import cnn_feedforward_exp_decay
from utils.functions import *

###################################################################################### 
# CHANGE SETTINGS HERE

# data settings
contrast = 'mh_contrast'
mnist = False                # True if MNIST, False if fMNIST

###################################################################################### 
###################################################################################### 

# set home directory
if mnist:
    dir = '/home/amber/OneDrive/code/git_nAdaptation_DNN/datasets/DNN/MNIST/' + contrast
else:
    dir = '/home/amber/OneDrive/code/git_nAdaptation_DNN/datasets/DNN/fMNIST/' + contrast

try:
    os.mkdir(dir)
except:
    print('Directory already exists')

# set contrast value
contrast_value = 0
if contrast == 'l_contrast':
    contrast_value = 0.2
elif contrast == 'lm_contrast':
    contrast_value = 0.4
elif contrast == 'm_contrast':
    contrast_value = 0.6
elif contrast == 'mh_contrast':
    contrast_value = 0.8
elif contrast == 'h_contrast':
    contrast_value = 1

# tsteps_10_v1
t_steps = 10
dur = [5, 3]
start = [1, 7]

# Data to be written
dictionary = { 
    "total no. of time steps": t_steps,
    "stimulus durations": dur,
    "stimulus start": start,
}

# Serializing json
json_object = json.dumps(dictionary, indent=4)

# Writing to sample.json
os.chdir(dir)
with open("sequenceInfo.json", "w") as outfile:
    outfile.write(json_object)

###################################################################################### 

def create_stimuli(data, noise, mnist):

    # encode timtesteps
    t_steps_label = torch.zeros(t_steps)
    t_steps_label[start[0]:start[0]+dur[0]] = 1
    if noise == 'same':
        t_steps_label[start[1]:start[1]+dur[1]] = 2
    elif noise == 'different':
        t_steps_label[start[1]:start[1]+dur[1]] = 3

    # download data
    if data == 'train':
        dt, _, data_n, _ = download_data('/home/amber/OneDrive/code/git_nAdaptation_DNN/datasets', mnist=mnist, download=True)
    elif data == 'test':
        _, dt, _, data_n = download_data('/home/amber/OneDrive/code/git_nAdaptation_DNN/datasets', mnist=mnist, download=True)
    print(data, ': ', data_n)

    # define input shape
    input_shape = dt[0][0].shape
    c = input_shape[0]
    w = input_shape[1]
    h = input_shape[2]
    print('Input shape: ', input_shape) 

    # grayscale image
    x = torch.ones(input_shape)
    isi = torch.multiply(x, 0.5)

    # dataset settings
    img_num = data_n

    noise_imgs = torch.empty([img_num, t_steps, c, w, h])
    noise_lbls = torch.empty(img_num, dtype=torch.long)

    for i in range(img_num): # create datasamples

        if i == 0:
            fig, axs = plt.subplots(1, t_steps)
            axs[0].set_title(noise + '_' + data + '_contrast_' + str(contrast))

        # create noise pattern
        adapter = torch.rand(input_shape)

        # iterate over timesteps
        for t in range(t_steps):

            # assign stimuli to current timestep
            if t_steps_label[t] == 0:                                   # grey-scale            
                noise_imgs[i, t, :, :, :] = isi
            elif t_steps_label[t] == 1:                                 # adaptation stage
                noise_imgs[i, t, :, :, :] = adapter
            elif (t_steps_label[t] == 2) or (t_steps_label[t] == 3):    # target stage
                
                # adjust contrast of target image
                img = dt[i][0]
                img = F.adjust_contrast(img, contrast_value)            # lower contrast (less/more uniform target image?) (0=solid grey image, 1=original image)
                noise_lbls[i] = dt[i][1]
                
                # create input
                if (t_steps_label[t] == 2):
                    noise_imgs[i, t, :, :, :] = adapter + img           # same  
                else:               
                    adapter2 = torch.rand(input_shape)                  # different
                    noise_imgs[i, t, :, :, :]  = adapter2 + img

            # create figure
            if i == 0:
                axs[t].imshow(noise_imgs[i, t, :, :, :].reshape(28, 28, 1), cmap='gray', vmin=0, vmax=1)

        # save figure
        if i == 0:
            plt.savefig(dir + '/' + noise + '_' + data)
            plt.close()

        # print progress
        if (i+1)%500 == 0:
            print('Created all timesteps for image: ', i+1)

    # save 
    torch.save(noise_imgs, dir + '/' + noise + '_' + data + '_imgs')
    torch.save(noise_lbls, dir + '/' + noise + '_' + data + '_lbls')

# create stimuli
create_stimuli(data='train', noise='same', mnist=mnist)
create_stimuli(data='test', noise='same', mnist=mnist)

create_stimuli(data='train', noise='different', mnist=mnist)
create_stimuli(data='test', noise='different', mnist=mnist)