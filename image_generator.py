import os, sys
from model import pggan, stylegan2

import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd


batch_size = 16

device = torch.device("cuda:5" if (torch.cuda.is_available()) else "cpu")
#/home/hyun_s/stylegan2/StyleGAN2_featuremap/stylegan/stylegan2_cat256.pth
model_path = 'stylegan/stylegan2_cat256.pth'
G, D = stylegan2(path=model_path, res=256)
G.to(device)


df = pd.read_csv('stylegan/cat_40000.csv')

z_s = []
for x in range(len(df)):
    tmp = (df['z'].iloc[x][2:-2]).split(' ')
    tmp = [x.split('\n')[0] for x in tmp if x != '']
    tensor = torch.Tensor(np.asarray(tmp,dtype=float)[np.newaxis,])
    z_s.append(tensor)

zs = torch.cat(z_s,axis=0)
v_score = df['vog_score'].to_numpy()



output_dir = '/home/hyun_s/stylegan2/StyleGAN2_featuremap/g_image'

print(zs.shape)