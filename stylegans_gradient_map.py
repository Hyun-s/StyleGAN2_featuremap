import os, sys

from model import pggan, stylegan2
from tqdm import tqdm

import pandas as pd
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

print('load model...')
model_path = 'stylegan/stylegan2_cat256.pth'
# model_path = 'stylegan/stylegan2_church256.pth'
G, D = stylegan2(path=model_path, res=256)
G.to(device)
D.to(device)
print('models completes')


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()
    return loss

z = torch.randn(4, 512, device=device)


def feature_map_gradient(G, z, wandb=False):
    G.eval()
    D.eval()

    var = {}
    grads = {}
    dic = G.mapping(z)
    const = dic['z']
    w = dic['w']
    wp = G.truncation(w)
    x = G.synthesis.early_layer(wp[:, 0])
    
    for layer_idx in range(G.synthesis.num_layers - 1):
        x, style = G.synthesis.__getattr__(f'layer{layer_idx}')\
                                        (x, wp[:, layer_idx])
        if layer_idx % 2 == 0:
            temp, style = G.synthesis.__getattr__(f'output{layer_idx // 2}')\
                                                (x, wp[:, layer_idx + 1])
            if layer_idx == 0:
                image = temp
            else:
                image = temp + G.synthesis.upsample(image)
        name = f'layer{layer_idx}'
        tmp = x
        tmp.retain_grad()
        grads[name] = tmp

        var[name] = torch.var(x,axis=1)
    out = G.synthesis.final_activate(image)# * 127.5 + 127.5
    fake_pred = D(out)
    g_loss = g_nonsaturating_loss(fake_pred)
    g_loss.backward()
    for k in grads.keys():
        grads[k] = grads[k].grad
    return grads, var, out

z = torch.randn(4, 512, device=device)
g, v, i = feature_map_gradient(G,z)


idx_=1
img = i.permute(0,2,3,1).detach().cpu().numpy()
img = (img+1)*127.5
img = img.astype(int)
for x in range(4):
    map=[]
    for idx in range(3,13):
        tmp = v['layer{}'.format(idx)][x][np.newaxis,np.newaxis,...]
        # tmp = torch.Tensor(tmp)
        # print(tmp.shape)
        tmp = torch.nn.UpsamplingBilinear2d(size=256)(tmp)
        map.append(tmp.detach().cpu().numpy())
    v_img = np.sum(map[0][0], axis=0)
    map=[]
    for idx in range(3,13):
        tmp = g['layer{}'.format(idx)][x]
        tmp = torch.var(tmp,axis=0)[np.newaxis,np.newaxis,...]
        tmp = torch.nn.UpsamplingBilinear2d(size=256)(tmp)
        map.append(tmp.detach().cpu().numpy())
    g_img = np.sum(map[0][0], axis=0)
    g_img = (g_img - g_img.min()) / (g_img.max() - g_img.min()) 
    plot_img = img[x]
    row = 4
    col = 3
    fig = plt.figure(figsize=(10,60))
    # for x ,(g, d) in enumerate(zip(g_img,d_img)):   
    plt.subplot(row,col,idx_)  
    plt.title('v_var_{:.4f}_{:.4f}_{:.4f}'.format(np.std(v_img),v_img.min(),v_img.max()))
    plt.imshow(v_img)
    plt.subplot(row,col,idx_+1)
    plt.title('g_var_{:.6f}_{:.4f}_{:.4f}'.format(np.std(g_img),g_img.min(),g_img.max()))
    plt.imshow(g_img)
    plt.subplot(row,col,idx_+2)
    plt.title('image_{}'.format(x))
    plt.imshow(plot_img)
    idx_+=3

plt.show()

z = torch.randn(4, 512, device=device)
g, v, i = feature_map_gradient(G,z)


idx_=1
img = i.permute(0,2,3,1).detach().cpu().numpy()
img = (img+1)*127.5
img = img.astype(int)


row = 4
col = 14
idx_=1
plt.figure(figsize=(60,30))
for idx in range(row):
    for x in range(13):
        tmp = g['layer{}'.format(x)][idx]
        tmp = torch.var(tmp,axis=0)
        tmp = tmp.detach().cpu().numpy()
        plt.subplot(row,col,idx_)
        plt.title('layer{}'.format(x))
        plt.imshow(tmp)
        idx_ += 1
    plt.subplot(row,col,idx_)
    plt.title('layer{}'.format(x))
    plt.imshow(img[idx])
    idx_ += 1

plt.show()