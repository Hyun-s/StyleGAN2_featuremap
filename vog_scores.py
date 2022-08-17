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

out = []


for x in tqdm(range(10000)):
    G.eval()
    D.eval()
    z = torch.randn(4, 512, device=device)
    dic = G.mapping(z)
    const = dic['z']
    wp = dic['w']


    grad_X = wp.detach()
    grad_X.requires_grad = True

    fake_img = G(grad_X)
    fake_pred = D(fake_img['image'])
    g_loss = g_nonsaturating_loss(fake_pred)
    g_loss.backward()

    grad = grad_X.grad.data

    img = fake_img['image'].permute(0,2,3,1).detach().cpu().numpy()
    img = (img+1) * 127.5
    img = img.astype(int)
    np_grad = grad.cpu().numpy()
    np_w = wp.detach().cpu().numpy()
    grads = np.std(np_grad,axis=1)
    for (w, gr,im) in zip(np_w,grads,img):
        out.append((w,gr))        

df = pd.DataFrame(df)
df.to_csv('cat_40000.csv',index=False)