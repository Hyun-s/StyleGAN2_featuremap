import os, sys
from model import pggan, stylegan2
import torch
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from torch.utils.data import TensorDataset,DataLoader
from PIL import Image
import matplotlib.image as im


batch_size = 16

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
#/home/hyun_s/stylegan2/StyleGAN2_featuremap/stylegan/stylegan2_cat256.pth
model_path = 'stylegan/stylegan2_cat256.pth'
G, _ = stylegan2(path=model_path, res=256)
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
g = torch.Tensor(v_score)


output_dir = '/home/hyun_s/Gans/StyleGAN2_featuremap/g_image'
idx = 0


zs = TensorDataset(zs,g)
loader = DataLoader(zs,batch_size=10,shuffle=False)
df = []
for z, g in tqdm(loader):
    out = G(z.to(device))['image']
    out = out.permute(0,2,3,1)
    out = out.detach().cpu().numpy()
    out = (out+1)*127.5
    out = out.astype('uint8')
    g = g.detach().cpu().numpy()
    for img,g_s in zip(out, g):
        out_p = os.path.join(output_dir,str(idx)+'.png')
        im.imsave(out_p,img)

        dic = {'image_path': out_p,
               'gradient_var': g_s}
        df.append(dic)
        idx += 1

df = pd.DataFrame(df)

df.to_csv('/home/hyun_s/Gans/StyleGAN2_featuremap/path_score.csv',index=False)