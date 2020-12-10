import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.special import softmax
import h5py

from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class HDF5Dataset(Dataset):
    def __init__(self, file_path, item_list, transform=None):
        super().__init__()
        h5_file = h5py.File(file_path)
        self.data = h5_file.get('img')
        self.items = item_list
        self.c = 7
        self.classes = ['ADI', 'DEB', 'LYM', 'MUC', 'NORM', 'STR', 'TUM']
        self.transform=transform
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if self.transform:
            return (self.transform(self.data[index,:,:,:]),
                    torch.tensor([index]).long())
        else:
            return (torch.from_numpy(self.data[index,:,:,:]).long(),
                    torch.tensor([index]).long())

def get_dataset(path2hdf5, transforms):
    h5 = h5py.File(path2hdf5, "r") # hdf5_file_tcga
    item_list = [i.decode('utf-8') for i in h5.get('fnames')]
    ds = HDF5Dataset(path2hdf5, item_list, transforms)
    return ds

def get_results(learn, ds_type):

    probs, _ = learn.get_preds(ds_type=ds_type)
    _, pred = torch.max(probs, 1)
    probs_softmax = softmax(probs.numpy(), axis=1)
    if ds_type == DatasetType.Fix:
        fnames = learn.data.train_ds.items
    elif ds_type == DatasetType.Valid:
        fnames = learn.data.valid_ds.items

    df = pd.DataFrame(
        data={
            'fname': fnames,
            'probs': probs_softmax.tolist(),
            'prob_max': probs_softmax.max(axis=1).tolist(),
            'pred': pred.tolist(),
        },
        columns=['fname', 'probs', 'prob_max', 'pred']
    )

    get_id = lambda x: x.split('_')[0]   
    df['ids'] = df['fname'].map(get_id)

    get_x = lambda x: int(x.split('_')[3])
    get_y = lambda x: int(x.split('_')[6])
    df['x'] = df['fname'].map(get_x)
    df['y'] = df['fname'].map(get_y)

    get_max_x = lambda x: int(x.split('_')[4])
    get_max_y = lambda x: int(x.split('_')[7][:-4])
    df['max_x'] = df['fname'].map(get_max_x)
    df['max_y'] = df['fname'].map(get_max_y)

    return df

def save_tissueMaps(df, unique, size, classes, colors, savepath, show=False):
    for h in range(len(unique)):
        df_temp = df[df.ids==unique[h]]
        whole = np.empty([0,(df_temp.max_x.iat[0]+1)*size, 3])
        
        for i in range(0,df_temp.max_y.iat[0]+1):
            h_map = []
            for j in range(0,df_temp.max_x.iat[0]+1):
                if [j,i] in df_temp[['x','y']].values.tolist():
                    p = df_temp[(df_temp.x==j) & (df_temp.y==i)].pred.values[0]
                    q = np.stack([np.stack([np.array(colors[classes[p]]) for _ in range(size)], axis=0) for _ in range(size)], axis=1)
                    h_map.append(q)
                else:
                    h_map.append(np.zeros([size,size, 3]))
            raw = np.concatenate(h_map,1)
            whole = np.append(whole,raw,axis=0)
        whole = whole.astype(np.uint8)
        if show:
            print(unique[h])
            plt.figure(figsize=(6,6))
            plt.imshow(whole)
            plt.show()
        whole_img = Image.fromarray((whole).astype(np.uint8))
        whole_img.save(savepath+unique[h]+'.png')

transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    np.random.seed(218)

    path_tcga = '/path/to/tcga_tiles.hdf5'
    tcga_ds = get_dataset(path_tcga, transforms)

    path_stanford = '/path/to/stanford_tiles.hdf5'
    stanford_ds = tcga_ds = get_dataset(path_tcga, transforms)

    whole_data = DataBunch.create(tcga_ds, stanford_ds, bs=64, num_workers=0)
    learn = cnn_learner(whole_data, models.resnet34, metrics=[error_rate, accuracy])
    learn.load('tissueType_classifier')

    df_tcga = get_results(learn, ds_type=DatasetType.Fix)
    tcga_unique = df_tcga.ids.unique().tolist()

    df_stanford = get_results(learn, ds_type=DatasetType.Valid)
    stanford_unique = df_stanford.ids.unique().tolist()

    size=10
    classes=['ADI', 'DEB', 'LYM', 'MUC', 'NORM', 'STR', 'TUM']
    colors={'ADI': [141, 141, 141], 'DEB': [189, 41, 153], 'LYM': [30, 149, 191],
            'MUC': [250, 216, 206], 'NORM': [204, 102, 0], 'STR': [247, 188, 10], 'TUM': [230, 73, 22]}

    save_tissueMaps(df_tcga, tcga_unique, size, classes, colors, savepath='/path/to/save/tissue_map_tcga/tmap_', show=False)
    save_tissueMaps(df_stanford, stanford_unique, size, classes, colors, savepath='/path/to/save/tissue_map_stanford/tmap_', show=False)