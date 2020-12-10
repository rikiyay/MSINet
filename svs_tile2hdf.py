
import os
import glob
import random
import numpy as np
import pandas as pd
import cv2
import tables

img_path = 'path/to/svs_tiles/'
# change below to '/path/to/tcga_tiles.hdf5' for tcga dataset and '/path/to/stanford_tiles.hdf5' for stanford dataset
hdf5_path = '/path/to/svs_tiles.hdf5'

random.seed(218)

print('Start Glob.Glob')
all_tiles = glob.glob(img_path+'*/*.png')
print('End Glob.Glob')

tiles = []
ids = []
fnames = []

for j in all_tiles:    
    idx = '-'.join(os.path.split(j)[1].split('.')[0].split('-')[:-3])
    fname = os.path.basename(j)

    tiles.append(j)
    ids.append(idx)
    fnames.append(fname)

img_dtype = tables.UInt8Atom()
data_shape = (0, 256, 256, 3)
# open a hdf5 file and create earrays
hdf5_file = tables.open_file(hdf5_path, mode='w')
storage = hdf5_file.create_earray(hdf5_file.root, 'img', img_dtype, shape=data_shape)

err=[]

for i in range(len(all_tiles)):
    if i % 1000 == 0 and i > 1:
        print('Done: {}/{}'.format(i, len(all_tiles)))

    tile = all_tiles[i]
    try:
        img = cv2.imread(tile)
        if img.shape[0]!=256 or img.shape[1]!=256:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        storage.append(img[None])
    except:
        err.append(tile)
        
dellist = lambda items, indexes: [item for index, item in enumerate(items) if index not in indexes]

if len(err)!=0:
    idx = [tiles.index(i) for i in err]
    
    tiles = dellist(tiles, idx)    
    ids = dellist(ids, idx)
    fnames = dellist(fnames, idx)    
  
hdf5_file.create_array(hdf5_file.root, 'ids', ids)
hdf5_file.create_array(hdf5_file.root, 'fnames', fnames)

hdf5_file.close()