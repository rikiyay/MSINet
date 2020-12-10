import os
import random
import numpy as np
import pandas as pd
import h5py
from sklearn.metrics import confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
from utils import new_transforms

class MyTestDataset(torch.utils.data.Dataset):
    def __init__(self, path2hdf5, df, transform=None):

        h5 = h5py.File(path2hdf5)
        self.df = df
        self.h5_imgs = h5['img']
        self.h5_fnames = [i.decode('UTF-8') for i in h5['fnames']]
        self.transform=transform

    def __getitem__(self, index):
        fname = self.df.fnames[index]
        label = self.df.labels[index]
        h5_idx = self.h5_fnames.index(fname)
        img = self.h5_imgs[h5_idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)

def get_datasets_loaders(path2hdf5, transform, batchSize):

    h5 = h5py.File(path2hdf5)
    fnames = [i.decode('UTF-8') for i in h5['fnames']]
    ids = [i.decode('UTF-8') for i in h5['ids']]
    labels = [i for i in h5['labels']]
    df = pd.DataFrame(columns=['fnames', 'ids', 'labels'])
    df.fnames = fnames
    df.ids = ids
    df.labels = labels
    ext_dset = MyTestDataset(path2hdf5, df, transform=transform)
    ext_loader = torch.utils.data.DataLoader(ext_dset, batch_size=batchSize, shuffle=False)
        
    return ext_dset, ext_loader

def get_model(num_classes, model_path):
    model = models.mobilenet_v2(pretrained=True)
    model.classifier = nn.Sequential(nn.Dropout(p=0.25), nn.Linear(1280, num_classes))
    model.cuda()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)

    return model

def test_model(model, loader, dataset_size, criterion):
    
    print('-' * 10)
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    whole_probs = torch.FloatTensor(dataset_size)
    whole_labels = torch.LongTensor(dataset_size)  
    
    with torch.no_grad():

        for i, data in enumerate(loader):
            inputs = data[0].to(device)
            labels = torch.tensor(data[1], dtype=torch.long, device=device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            outputs = F.softmax(outputs, dim=1)
            whole_probs[i*batchSize:i*batchSize+inputs.size(0)]=outputs.detach()[:,1].clone()
            whole_labels[i*batchSize:i*batchSize+inputs.size(0)]=labels.detach().clone()

        total_loss = running_loss / dataset_size
        total_acc = running_corrects.double() / dataset_size

    print('Test Loss: {:.4f} Acc: {:.4f}'.format(total_loss, total_acc))

    return whole_probs.cpu().numpy(), whole_labels.cpu().numpy(), total_loss, total_acc

def evaluate(model, ext_dset, ext_loader, ext_dset_size, criterion):
    prob, label, loss, acc = test_model(model, ext_loader, ext_dset_size, criterion)   

    df_tile = pd.DataFrame(columns=['id', 'prob', 'label'])
    df_tile.id = ext_dset.df.ids.values.tolist()
    df_tile.prob = prob
    df_tile.label = label
    unique=np.unique(df_tile.id.values).tolist()
    pts = list(set([i[:12] for i in unique]))

    pt_prob=[]
    pt_pred=[]
    pt_label=[]
    for j in pts:
        slides = [s for s in unique if j in s]
        if len(slides) == 1:
            ave_prob=np.mean(df_tile[df_tile.id==slides[0]].prob.values)
        elif len(slides) > 1:
            ave_prob = 0
            for x in slides:
                ave_prob += np.mean(df_tile[df_tile.id==x].prob.values)
            ave_prob = ave_prob/len(slides)        
        ave_pred=1 if ave_prob>0.50 else 0
        label=df_tile[df_tile.id==slides[0]].label.values.tolist()[0]
        pt_prob.append(ave_prob)
        pt_pred.append(ave_pred)
        pt_label.append(label)

    return pt_prob, pt_pred, pt_label

def bootstrap_auc(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    n_bootstraps = n_bootstraps
    rng_seed = rng_seed
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(len(y_pred), size=len(y_pred))
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    bootstrapped_scores = np.array(bootstrapped_scores)

    print("AUROC: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
    print("Confidence interval for the AUROC score: [{:0.3f} - {:0.3}]".format(
        np.percentile(bootstrapped_scores, (2.5, 97.5))[0], np.percentile(bootstrapped_scores, (2.5, 97.5))[1]))
    
    return roc_auc_score(y_true, y_pred), np.percentile(bootstrapped_scores, (2.5, 97.5))

transform = transforms.Compose([transforms.ToPILImage(),
                                new_transforms.Resize((imgSize,imgSize)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    batchSize=32
    imgSize=int(224)
    num_classes = 2
    model_path = '/path/to/save/state_dict/msi_predictor.pth'

    model = get_model(num_classes, model_path)

    ext_path = '/path/to/tcga-crc.hdf5'
    ext_dset, ext_loader = get_datasets_loaders(ext_path, transform, batchSize)
    ext_dset_size = len(ext_dset)

    criterion = nn.CrossEntropyLoss()

    pt_prob, pt_pred, pt_label = evaluate(model, ext_dset, ext_loader, ext_dset_size, criterion)

    cm = confusion_matrix(pt_label, pt_pred)
    print(cm)

    acc = (cm[0][0]+cm[1][1])/len(pt_label)*100
    print(f'accuracy = {acc}')
    
    roc_auc, ci = bootstrap_auc(np.array(pt_label), np.array(pt_prob))