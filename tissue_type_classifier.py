import os
import glob
import numpy as np

from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate

def get_modified_dataset(src, dst):
    mus = glob.glob(src+'MUS/*.tif')
    stroma = glob.glob(src+'STR/*.tif')
    for i in range(len(mus)):
        if i%2 == 1:
            os.remove(mus[i])
    for i in range(len(stroma)):
        if i%2 == 1:
            os.remove(stroma[i])
    mus = glob.glob(src+'MUS/*.tif')
    for i in range(len(mus)):
        os.rename(mus[i], os.path.split(os.path.split(mus[i])[0])[0] + '/STR/' + os.path.split(mus[i])[1])
    os.rename(src, dst)

tfms = get_transforms(flip_vert=True, max_rotate=None, 
                     max_zoom=1.1, max_lighting=None, 
                     max_warp=None, p_affine=0.75, p_lighting=0)

if __name__ == '__main__':

    # modify NCT-CRC-HE-100K dataset
    get_modified_dataset(src = '/path/to/NCT-CRC-HE-100K/', dst = '/path/to/Modified-NCT-CRC-HE-100K/')
    # modify CRC-VAL-HE-7K dataset
    get_modified_dataset(src = '/path/to/CRC-VAL-HE-7K/', dst = '/path/to/Modified-CRC-VAL-HE-7K/')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    np.random.seed(218)

    path = Path('/path/to/Modified-NCT-CRC-HE-100K/')
    data = ImageDataBunch.from_folder(
                        path, 
                        train=".",
                        valid_pct=0.2,
                        ds_tfms=tfms,
                        size=224,
                        bs=64,
                        num_workers=4).normalize(imagenet_stats)

    learn = cnn_learner(data, models.resnet34, metrics=error_rate)
    # learn.lr_find()
    # learn.recorder.plot()
    learn.fit_one_cycle(6, max_lr=1e-2)
    learn.unfreeze()
    # learn.lr_find(start_lr=1e-07, end_lr=1, stop_div=False)
    # learn.recorder.plot()
    learn.fit_one_cycle(4, slice(1e-6, 1e-4))
    learn.save('tissueType_classifier')

    test_path = Path('/path/to/Modified-CRC-VAL-HE-7K/')
    data_test = ImageDataBunch.from_folder(
                        test_path, 
                        train=".",
                        valid_pct=0,
                        ds_tfms=tfms,
                        size=224,
                        bs=64,
                        num_workers=4).normalize(imagenet_stats)

    learn = cnn_learner(data_test, models.resnet34, metrics=[error_rate, accuracy])
    learn.load('tissueType_classifier')
    preds, ys, losses = learn.get_preds(ds_type=DatasetType.Fix, with_loss=True)

    interp = ClassificationInterpretation(learn, preds, ys, losses)
    fig = interp.plot_confusion_matrix(return_fig=True)
    fig.savefig('/path/to/save/confusion_matrix.pdf', bbox_inches="tight")
    print(accuracy(preds, ys))