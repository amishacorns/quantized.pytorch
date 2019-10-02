import os
import torchvision.datasets as datasets
import torch
from utils.dataset import RandomDatasetGenerator
from utils.misc import _META
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class _DS_META(_META):
    _ATTRS=['nclasses','shape','mean','std']
    def __init__(self,**kwargs):
        super().__init__(**kwargs)


__DATASETS_DEFAULT_PATH = 'Datasets'
_CIFAR10=_DS_META(nclasses=10,shape=(3,32,32),mean=[.491, .482, .446],std=[.247, .243, .261])
_CIFAR100=_DS_META(nclasses=100,shape=(3,32,32),mean=[.491, .482, .446],std=[.247, .243, .261])
_IMAGENET=_DS_META(nclasses=1000,shape=(3,224,224),mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

_DATASET_META_DATA={
    'cifar10':_CIFAR10,
    'cifar100':_CIFAR100,
    'imagenet':_IMAGENET
}

_IMAGINE_CONFIGS={
    'no_dd',
    'no_bn',
    'no_dd_smooth_loss',
    'smooth_loss',
    'dd_only'
}

def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH,
                limit=None,shuffle_before_limit=False):
    train = (split == 'train')
    if name.endswith('-raw'):
        if name[:-4] in ['cifar10','cifar100']:
            ds_dir_name = name[:-4]
        else:
            raise NotImplementedError
    elif name.startswith('imagine-'):
        if train:
            ds_dir_name=None
            for i_cfg in _IMAGINE_CONFIGS:
                idx=name.find(i_cfg)
                if idx >0:
                    ds_dir_name=os.path.join(name[:idx-1],i_cfg,name[idx+len(i_cfg)+1:])
                    break
            assert ds_dir_name is not None
        else:
            return get_dataset(name.split('-')[1], split, transform, target_transform, limit=limit)

    else:
        ds_dir_name = name
    root = os.path.join(datasets_path, ds_dir_name)
    if name == 'cifar10':
        return datasets.CIFAR10(root=root,
                                train=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)
    elif name == 'cifar100':
        return datasets.CIFAR100(root=root,
                                 train=train,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=download)
    elif name == 'mnist':
        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'SVHN':
        return datasets.SVHN(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'stl10':
        return datasets.STL10(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name in ['imagenet'] or any(i in name for i in ['imagine-', '-raw']):
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        ds= datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
        if limit:
            if 'no_dd' in name:
                ds = balance_image_folder_ds(ds, limit*len(ds.classes),per_class=False,shuffle=shuffle_before_limit)
            else:
                ds=balance_image_folder_ds(ds, limit,True,shuffle=shuffle_before_limit)

        return ds
    elif name.startswith('random-'):
        ds_name = name[7:]
        if ds_name in _DATASET_META_DATA:
            meta=_DATASET_META_DATA[ds_name]
            nclasses, data_shape, mean, std = meta.get_attrs().values()
        else:
            raise NotImplementedError

        limit = limit*nclasses if limit else None
        if train:
            return RandomDatasetGenerator(data_shape,mean,std,limit=limit,transform=transform,train=train)
        else:
            return get_dataset(name[7:],split,transform,target_transform,limit=limit)


def balance_image_folder_ds(dataset, n_samples,per_class=True,shuffle=False,seed=0):
    assert isinstance(dataset,datasets.DatasetFolder)
    if shuffle:
        import random
        random.seed(seed)
    samps = []
    if per_class:
        samp_reg_per_class=[[]]*num_classes
        num_classes = len(dataset.classes)
        for s in dataset.samples:
            samp_reg_per_class[s[1]].append(s)

        for jj in range(num_classes):
            if shuffle:
                samps += random.sample(samp_reg_per_class[jj],n_samples)
            else:
                samps += samp_reg_per_class[jj][:n_samples]
    else:
        if shuffle:
            samps = random.sample(dataset.samples,n_samples)
        else:
            samps = dataset.samples[:n_samples]

    if hasattr(dataset,'imgs'):
        dataset.imgs = samps
    dataset.samples = samps
    return dataset
