import os
import torchvision.datasets as datasets
import torch
from utils.dataset import RandomDatasetGenerator

__DATASETS_DEFAULT_PATH = 'Datasets'

def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH,limit=None):
    train = (split == 'train')
    if name == 'cifar10-raw':
        ds_dir_name = 'cifar10'
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
    elif name == 'stl10':
        return datasets.STL10(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name in ['imagenet','imagine-imagenet-r18','imagine-cifar10-r44','cifar10-raw']:
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        ds= datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
        if limit:
            ds=balance_image_folder_ds(ds, limit)

        return ds

    elif 'random' in name:
        if 'cifar' in name:
            mean = [.491, .482, .446]
            std = [.247, .243, .261]
            data_shape=(3,32,32)
            nclasses = 10 if 'cifar10' in name else 100
        elif 'imagenet' in name:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            data_shape=(3,224,224)
            nclasses=1000
        else:
            raise NotImplementedError

        limit = limit*nclasses if limit else None
        if train:
            return RandomDatasetGenerator(data_shape,mean,std,limit=limit,transform=transform,train=train)
        else:
            return get_dataset(name[7:],split,transform,target_transform,limit=limit)


def balance_image_folder_ds(dataset, samples_per_class,shuffle=False,seed=0):
    assert isinstance(dataset,datasets.DatasetFolder)
    if shuffle:
        import random
        random.seed(seed)
    samps = []
    num_classes = len(dataset.classes)
    samp_reg_per_class=[[]]*num_classes

    for s in dataset.samples:
        samp_reg_per_class[s[1]].append(s)
    for jj in range(num_classes):
        if shuffle:
            samps += random.sample(samp_reg_per_class[jj],samples_per_class)
        else:
            samps += samp_reg_per_class[jj][:samples_per_class]
    if hasattr(dataset,'imgs'):
        dataset.imgs = samps
    dataset.samples = samps
    return dataset
