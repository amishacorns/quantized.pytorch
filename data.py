import os
import torchvision.datasets as datasets

#__DATASETS_DEFAULT_PATH = '/media/hdd/Datasets/'
__DATASETS_DEFAULT_PATH = 'Datasets'

def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH):
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
    elif name in ['imagenet','imaginet','randomnet','imagine-cifar10-r44','cifar10-raw']:

        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        return datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
