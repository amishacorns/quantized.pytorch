import os
import warnings

import torch
import torchvision.datasets as datasets

from utils.dataset import RandomDatasetGenerator
from utils.misc import _META

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class _DS_META(_META):
    _ATTRS = ['nclasses', 'shape', 'mean', 'std']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_normalization(self):
        return {'mean': self.mean, 'std': self.std}

__DATASETS_DEFAULT_PATH = 'Datasets'
_CIFAR10=_DS_META(nclasses=10,shape=(3,32,32),mean=[.491, .482, .446],std=[.247, .243, .261])
_CIFAR100=_DS_META(nclasses=100,shape=(3,32,32),mean=[.491, .482, .446],std=[.247, .243, .261])
_STL10=_DS_META(nclasses=10,shape=(3,32,32),mean=[.491, .482, .446],std=[.247, .243, .261])
_IMAGENET=_DS_META(nclasses=1000,shape=(3,224,224),mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
_SVHN=_DS_META(nclasses=10,shape=(3,32,32),mean=[0.437,0.444,0.473],std=[0.198,0.201,0.197])
_MNIST=_DS_META(nclasses=10,shape=(3,32,32),mean=[0.5],std=[0.5])
_MNIST_3C=_DS_META(nclasses=10,shape=(3,32,32),mean=[0.131]*3,std=[0.308]*3)


_DATASET_META_DATA={
    'cifar10':_CIFAR10,
    'cifar100':_CIFAR100,
    'stl10':_STL10,
    'imagenet':_IMAGENET,
    'SVHN':_SVHN,
    'mnist':_MNIST,
    'mnist_3c':_MNIST_3C,
}

_IMAGINE_CONFIGS=[
    'no_dd_kl','no_dd_mse','no_dd_sym',
    'dd-exp_kl','dd-ce_kl','dd-exp_mse',
    'dd-exp', 'dd-ce']

def get_dataset(name, split='train', transform=None,
                target_transform=None, download=True, datasets_path=__DATASETS_DEFAULT_PATH,
                limit=None,shuffle_before_limit=False,limit_shuffle_seed=None,class_ids=None,per_class_limit=True):
    train = (split == 'train')
    if '+' in name:
        ds=None
        for ds_name in name.split('+'):
            ds_=get_dataset(ds_name, split, transform, target_transform,download, limit=limit,
                            shuffle_before_limit=shuffle_before_limit,datasets_path=__DATASETS_DEFAULT_PATH)
            if ds is None:
                ds=ds_
            else:
                ds += ds_
        return ds
    if name.endswith('-raw'):
        ds_dir_name = name[:-4]
    elif name.startswith('folder-'):
        ds_dir_name = name[7:]
    elif name == 'places365_standard-lsun':
        ds_dir_name = 'places365_standard'
        name = ds_dir_name
        class_ids = filter(lambda x: x not in [52, 66, 91, 92, 102, 121, 203, 215, 284, 334], range(365))

    elif name.startswith('DomainNet-'):
        domain, set = name.split('-')[1:3]
        class_ids = range(173) if set == 'A' else range(173, 345)
        if name.endswith('-measure') and train:
            ds_dir_name = os.path.join('DomainNet', 'measure', domain)
        else:
            ds_dir_name = os.path.join('DomainNet', 'train' if train else 'test', domain)


    elif name.endswith('-dogs') or name.endswith('-cats'):
        if name.startswith('imagenet-'):
            if name.endswith('dogs'):
                _ids = _imagenet_dogs.keys()
            else:
                _ids = _imagenet_cats.keys()
        else:
            _ids = [1] if name.endswith('dogs') else [0]

        return get_dataset(name[:-5], split, transform, target_transform, download, limit=limit,
                       shuffle_before_limit=True, datasets_path=__DATASETS_DEFAULT_PATH,
                       class_ids=_ids,per_class_limit=False,limit_shuffle_seed=0)
    elif name.startswith('imagine-'):
        if train:
            ds_dir_name=None
            for i_cfg in _IMAGINE_CONFIGS:
                idx=name.find(i_cfg)
                if idx >0:
                    ds_dir_name=os.path.join(name[:idx-1],i_cfg,name[idx+len(i_cfg)+1:])
                    #print(ds_dir_name)
                    break
            assert ds_dir_name is not None
        else:
            return get_dataset(name.split('-')[1], split, transform, target_transform,download, limit=limit,
                               shuffle_before_limit=shuffle_before_limit,
                               datasets_path=__DATASETS_DEFAULT_PATH)
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
    elif name == 'mnist' or name == 'mnist_3c':
        return datasets.MNIST(root=root,
                              train=train,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'SVHN':
        return datasets.SVHN(root=root,
                              split= 'test' if not train else split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif 'stl10' in name:
        if train and name.endswith('train_test'):
            return datasets.STL10(root=root,
                                  split='train',
                                  transform=transform,
                                  target_transform=target_transform,
                                  download=download) + datasets.STL10(root=root,
                                                                      split='test',
                                                                      transform=transform,
                                                                      target_transform=target_transform,
                                                                      download=download)
        return datasets.STL10(root=root,
                              split=split,
                              transform=transform,
                              target_transform=target_transform,
                              download=download)
    elif name == 'LSUN':
        return datasets.LSUN(root=root,
                             classes=split,
                             transform=transform,
                             target_transform=target_transform)
    elif name.startswith('folder'):
        ds = datasets.ImageFolder(root=root,
                                  transform=transform,
                                  target_transform=target_transform)
        ds = balance_image_folder_ds(ds, limit, per_class=per_class_limit, shuffle=shuffle_before_limit,
                                     seed=limit_shuffle_seed, class_ids=class_ids)
        return ds
    elif name in ['imagenet', 'cats_vs_dogs', 'places365_standard'] or any(i in name for i in ['imagine-', '-raw']):
        if train:
            root = os.path.join(root, 'train')
        else:
            root = os.path.join(root, 'val')
        ds = datasets.ImageFolder(root=root,
                                  transform=transform,
                                  target_transform=target_transform)
        if limit or class_ids:
            if 'no_dd' in name:
                ds = balance_image_folder_ds(ds, limit * len(ds.classes), per_class=False, shuffle=shuffle_before_limit,
                                             seed=limit_shuffle_seed)
            else:
                ds=balance_image_folder_ds(ds, limit,per_class=per_class_limit,shuffle=shuffle_before_limit,seed=limit_shuffle_seed,class_ids=class_ids)
        return ds
    elif name.startswith('DomainNet-'):
        ds = datasets.ImageFolder(root=root,
                                    transform=transform,
                                    target_transform=target_transform)
        return limit_ds(ds,-1,shuffle=False,allowed_classes=class_ids)

    elif name.startswith('random-'):
        ds_name = name[7:]
        if ds_name in _DATASET_META_DATA:
            meta=_DATASET_META_DATA[ds_name]
            nclasses, data_shape, mean, std = meta.get_attrs().values()
        else:
            raise NotImplementedError

        limit = limit*nclasses if limit else 1000*nclasses
        if train:
            return RandomDatasetGenerator(data_shape,mean,std,limit=limit,transform=transform,train=train)
        else:
            return get_dataset(name[7:],split, transform, target_transform,download, limit=limit,
                               shuffle_before_limit=shuffle_before_limit, datasets_path=__DATASETS_DEFAULT_PATH)

    elif hasattr(datasets,name):
        return getattr(datasets,name)(root=root,
                                split=train,
                                transform=transform,
                                target_transform=target_transform,
                                download=download)


def balance_image_folder_ds(dataset, n_samples=None,per_class=True,shuffle=False,seed=None,class_ids=None):
    assert isinstance(dataset,datasets.DatasetFolder)

    if shuffle:
        import random
        random.seed(seed)
        print(f'shufflling with seed {seed}')
    samps = []

    n_samples = n_samples or len(dataset)
    if per_class or class_ids is not None:
        samp_reg_per_class={}
        for s in dataset.samples:
            if s[1] in samp_reg_per_class:
                if shuffle or len(samp_reg_per_class[s[1]])<n_samples:
                    samp_reg_per_class[s[1]]+=[s]
            elif class_ids is not None and s[1] not in class_ids:
                continue
            else:
                samp_reg_per_class[s[1]]=[s]

        for k in samp_reg_per_class.keys():
            if shuffle and per_class:
                samps += random.sample(samp_reg_per_class[k],n_samples)
            else:
                samps += samp_reg_per_class[k]

        if not per_class and shuffle and len(samps)>n_samples:
            samps = random.sample(samps, n_samples)
    else:
        if shuffle:
            samps = random.sample(dataset.samples,n_samples)
        else:
            samps = dataset.samples[:n_samples]

    if hasattr(dataset,'imgs'):
        dataset.imgs = samps
    dataset.samples = samps
    return dataset

def limit_ds(dataset, n_samples=-1,per_class=True,shuffle=True,seed=0,allowed_classes=None):
    if not hasattr(dataset,'targets'):
        if hasattr(dataset,'labels'):
            dataset.targets=dataset.labels
        else:
            assert 0, 'dataset not supported'

    if per_class:
        id_reg_per_class = {}
        # map id to class label
        for e, t in enumerate(dataset.targets):
            if t in id_reg_per_class:
                id_reg_per_class[t] += [e]
            else:
                id_reg_per_class[t] = [e]

        ids = []
        # shuffle and clip each class
        for t in id_reg_per_class.keys():
            if allowed_classes and t not in allowed_classes:
                continue

            class_ids = torch.tensor(id_reg_per_class[t])
            if shuffle:
                class_ids = class_ids[torch.randperm(len(class_ids), generator=torch.Generator().manual_seed(seed))]

            ids.append(torch.tensor(class_ids[:n_samples]))
        ids = torch.cat(ids)
    else:
        if shuffle:
            ids = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:n_samples]
        else:
            ids = torch.arange(0, len(dataset))[:n_samples]
    ds = torch.utils.data.Subset(dataset,ids)
    ds.targets = torch.tensor(dataset.targets)[ids]
    if allowed_classes:
        ds.classes = [c for i, c in enumerate(dataset.classes) if i in allowed_classes]
    else:
        ds.classes = dataset.classes

    return ds

_imagenet_dogs = {
 151: 'Chihuahua',
 152: 'Japanese spaniel',
 153: 'Maltese dog, Maltese terrier, Maltese',
 154: 'Pekinese, Pekingese, Peke',
 155: 'Shih-Tzu',
 156: 'Blenheim spaniel',
 157: 'papillon',
 158: 'toy terrier',
 159: 'Rhodesian ridgeback',
 160: 'Afghan hound, Afghan',
 161: 'basset, basset hound',
 162: 'beagle',
 163: 'bloodhound, sleuthhound',
 164: 'bluetick',
 165: 'black-and-tan coonhound',
 166: 'Walker hound, Walker foxhound',
 167: 'English foxhound',
 168: 'redbone',
 169: 'borzoi, Russian wolfhound',
 170: 'Irish wolfhound',
 171: 'Italian greyhound',
 172: 'whippet',
 173: 'Ibizan hound, Ibizan Podenco',
 174: 'Norwegian elkhound, elkhound',
 175: 'otterhound, otter hound',
 176: 'Saluki, gazelle hound',
 177: 'Scottish deerhound, deerhound',
 178: 'Weimaraner',
 179: 'Staffordshire bullterrier, Staffordshire bull terrier',
 180: 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier',
 181: 'Bedlington terrier',
 182: 'Border terrier',
 183: 'Kerry blue terrier',
 184: 'Irish terrier',
 185: 'Norfolk terrier',
 186: 'Norwich terrier',
 187: 'Yorkshire terrier',
 188: 'wire-haired fox terrier',
 189: 'Lakeland terrier',
 190: 'Sealyham terrier, Sealyham',
 191: 'Airedale, Airedale terrier',
 192: 'cairn, cairn terrier',
 193: 'Australian terrier',
 194: 'Dandie Dinmont, Dandie Dinmont terrier',
 195: 'Boston bull, Boston terrier',
 196: 'miniature schnauzer',
 197: 'giant schnauzer',
 198: 'standard schnauzer',
 199: 'Scotch terrier, Scottish terrier, Scottie',
 200: 'Tibetan terrier, chrysanthemum dog',
 201: 'silky terrier, Sydney silky',
 202: 'soft-coated wheaten terrier',
 203: 'West Highland white terrier',
 204: 'Lhasa, Lhasa apso',
 205: 'flat-coated retriever',
 206: 'curly-coated retriever',
 207: 'golden retriever',
 208: 'Labrador retriever',
 209: 'Chesapeake Bay retriever',
 210: 'German short-haired pointer',
 211: 'vizsla, Hungarian pointer',
 212: 'English setter',
 213: 'Irish setter, red setter',
 214: 'Gordon setter',
 215: 'Brittany spaniel',
 216: 'clumber, clumber spaniel',
 217: 'English springer, English springer spaniel',
 218: 'Welsh springer spaniel',
 219: 'cocker spaniel, English cocker spaniel, cocker',
 220: 'Sussex spaniel',
 221: 'Irish water spaniel',
 222: 'kuvasz',
 223: 'schipperke',
 224: 'groenendael',
 225: 'malinois',
 226: 'briard',
 227: 'kelpie',
 228: 'komondor',
 229: 'Old English sheepdog, bobtail',
 230: 'Shetland sheepdog, Shetland sheep dog, Shetland',
 231: 'collie',
 232: 'Border collie',
 233: 'Bouvier des Flandres, Bouviers des Flandres',
 234: 'Rottweiler',
 235: 'German shepherd, German shepherd dog, German police dog, alsatian',
 236: 'Doberman, Doberman pinscher',
 237: 'miniature pinscher',
 238: 'Greater Swiss Mountain dog',
 239: 'Bernese mountain dog',
 240: 'Appenzeller',
 241: 'EntleBucher',
 242: 'boxer',
 243: 'bull mastiff',
 244: 'Tibetan mastiff',
 245: 'French bulldog',
 246: 'Great Dane',
 247: 'Saint Bernard, St Bernard',
 248: 'Eskimo dog, husky',
 249: 'malamute, malemute, Alaskan malamute',
 250: 'Siberian husky',
 251: 'dalmatian, coach dog, carriage dog',
 252: 'affenpinscher, monkey pinscher, monkey dog',
 253: 'basenji',
 254: 'pug, pug-dog',
 255: 'Leonberg',
 256: 'Newfoundland, Newfoundland dog',
 257: 'Great Pyrenees',
 258: 'Samoyed, Samoyede',
 259: 'Pomeranian',
 260: 'chow, chow chow',
 261: 'keeshond',
 262: 'Brabancon griffon',
 263: 'Pembroke, Pembroke Welsh corgi',
 264: 'Cardigan, Cardigan Welsh corgi',
 265: 'toy poodle',
 266: 'miniature poodle',
 267: 'standard poodle',
 268: 'Mexican hairless'}

_imagenet_cats = {
 281: 'tabby, tabby cat',
 282: 'tiger cat',
 283: 'Persian cat',
 284: 'Siamese cat, Siamese',
 285: 'Egyptian cat'}