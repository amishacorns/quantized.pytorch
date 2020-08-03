import torch
import torchvision.transforms as transforms
import random
import numpy as np

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def scale_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.RandomCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    transforms.Compose(t_list)


def pad_random_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    if type(input_size) is tuple:
        padding = (int((scale_size - input_size[0]) / 2), int((scale_size - input_size[1]) / 2))
    else:
        padding = int((scale_size - input_size) / 2)
    return transforms.Compose([
        transforms.RandomCrop(input_size, padding=padding),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])
def inception_color_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
        ),
        Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
        transforms.Normalize(**normalize)
    ])

from data import _DATASET_META_DATA
def get_transform(name='imagenet', input_size=None,
                  scale_size=None, normalize=None, augment=True):
    if 'imagenet' in name or name in ['imaginet','randomnet','cats_vs_dogs']:
        normalize = normalize or __imagenet_stats
        scale_size = scale_size or 256
        input_size = input_size or 224
        if augment:
            return inception_preproccess(input_size, normalize=normalize)
        else:
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)
    elif any([i in name for i in ['cifar100', 'cifar10', 'stl10', 'SVHN']]):
        input_size = input_size or 32
        normalize = normalize or _DATASET_META_DATA.get(name,_DATASET_META_DATA['cifar10']).get_normalization()

        if augment:
            scale_size = scale_size or 40
            return pad_random_crop(input_size, scale_size=scale_size,
                                   normalize=normalize)
        else:
            scale_size = scale_size or 32
            return scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize)
    elif 'mnist' in name:
        normalize = normalize or _DATASET_META_DATA.get(name, _DATASET_META_DATA['mnist']).get_normalization()
        input_size = input_size or 28
        if name.endswith('_3c'):
            pre_transform = lambda org_trans: transforms.Compose([transforms.Resize(input_size),lambda x:x.convert('RGB'),org_trans])
        else:
            pre_transform = lambda org_trans : org_trans
        if augment:
            scale_size = scale_size or 32
            return pre_transform(pad_random_crop(input_size, scale_size=scale_size,
                                   normalize=normalize))
        else:
            scale_size = scale_size or 32
            return pre_transform(scale_crop(input_size=input_size,
                              scale_size=scale_size, normalize=normalize))


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class Grayscale(object):

    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587, gs[1]).add_(0.114, gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs


class Saturation(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Brightness(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class Contrast(object):

    def __init__(self, var):
        self.var = var

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(0, self.var)
        return img.lerp(gs, alpha)


class RandomOrder(object):
    """ Composes several transforms together in random order.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        if self.transforms is None:
            return img
        order = torch.randperm(len(self.transforms))
        for i in order:
            img = self.transforms[i](img)
        return img


class ColorJitter(RandomOrder):

    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))

class RandomNoise(object):
    _SUPPORTED_NOISE = ['uniform','normal']
    def __init__(self,type,ratio=0.05):
        assert type in RandomNoise._SUPPORTED_NOISE
        assert 0 < ratio < 1
        self.type = type
        self.ratio = ratio
        self.img = None
    def __call__(self, img):
        norm_signal = torch.norm(img)
        # set noise expectation to bias and variance to 1
        if self.type == 'uniform':
            alpha=1.7321
            noise = torch.distributions.Uniform(-alpha,alpha).sample(img.shape)
        elif self.type == 'normal':
            noise = torch.distributions.Normal(0,1).sample(img.shape)
        norm_noise = torch.norm(noise)
        factor = self.ratio * norm_signal / norm_noise
        return img * (1-self.ratio) + noise * factor

class ImgGhosting():
    def __init__(self, ratio=0.3, ghost_moment = 0.2, residual_init_rate = 0.25,fuse_distribution = torch.distributions.Beta(0.4,0.4)):
        self.ratio = ratio
        self.init_rate = residual_init_rate
        self.ghost_moment=ghost_moment
        assert 0 <= self.ratio / (1 - self.init_rate) <= 1
        #todo check exponential dist
        self.fuse_distribution = fuse_distribution
        self.residual = None

    def __call__(self, img):
        if self.residual and torch.rand(1) > self.init_rate:
            residual = self.residual.copy()
            #update residual
            self.residual = self.residual * self.ghost_moment + (1-self.ghost_moment) * img

            # ratio of ghosted images per sample
            if torch.rand(1) < self.ratio /(1- self.init_rate):
                gamma = self.fuse_distribution.sample()
                img = img * (1 - gamma) + gamma * residual

        else:
            self.residual = img

        return img

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, max_num_holes=10,ratio=1/4):#,area_threshold=0.65):
        super(Cutout,self).__init__()
        self.max_num_holes = max_num_holes
        self.ratio = ratio
        #self.area_threshold=area_threshold

    def __call__(self, img):

        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        #area = h*w
        mask = torch.ones((h,w),device=img.device)

        for n in range(torch.randint(self.max_num_holes,(1,))):
            hight = torch.randint(1,int(h * self.ratio),(1,))
            width = torch.randint(1,int(w * self.ratio),(1,))
            y = torch.randint(h,(1,))
            x = torch.randint(w,(1,))

            y1 = torch.clamp(y - hight // 2, 0, h)
            y2 = torch.clamp(y + hight // 2, 0, h)
            x1 = torch.clamp(x - width // 2, 0, w)
            x2 = torch.clamp(x + width // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
            # if mask.sum()/area > self.area_threshold:
            #     mask[y1: y2, x1: x2] = 1

        mask = mask.expand_as(img)
        img = img * mask
        return img
