import os
import models
import shutil
import sys
import time
from datetime import datetime
import torch as th
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch import optim
import torchvision as tv
from torchvision.utils  import save_image
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from PIL import Image
from ast import literal_eval
from apex import amp
from apex.parallel import SyncBatchNorm,convert_syncbn_model
from data import _DATASET_META_DATA, get_dataset
from utils.log import ResultsLog
from utils.absorb_bn import is_bn, search_absorbe_bn
from utils.misc import _META, AutoArgParser, GaussianSmoothing
from utils.meters import AverageMeter, accuracy, ConfusionMeter
from utils.mixup import MixUp
from preprocess import get_transform
_VERSION=15
############## cfg
cudnn.benchmark=True
def settings():
    preset = ''
    preset_sub_model = ''
    if preset!='':
        ckt_path = 'results/resnet44_cifar10/model_best.pth.tar'
        dataset = 'cifar10'
        model_config = "{'dataset': dataset,'depth':44}"
        model='resnet'
    else:
        ckt_path=''
        dataset=''
        model_config=''
        model=''
    use_amp=0
    sync_bn=0
    result_root_dir = 'results'
    exp_tag=''
    measure='' #'imagine-cifar10-r44-dd_only_r500'
    measure_steps=0 #500
    measure_ds_limit=0
    measure_seed=0
    mixup=0
    record_output_stats=0
    measure_fid=''
    fid_batch_size=512
    fid_self=False
    #split for measuring
    split='train'
    #reference split for fid
    split_fid=split
    output_temp=1.
    adversarial=0
    epsilon=0.1
    smooth_sigma = 1
    smooth_kernel = 5
    report_freq = 20
    batch_dup_factor = 4
    use_stats_loss = 1
    use_dd_loss = 0
    use_prior_loss = 0
    calc_stats_loss = 1
    calc_cont_loss = 0
    calc_smooth_loss = 1
    dd_loss_mode='exp'
    stats_loss_mode='kl'
    stat_scale = 1.
    cls_scale = 0.0004
    smooth_scale = 1. #20
    smooth_scale_decay={} #{100:0.2,5000:0.5,1000:0.5,1500:0.5}
    batch_size = 128
    betas = (0.9, 0.999)
    lr = 0.1
    replay_latent = 2000
    n_samples_to_generate = 20
    DEBUG_SHOW = 0
    gen_resize_ratio=1.0
    masking = False
    SGD=1
    return locals()

parser=AutoArgParser()
parser.add_argument('-d',default=[0],type=int,nargs='+')
parser.add_argument('-lr_drop_replays',default=[800, 1500, 3000],type=int,nargs='+')
parser.add_argument('-snapshots_replay',default= [1000],type=int,nargs='+')
parser.auto_add(settings())

class _MODEL_META(_META):
    _ATTRS = ['ds_meta','dataset','ckt_path','model','factory_method','model_config']
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

_R18_IMGNT_Q4W4A=_MODEL_META(
    ds_meta=_DATASET_META_DATA['imagenet'],
    dataset='imagenet',
    ckt_path='results/resnet18-Q4w4a_{}/model_best.pth.tar',
    ckt_models={'CE':'imagenet_CE','dist':'imagenet_distilled',
                'BNSKL':'imaginet_no_dd_kl_r1000',
                'BNSKL_I':'imaginet_dd-exp_kl_r1000'},
    model='resnet',
    factory_method=models.resnet,
    model_config={'depth': 18, 'conv1':{'a': 8,'w':8},'fc':{'a': 8,'w':8},
                  'activations_numbits':4,'weights_numbits':4, 'bias_quant':False,'quantize':True}
)
_R44_C10=_MODEL_META(
    ds_meta = _DATASET_META_DATA['cifar10'],
    dataset = 'cifar10',
    ckt_path ='results/resnet44_cifar10/model_best.pth.tar',
    model='resnet',
    factory_method=models.resnet,
    model_config={'dataset':'cifar10'}
)
_R44_C100=_MODEL_META(
    ds_meta=_DATASET_META_DATA['cifar100'],
    dataset='cifar100',
    ckt_path ='results/resnet44_ba_m40_cifar100/model_best.pth.tar',
    model='resnet',
    factory_method=models.resnet,
    model_config={'dataset':'cifar100','depth':44}
)
_R28_10_C100=_MODEL_META(
    ds_meta=_DATASET_META_DATA['cifar100'],
    dataset='cifar100',
    ckt_path ='results/wresnet28-10_ba_m10_cifar100/checkpoint.pth.tar',
    model='resnet',
    factory_method=models.resnet,
    model_config={'dataset':'cifar100','depth':28 ,'width':[160,320,640]}
)
_R18_IMGNT=_MODEL_META(
    ds_meta=_DATASET_META_DATA['imagenet'],
    dataset='imagenet',
    ckt_path ='',
    model='resnet',
    factory_method=tv.models.resnet18,
    model_config={'pretrained':True}
)
_DNS121_IMGNT=_MODEL_META(
    ds_meta=_DATASET_META_DATA['imagenet'],
    dataset='imagenet',
    ckt_path ='',
    model='densenet',
    factory_method=getattr(tv.models,'densenet121'),
    model_config={'pretrained':True}
)
_MBL2_IMGNT=_MODEL_META(
    ds_meta=_DATASET_META_DATA['imagenet'],
    dataset='imagenet',
    ckt_path ='',
    model='mobilenet_v2',
    factory_method=getattr(tv.models,'mobilenet_v2'),
    model_config={'pretrained':True}
)

_VGG16_BN_IMGNT=_MODEL_META(
    ds_meta=_DATASET_META_DATA['imagenet'],
    dataset='imagenet',
    ckt_path ='',
    model='vgg16_bn',
    factory_method=getattr(tv.models,'vgg16_bn'),
    model_config={'pretrained':True},
    get_final_layer=lambda m: list(m._modules['classifier']._modules)[-1]
)
_VGG11_BN_IMGNT=_MODEL_META(
    ds_meta=_DATASET_META_DATA['imagenet'],
    dataset='imagenet',
    ckt_path ='',
    model='vgg11_bn',
    factory_method=getattr(tv.models,'vgg11_bn'),
    model_config={'pretrained':True},
    get_final_layer=lambda m: list(m._modules['classifier']._modules)[-1]
)

_MODEL_CONFIGS={
    'r44_cifar100':_R44_C100,
    'r44_cifar10':_R44_C10,
    'wr28-10_cifar100':_R28_10_C100,
    'r18_imagenet':_R18_IMGNT,
    'r18_q4w4a_imagenet':_R18_IMGNT_Q4W4A,
    'densenet121_imagenet':_DNS121_IMGNT,
    'mobilenet_v2_imagenet':_MBL2_IMGNT,
    'vgg_16-bn_imagenet':_VGG16_BN_IMGNT,
    'vgg_11-bn_imagenet': _VGG11_BN_IMGNT,

}


class GenLoader():
    def __init__(self, input_data, nclasses, target_mode='random', limit=10):
        self.data = input_data
        self.nclasses = nclasses
        self.target_mode = target_mode
        self.limit = limit

    class GenIterator():
        _TARGET_MODES = ['random', 'running']

        def __init__(self, input_data, nclasses, target_mode='random', limit=10):
            assert target_mode in GenLoader.GenIterator._TARGET_MODES
            self.data = input_data
            self.nclasses = nclasses
            self.target_mode = target_mode
            self.iter = 0
            self.iter_limit = limit

        def reset_buffer(self):
            # reset buffer
            with th.no_grad():
                self.data.detach()
                self.data.normal_()

        def __next__(self):
            self.reset_buffer()
            if self.iter > self.iter_limit:
                raise StopIteration
            if self.target_mode == 'running':
                target = (th.arange(self.data.shape[0], device=self.data.device) + self.iter * self.data.shape[
                    0]) % self.nclasses
            else:
                # random mode should normally be used unless batch size is much larger then number of classes
                target = th.randint(0, self.nclasses, (self.data.shape[0],), device=self.data.device)
            self.iter += 1
            return self.data, target

        def __len__(self):
            return self.iter_limit

    def __iter__(self):
        return GenLoader.GenIterator(self.data, self.nclasses, self.target_mode, self.limit)


class GussianSmoothingLoss(nn.Module):
    def __init__(self,sigma=1,kernel_size=3,channels=3):
        super(GussianSmoothingLoss,self).__init__()
        self.smoothing_op=GaussianSmoothing(channels=channels, kernel_size=kernel_size, sigma=sigma)
        self.loss_criterion=nn.MSELoss()

    def forward(self, input,show=False):
        smoothed=self.smoothing_op(input)
        if show:
            plot_grid(input)
            plot_grid(smoothed)

        # save_image(smoothed[:16].detach().cpu(), f'smoothed.jpg', nrow=4)
        # save_image(image[:16].detach().cpu(), f'not_smoothed.jpg', nrow=4)
        return self.loss_criterion(input, smoothed)


class ContinuityLoss(nn.Module):
    def __init__(self,criterion=nn.MSELoss(reduction=None),diag=True,beta=0.33):
        super(ContinuityLoss,self).__init__()
        self.criterion=criterion
        self.beta=beta
        self.diag=diag

    def forward(self,image):
        lateral1 = image[:,:,:-1,:].contiguous().view(image.size(0),-1)
        lateral2 = image[:,:,1:,:].contiguous().view(image.size(0),-1)
        lat=self.criterion(lateral1,lateral2)
        horizontal1 = image[:, :, :, :-1].contiguous().view(image.size(0),-1)
        horizontal2 = image[:, :, :, 1:].contiguous().view(image.size(0),-1)
        hor=self.criterion(horizontal1,horizontal2)
        if self.diag:
            diagonal1 = image[:, :, :-1, :-1].contiguous().view(image.size(0),-1)
            diagonal2 = image[:, :, 1:, 1:].contiguous().view(image.size(0),-1)
            diag=self.criterion(diagonal1,diagonal2)
            # s=th.cat([lateral1,horizontal1,diagonal1],-1)
            # t=th.cat([lateral2,horizontal2,diagonal2],-1)
        else:
            # s = th.cat([lateral1, horizontal1], -1)
            # t = th.cat([lateral2, horizontal2], -1)
            diag=0
        return (hor+lat+diag).pow(self.beta).sum()
        #return self.criterion(s,t)


class RandCropResize(nn.Module):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self,min=0.33):
        super(RandCropResize,self).__init__()
        self.min = min

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        b=img.size(0)
        h = img.size(2)
        w = img.size(3)
        min_h=int(self.min * h)
        min_w=int(self.min * w)
        hight = th.randint(min_h,h,(b,))
        width = th.randint(min_w,w,(b,))
        #center position
        y = th.randint(min_h//2,h,(b,))
        x = th.randint(min_w//2,w,(b,))
        # size of crop around center
        y1 = th.clamp(y - hight // 2, 0, h)
        y2 = th.clamp(y + hight // 2, 0, h)
        x1 = th.clamp(x - width // 2, 0, w)
        x2 = th.clamp(x + width // 2, 0, w)
        #
        # img_ = img[:,y1: y2, x1: x2]
        # img = th.nn.functional.upsample(img_,size=img.shape[1:],mode='bilinear').squeeze()
        im = []
        for i, (y1_, y2_, x1_, x2_) in enumerate(zip(y1, y2, x1, x2)):
            im += [th.nn.functional.interpolate(img[i:i + 1, :, y1_: y2_, x1_: x2_], size=img.shape[2:],mode='bilinear')]
        img=th.cat(im)
        #mask = mask.expand_as(img)
        #img = img * mask
        return img


class Flip(nn.Module):
    """Randomly mask out one or more patches from an image.
    Args:
        prob (int): flip probability
    """
    def __init__(self,prob=0.2):
        super(Flip,self).__init__()
        self.flip = prob

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            flipped image w.p. flip
        """
        #lateral
        if th.rand(1) < self.flip:
            img = img.flip(1)
        #horizontal
        if th.rand(1) < self.flip:
            img = img.flip(2)
        return img


class Cutout(nn.Module):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, max_num_holes=5,ratio=1/3):
        super(Cutout,self).__init__()
        self.max_num_holes = max_num_holes
        self.ratio = ratio

    def forward(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        b = img.size(0)
        h = img.size(2)
        w = img.size(3)

        mask = th.ones((b,h,w),device=img.device)
        for i in range(b):
            for n in range(th.randint(self.max_num_holes,(1,))):
                hight = th.randint(1,int(h * self.ratio),(1,))
                width = th.randint(1,int(w * self.ratio),(1,))
                y = th.randint(h,(1,))
                x = th.randint(w,(1,))

                y1 = th.clamp(y - hight // 2, 0, h)
                y2 = th.clamp(y + hight // 2, 0, h)
                x1 = th.clamp(x - width // 2, 0, w)
                x2 = th.clamp(x + width // 2, 0, w)

                mask[i:i+1,y1: y2, x1: x2] = 0.

        mask = mask.unsqueeze(1).expand_as(img)
        img = img * mask
        return img


class AugmentImage(nn.Module):
    _AUGS=['flip','crop','cutout']
    def __init__(self,augmentations = {'flip':{},'crop':{},'cutout':{}}):
        super(AugmentImage,self).__init__()
        self.augs=[]
        if 'flip' in augmentations:
            self.flip = Flip(**augmentations['flip'])
            self.augs.append(self.flip)
        # else:
        #     self.Flipper = None
        if 'crop' in augmentations:
            self.crop = RandCropResize(**augmentations['crop'])
            self.augs.append(self.crop)
        # else:
        #     self.Cropper = None
        if 'cutout' in augmentations:
            self.cutout = Cutout(**augmentations['cutout'])
            self.augs.append(self.cutout)
        # else:
        #     self.Cutter = None
        self.n_augs=len(augmentations)

    def forward(self, input):
        rand_aug_ids=th.randperm(self.n_augs).tolist()[:th.randint(self.n_augs,(1,)).item()]
        i=input
        for aug_id in rand_aug_ids:
            aug=self.augs[aug_id]
            i = aug(i)
        return i


class AugmentBatch(nn.Module):
    def __init__(self,dup_factor=8,aug_conf={}):
        super(AugmentBatch,self).__init__()
        self.dups=dup_factor
        self.img_augmenter=AugmentImage(**aug_conf)

    def forward(self,samples_,labels_,other=None):
        samp = []
        labs = [labels_]*self.dups
        for _ in range(self.dups):
            samp+=[self.img_augmenter(samples_)]

        samples = th.cat(samp)
        labels = th.cat(labs)
        if other is None:
            return samples,labels
        others = th.cat([other] * self.dups)
        return samples, (labels, others)


def freeze_params(model):
    for param in model.parameters(True):
        param.require_grad = False


def Gaussian_KL(mu1,sigma1,mu2,sigma2):
    return -1/2 + th.log(sigma2)-th.log(sigma1)+(sigma1.pow(2)+(mu1-mu2).pow(2))/(2*sigma2.pow(2))


def Gaussian_KLNorm(mu1,sigma1,epsilon=1e-8):
    #assert all(sigma1>1e-8)
    return -1/2 - th.log(sigma1+epsilon) + (sigma1.pow(2)+(mu1).pow(2))/2


def Gaussian_sym_KLNorm(mu,sigma,eps=1e-3):
    sigma_p2=sigma.pow(2)+eps
    mu_p2=mu.pow(2)
    inverse_sigma_p2=1/(sigma_p2)
    return 0.5*(sigma_p2+mu_p2+inverse_sigma_p2+(1+mu_p2)*inverse_sigma_p2) - 1


def calc_stats_loss(loss_stat,inputs,stat_list,mode='kl'):
    in_mu, in_std = inputs.mean((0, 2, 3)), inputs.transpose(1, 0).contiguous().view((3, -1)).std(1)
    stat_list.append((in_mu,in_std))
    if mode == 'mse':
        for m, v in stat_list:
            loss_stat += m.pow(2).mean()
            loss_stat += (v - 1).pow(2).mean()
    else:
        for i,(m,v) in enumerate(stat_list):
            if mode=='kl':
                kl = Gaussian_KLNorm(m,v).mean()
            elif mode== 'sym':
                kl = 0.5*Gaussian_sym_KLNorm(m,v).mean()
            with th.no_grad():
                if kl/len(stat_list)> 0.5:
                    mse_error_mu=m.pow(2).mean()
                    mse_error_sigma=(v-1).pow(2).mean()
                    print(f'high divergence in layer {i}: {kl/len(stat_list)}\nmse: mu-{mse_error_mu}\tsigma-{mse_error_sigma}')
            loss_stat=loss_stat+kl

    ret_val=loss_stat / len(stat_list)
    stat_list.clear ()
    return ret_val


def plot_grid(image_tensor,samples= 16,nrow=4,denorm_meta=None):
    g=image_tensor[:samples].detach().cpu()
    if denorm_meta:
        mean = th.tensor(denorm_meta.mean, requires_grad=False).reshape((1, 3, 1, 1))
        std = th.tensor(denorm_meta.std, requires_grad=False).reshape((1,3, 1, 1))
        g = g*std+mean
    g = tv.utils.make_grid(g, nrow=nrow)
    plt.imshow(g.permute((1, 2, 0)))
    #plt.waitforbuttonpress()
    pass


def layer_stats_hook(stat_list,device):
    def stat_record(m,inputs,outputs):
        mean = inputs[0].mean((0,2,3)).to(device)
        std = inputs[0].transpose(1,0).contiguous().view((inputs[0].size(1),-1)).std(1).to(device)
        stat_list.append((mean,std))
    return stat_record

def sqrt_newton_schulz_autograd(A, numIters=100, dtype=th.float64,debug=False,eps=1e-6):
    ## based on https://github.com/msubhransu/matrix-sqrt
    def compute_error(A, sA):
        normA = th.sqrt(th.sum(th.sum(A * A, dim=1), dim=1))
        error = A - th.bmm(sA, sA)
        error = th.sqrt((error * error).sum(dim=1).sum(dim=1)) / normA
        return th.mean(error)

    source_type=A.dtype
    A+= th.eye(A.size(0),dtype=A.dtype,device=A.device) * eps
    A=A.type(dtype)

    if A.dim()==2:
        A=A.unsqueeze(0)

    batchSize = A.size(0)
    dim = A.size(1)
    normA = A.mul(A).sum((1, 2)).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A))
    I = th.eye(dim,dtype=dtype,device=A.device).unsqueeze(0).repeat(batchSize, 1, 1)
    Z = th.eye(dim,dtype=dtype,device=A.device).unsqueeze(0).repeat(batchSize, 1, 1)
    for i in range(numIters):
        T_ = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T_)
        Z = T_.bmm(Z)
        sA = Y * th.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
    if debug:
        error = compute_error(A, sA)
        print('sqrtm error:', error,
              'max error wrt numpy:',abs(sA.squeeze().cpu().numpy()-linalg.sqrtm(A.squeeze().cpu().numpy()).real).max())

    return sA.type(source_type)

def np_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-8):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1).dot(sigma2)+ offset)

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    tr= np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
    diff_l2=diff.dot(diff)
    return (diff_l2 + tr),covmean,tr,diff_l2

def np_calculate_activation_statistics(features):
    mu = np.mean(features.cpu().numpy(), axis=0)
    sigma = np.cov(features.cpu().numpy(), rowvar=False)
    return mu, sigma
    
class InceptionDistance(nn.Module):
    #todo: FID results do not match previously reported scores, statistics as well as matrix factorization are almost
    # identical to numpy reference
    debug_diff = lambda x, y: abs(x - y.cpu().numpy()).mean()

    def __init__(self,model=None,ref_inputs=None,accumulate=False,model_grad=False,input_size=None,ref_path=None):
        super().__init__()
        self.input_size= input_size or ref_inputs.shape[2:]
        if model is None:
            self.input_size=(299,299)
            self._I3 =tv.models.inception_v3(True)
        else:
            assert hasattr(model,'fc')
            self._I3 = model
        assert self.input_size is not None and type(self.input_size[0]) == int
        if not model_grad:
            freeze_params(self._I3)

        self.pool_output=None
        self.fc_output=None
        self.mu=None
        self.sigma=None
        self.accumulate=accumulate

        if accumulate:
            self.collected_activations=None
            self.collected_ref_activations=None

        def feature_recording_hook(m,inputs,outputs):
            assert inputs[0] is not None
            print('updated logged features with input size',inputs[0].size())
            self.pool_output=inputs[0]
            self.fc_output=outputs[0]

        self._I3.fc.register_forward_hook(feature_recording_hook)
        self.ref_path = ref_path + '.pth' if ref_path is not None else None

        if ref_inputs is None:
            self.ref_mu=None
            self.ref_sigma=None
        else:
            self.to(ref_inputs.device)
            print('calc initial ref stats')
            with th.no_grad():
                self._update_features(ref_inputs,ref=True)
                self.update_ref_stats(self.pool_output,debug=True)

    def measure_or_load_ref_stats(self,ref_loader=None,limit=None,ref_path=None):
        ref_path= ref_path or self.ref_path
        assert ref_path is not None
        if not os.path.exists(ref_path + '.pth'):
            print(f'reference statistics are not found at {ref_path}.pth\ncollecting reference data')
            mode=self.accumulate
            self.accumulate=True
            with th.no_grad():
                for i, (fid_ref_b, _) in enumerate(ref_loader):
                    if i * fid_ref_b.size(0) >= limit:
                        break
                    fid_ref_b = fid_ref_b.to(self.device)
                    self.forward(ref=fid_ref_b, report=False)
            self.accumulate=mode
        else:
            self.ref_mu, self.ref_sigma = th.load(ref_path)

    def update_ref_stats(self,ref_feat,debug=False):
        print('overwriting referece staistics with new ones', ref_feat.shape[0])
        self.ref_mu, self.ref_sigma = self._calc_stat(ref_feat)
        if self.ref_path is not None and not os.path.exists(self.ref_path):
            print(f'saving reference measurements as {self.ref_path} based on {ref_feat.shape[0]} samples')
            th.save([self.ref_mu, self.ref_sigma],self.ref_path)
        if debug:
            self.np_ref_mu, self.np_ref_sigma = np_calculate_activation_statistics(ref_feat)

    def to(self,*args,**kwargs):
        print(args,kwargs)
        self._I3.to(*args,**kwargs)

    def train(self, mode=True):
        self._I3.train(mode)

    def _update_features(self,inputs,ref=False):
        if ref:
            with th.no_grad():
                train= self._I3.training
                self._I3.eval()
                if inputs.shape[2:]!=self.input_size:
                    inputs=F.interpolate(inputs,self.input_size)
                self._I3(inputs)
                self._I3.train(train)

            if self.accumulate:
                if self.collected_ref_activations is None:
                    self.collected_ref_activations=self.pool_output.cpu()
                else:
                    self.collected_ref_activations = th.cat([self.collected_ref_activations,self.pool_output.cpu()])

        else:
            if inputs.shape[2:] != self.input_size:
                inputs = F.interpolate(inputs, self.input_size)
            self._I3(inputs)
            if self.accumulate:
                if self.collected_activations is None:
                    self.collected_activations=self.pool_output.cpu(),self.fc_output.clone().cpu()
                else:
                    self.collected_activations = (th.cat([self.collected_activations[0],self.pool_output.cpu()]),
                                                  th.cat([self.collected_activations[1],self.fc_output.clone().cpu()]))

    def _calc_stat(self,feat_2d):
        assert feat_2d.dim() == 2
        #assume first dim is batch,second dim is features
        print(f'compute stats for {feat_2d.size(1)} variables based on {feat_2d.size(0)} examples')

        mu=feat_2d.mean(0,keepdim=True)
        z_mean_feat=feat_2d-mu
        sigma= z_mean_feat.transpose(1,0).matmul(z_mean_feat)/(z_mean_feat.size(0)-1)
        return mu,sigma

    def _calc_Frechet_from_stats(self,mu,sigma,ref_mu,ref_sigma):
        A = sigma.matmul(ref_sigma)
        B = sigma + ref_sigma - 2 * sqrt_newton_schulz_autograd(A, debug=False).squeeze()
        tr = th.trace(B)
        diff = (ref_mu - mu).squeeze()
        diff_l2 = diff.dot(diff)
        FId = th.clamp(diff_l2 + tr, 0)
        return FId


    def _calc_Frechet(self,features,ref_features=None,debug=False):
        mu,sigma = self._calc_stat(features)
        if ref_features is not None:
            with th.no_grad():
                ref_mu, ref_sigma = self._calc_stat(ref_features)
        else:
            if self.ref_mu is None and self.collected_ref_activations is not None:
                self.update_ref_stats(self.collected_ref_activations,debug=debug)
            else:
                assert self.ref_mu is not None
                assert self.ref_sigma is not None
                # using precomputed values
            ref_mu, ref_sigma=self.ref_mu,self.ref_sigma
        ref_mu, ref_sigma = ref_mu.to(mu.device), ref_sigma.to(mu.device)
        FID=self._calc_Frechet_from_stats(mu,sigma,ref_mu, ref_sigma)
        return FID

    def _calc_IS(self,features):
        return None

    def forward(self,inputs=None,ref=None,debug=False,report=True,reset_buffers_on_report=True):
        # report - is used to report on the statistics of the collected activations,
        # False: statistics are only on the most recent inputs.
        # reset_buffers_on_report-only when module accomulation is on
        # False:statistics of inputs and references are computed from scratch after every invocation, while both buffers keep tracking history.
        # True: history will reset after reporting accumulated results. Reference statistics are kept as long as no new referece inputs are passed.
        if ref is not None:
            # calculate fresh reference features for current batch
            self._update_features(ref,True)
            ref=self.pool_output

        if inputs is not None:
            # update prediction if inputs are provided, none inputs are usefull when the same model model is used
            # for inference and measuring IS/FID on same data.
            self._update_features(inputs)

        if not report:
            return

        if self.accumulate and report:
            assert self.collected_activations is not None, 'first run some input data'

            if self.collected_ref_activations is not None:
                # update reference stats
                self.update_ref_stats(self.collected_ref_activations,debug=debug)
            else:
                assert self.ref_mu is not None and self.ref_sigma is not None, 'please provided reference statistics'

            FId, IS = self._calc_Frechet(self.collected_activations[0],debug=debug), \
                                      self._calc_IS(self.collected_activations[1])

            if reset_buffers_on_report:
                # reset collected activations
                self.collected_activations = None
                self.collected_ref_activations = None

            return FId, IS

        else:
            assert inputs is not None
            # report FID on the input batch.
            FId,IS=self._calc_Frechet(self.pool_output,ref,debug=debug), self._calc_IS(self.fc_output)

            return FId,IS

def layer_c_norm_hook(norm_ranking_dict,device,collect):
    #collect norms per channel accross batch and spatial dimensions, compute softmax to normalize according to channel activity
    def c_norm_hook(m,inputs,outputs):
        if collect:
            norms = inputs[0].transpose(1,0).contiguous().view((inputs[0].size(1),-1)).norm(1).to(device)
            norm_ranking_dict[m] = F.softmax(norms)
    return c_norm_hook


def generate_channel_mask(norm_ranking_dict,batch_size=1):
    suppression_mask_dict={}
    for m,nr in norm_ranking_dict:
        sampler=th.distributions.Bernoulli(nr)
        suppression_mask_dict[m] = sampler.sample((batch_size,)).reshape(batch_size,-1,1,1)
    return suppression_mask_dict


def layer_output_mask(suppression_mask_dict,device):
    def c_mask_output_hook(m,inputs,outputs):
        outputs[0]=outputs[0]*suppression_mask_dict[m]
    return c_mask_output_hook

## from pytorch tutorial
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    data_grad = data_grad.sign()
    #sign_data_grad = th.clamp(data_grad,0,1)
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*data_grad
    # Adding clipping to maintain [0,1] range
    # Return the perturbed image
    #perturbed_image = th.clamp(perturbed_image, 0, 1)
    return perturbed_image

def forward(model, data_loader, inp_shape, args, device , batch_augment = None,normalize_inputs=None,
            optimizer=None, smooth_loss=None, cont_loss=None,stat_list=[],adversarial=False,
            mixer=None,FID=None,log=None,save_path='tmp',time_stamp=None,use_amp=False):
    CE = nn.CrossEntropyLoss().to(device)
    CE_meter = AverageMeter()
    output_meter = AverageMeter()
    output_meter_hard = AverageMeter()
    batch_time = AverageMeter()
    generator_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_avg = AverageMeter()
    loss_cont_avg = AverageMeter()
    loss_stat_avg = AverageMeter()
    loss_smoothness = AverageMeter()
    confusion = ConfusionMeter()
    hot_one_map = np.eye(args.nclasses)
    hot_one_map_th=th.tensor(hot_one_map,device=device)
    n_iter = (args.n_samples_to_generate * args.nclasses) // args.batch_size
    end = time.time()
    for step, (inputs_, labels_) in enumerate(data_loader):
        labels_ = labels_.to(device)
        inputs_ = inputs_.to(device)
        if optimizer is None and args.measure_ds_limit > 0 and step * inputs_.size(0) >= args.measure_ds_limit * args.nclasses:
            break

        ##currently broken
        # if FID:
        #    with th.no_grad():
        #        #aggregate target dataset activations
        #        FID(inputs,report=False)
        #        print('FID:',FId.item())
        if adversarial:
            ## choose arbitrary adv_targets
            ##todo optional: choose adv_targets closest to predicted class instead of arbitrary assignment
            adv_targets = (labels_ + th.randint_like(labels_, 1, args.nclasses - 2)) % args.nclasses
            inputs_.requires_grad = True
            inputs_.retain_grad()
            inp_ptr = inputs_
        else:
            adv_targets = None
        if not time_stamp:
            time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        for n_replay in range(args.replay_latent):
            if optimizer:
                with th.no_grad():
                    inputs_.data.mul_(255).add_(0.5).clamp_(0, 255).floor_().div_(255)
                #inputs = inputs_/inputs_.norm() #var((0,2,3),keepdim=True)

            prior_loss = th.zeros(1).to(device)
            loss_stat = th.zeros(1).to(device)
            loss = th.zeros(1).to(device)

            if smooth_loss:
                loss_smooth = smooth_loss(inputs_).mean()
                if optimizer and n_replay in args.smooth_scale_decay:
                    args.smooth_scale = args.smooth_scale * args.smooth_scale_decay[n_replay]
                    print(f'reducing blur by factor {args.smooth_scale_decay[n_replay]}')
                loss_smoothness.update(loss_smooth.item())
                prior_loss = prior_loss + loss_smooth * args.smooth_scale

            if cont_loss:
                loss_cont = cont_loss(inputs_).mean()
                loss_cont_avg.update(loss_cont.item())
                prior_loss = prior_loss + loss_cont

            if normalize_inputs:
                inputs = (inputs_ - normalize_inputs['mean']) / normalize_inputs['std']
            else:
                inputs = inputs_

            if batch_augment:
                inputs, labels = batch_augment(inputs, labels_, adv_targets)
                if adv_targets:
                    labels, adv_targets = labels
            else:
                labels = labels_

            if args.DEBUG_SHOW:
                plot_grid(inputs)

            if (inputs.shape[2:]) != inp_shape[1:]:
                inputs = nn.functional.interpolate(inputs, mode='bilinear', size=inp_shape[1:])

            if mixer:
                inputs = mixer(inputs, [0.5, inputs.size(0), True])

            out = model(inputs)

            if args.measure:
                soft_out = F.softmax(out.detach() / args.output_temp).cpu().numpy()
                output_meter.update(soft_out.mean(0))
                output_meter_hard.update(hot_one_map[soft_out.argmax(1)].mean(0))
                if labels.max()<= out.size(-1):
                    CE_meter.update(CE(out.detach(), labels))

            ## model activations statistics loss
            if args.calc_stats_loss:
                # norm to compare to the 0,1 input statistics
                loss_stat = calc_stats_loss(loss_stat, inputs, stat_list, args.stats_loss_mode)

            if adversarial:
                adv_loss = adversarial(out, labels)  # adv_targets)
                adv_loss.backward()
                data_grad = inp_ptr.grad.data
                # Call FGSM Attack
                with th.no_grad():
                    perturbed_data = fgsm_attack(inp_ptr, args.epsilon,
                                                 th.cat([data_grad] * (len(inputs) // len(inp_ptr))))
                    if args.DEBUG_SHOW:
                        plot_grid(inputs, denorm_meta=_DATASET_META_DATA[args.dataset])
                        plot_grid(perturbed_data, denorm_meta=_DATASET_META_DATA[args.dataset])
                    fooled_out = model(perturbed_data)
                    fooled_stats_loss = calc_stats_loss(th.zeros(1, device=device), perturbed_data, stat_list,
                                                        mode=args.stats_loss_mode)
                    t1, t5 = accuracy(out.detach(), labels.detach(), (1, 5))
                    t1_, t5_ = accuracy(fooled_out.detach(), labels.detach(), (1, 5))
                    fooled_t1, fooled_t5 = accuracy(fooled_out.detach(), adv_targets.detach(), (1, 5))
                    print('top1 before {}\ttop1 after {}\ttop1 fooled {}\t||\tstats pre {}\tstats post {}'.format(
                        t1.item(), t1_.item(), fooled_t5.item(), loss_stat.item(), fooled_stats_loss.item()))
                    model.zero_grad()
                    # keep adversarials measures to report mean adversarial stats loss value and confusion
                    loss_stat = fooled_stats_loss
                    out = fooled_out

            if args.use_dd_loss:
                if args.dd_loss_mode=='kl':
                    if mixer:
                        targets = mixer(target=labels, n_class=args.nclasses)
                    else:
                        targets = labels
                    loss = F.kl_div(F.log_softmax(out),targets,reduction='mean')*10 #*args.cls_scale
                elif args.dd_loss_mode == 'exp':
                    ## exponentioal dd loss contribution decays as logit norm grows
                    loss = (th.exp(-(out.gather(1, labels.unsqueeze(1)).mean() * args.cls_scale)))
                elif args.dd_loss_mode == 'ce':
                    loss = CE(out, labels) * args.cls_scale
                else:
                    ## linear loss
                    loss = -out.gather(1, labels.unsqueeze(1)).mean() * args.cls_scale

                # add small CE loss to reduce confidence (may add artifacts to reduce confusion with similar classes)
                # loss_ce = criterion(out,label) * 0.001
                # loss = loss + loss_ce

            if args.use_prior_loss:
                loss = loss + prior_loss

            if args.use_stats_loss:
                loss = loss + loss_stat * args.stat_scale
            #loss=loss.mean()
            ## update metrics
            batch_time.update(time.time() - end)
            end = time.time()
            loss_avg.update(loss.item(), args.batch_size)
            loss_stat_avg.update(loss_stat.item())
            t1, t5 = accuracy(out.detach(), labels.detach(), (1, 5))
            confusion.update(out.detach(), labels.detach())
            top1.update(t1.item() / 100)
            top5.update(t5.item() / 100)

            if optimizer:
                optimizer.zero_grad()
                if use_amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                if n_replay in args.lr_drop_replays:
                    print('lowering lr at replay {}'.format(n_replay))
                    for p_g in optimizer.param_groups:
                        p_g['lr'] *= 1e-1

                # with th.no_grad():
                #     g_norm=inputs_.grad.norm()
                #     print('grad norm,input norm')
                #     print(g_norm, inputs_.norm())
                #     r_noise=th.randn_like(inputs_.grad)
                #     r_noise/=r_noise.norm()
                #     grad_to_noise_scale=0.5/np.log(n_replay+1)
                #     inputs_.grad+=g_norm*grad_to_noise_scale*r_noise
                #     if (n_replay + 1)%args.report_freq == 0:
                #         print(inputs_.grad.norm(),grad_to_noise_scale)
                    #clip_grad_norm_(inputs,3)
                optimizer.step()
                ## save snapshot samples

                if n_replay + 1 in args.snapshots_replay:
                    print(
                        f'replay {(n_replay + 1)}\t global loss {loss_avg.avg:.4f}, '
                        f'statistics loss {loss_stat_avg.avg:.4f}, smoothness {loss_smoothness.avg:.4f}, '
                        f'top1 {top1.avg:.2f} top5 {top5.avg:.2f}'
                    )
                    print('saving deep dreams')
                    snapshot_path = os.path.join(save_path, f'r{n_replay + 1}')
                    save_batched_images(inputs_, labels_, snapshot_path, f'{time_stamp}_r{n_replay + 1}',
                                        (10, 5) if step < 5 else None, f'snapshot_e{step}r{n_replay + 1}')

                if n_replay % args.report_freq == 0 or n_replay in args.snapshots_replay:
                    if step == 0 and log:
                        log.add(replay=n_replay, loss=loss_avg.avg, loss_smooth=loss_smoothness.avg,
                                loss_cont=loss_cont_avg.avg, loss_stats=loss_stat_avg.avg, top1=1 - top1.avg)
                    print(
                        f'iteration {step}/{n_iter}\t replay {n_replay}\t time {batch_time.avg:.2f}, generator time '
                        f'{generator_time.avg:.2f} global loss {loss_avg.avg:.4f}, statistics loss '
                        f'{loss_stat_avg.avg:.4f}, smoothness {loss_smoothness.avg:.4f}, '
                        f'top1 {top1.avg:.2f} top5 {top5.avg:.2f}')

        print(f'global loss {loss_avg.avg:.4f}, statistics loss '
            f'{loss_stat_avg.avg:.4f}, smoothness {loss_smoothness.avg:.4f}, '
            f'top1 {top1.avg:.2f} top5 {top5.avg:.2f}')

        # reset stats
        if step == 0 and log:
            # log.plot(x='replay',y=['loss','loss_stats','loss_smooth','loss_cont','top1'],title=exp_tag,legend=['Global Loss','Statistics Loss','Smoothing Loss','Continuity Loss','Top1 Error'])
            log.plot(x='replay', y=['loss_stats', 'top1'], title='accuracy vs stats',
                     legend=['Statistics Loss', 'Top1 Error'])
            log.plot(x='replay', y=['loss_stats', 'loss_smooth'], title='stats vs smoothness',
                     legend=['Statistics Loss', 'Gaussian Smoothing Loss'])
            log.save()

        if optimizer and args.replay_latent> 1:
            print('resetting optimizer')
            if args.SGD:
                optimizer = optim.SGD([data_loader.data], lr=args.lr, momentum=0.9)
            else:
                optimizer = optim.Adam([data_loader.data], lr=args.lr, betas=args.betas)
            if use_amp:
                _,optimizer=amp.initialize(nn.Module(),optimizer,opt_level='O1')
            print('RESETTING METRICS')
            top1.reset()
            top5.reset()
            loss_avg.reset()
            loss_cont_avg.reset()
            loss_stat_avg.reset()
            loss_smoothness.reset()

    final_results = f'BNloss_{args.stats_loss_mode}-{loss_stat_avg.avg:.3f}_prior-{loss_smoothness.avg:.3f}_top1-{top1.avg * 100:.2f}_CE-{CE_meter.avg:.3f}'
    if args.measure:
        t_name = f'model-'
        if args.preset != '':
            t_name += args.preset
        else:
            t_name += args.model
        t_name += f'_measure-{args.measure}-{args.split}_bs-{args.batch_size}_seed-{args.measure_seed}'
        if args.measure_ds_limit > 0:
            t_name += f'_lim-{args.measure_ds_limit}'
        if mixer:
            t_name += '_mixup'
        mean_output = th.tensor(output_meter.avg, requires_grad=False)
        mean_output_hard = th.tensor(output_meter_hard.avg, requires_grad=False)
        tensors = [mean_output, mean_output_hard, confusion.confusion, confusion.per_class_accuracy]
        t_tags = ['soft', 'hard', 'confusion', 'per_class_accuracy']
        regs = {}
        for t, n in zip(tensors, t_tags):
            regs[n] = t
        if args.split == 'val':
            fig1, ax1 = plt.subplots()
            # ax1.set_ylim(0, 6/args.nclasses)
            ax1.set_title(t_name + '-worst per class accuracy')
            val, id = regs['per_class_accuracy'].sort()
            ax1.bar([str(i.item()) for i in id[:10]], val[:10], width=1)
            ax1.legend(t_tags)
            fig1.set_size_inches(12, 9)
            fig1.savefig(t_name + '_worst_per_class_accuracy.jpg')
        fig, ax = plt.subplots()
        ax.set_ylim(0, 4 / args.nclasses)
        ax.set_title(t_name)
        for t, n in zip(tensors[:2], t_tags[:2]):
            # th.save(t, f'{t_name}_{n}.pth')
            val, id = t.sort(descending=True)
            ax.bar([str(i.item()) for i in id], val, width=1)
        ax.legend(t_tags)
        fig.set_size_inches(12, 9)
        save_fig_path = t_name + final_results
        fig.savefig(save_fig_path + '.jpg')
        if FID:
            fid, _ = FID(report=True)
            fid.sqrt_()
            regs['fid_ref_desc'] = FID.ref_path[:-4]
            final_results += '_' + FID.ref_path[:-4] + f'_score-{fid.item():.3f}'
        regs['final_results'] = final_results
        th.save(regs, f'Summary_{t_name}.pth')

        print(final_results)


def main():
    args=parser.parse_args()
    args.result_root_dir=os.path.join(args.result_root_dir,'generator')
    ############### setup
    device_id = args.d
    device = f'cuda:{device_id[0]}'
    if args.preset in _MODEL_CONFIGS:
        cfg=_MODEL_CONFIGS[args.preset]
        args.ckt_path=cfg.ckt_path
        if args.preset_sub_model!='' and hasattr(cfg,'ckt_models'):
            args.ckt_path=args.ckt_path.format(cfg.ckt_models[args.preset_sub_model])
            args.preset+='_'+cfg.ckt_models[args.preset_sub_model]
        args.dataset=cfg.dataset
        args.model_config=cfg.model_config
        args.model=cfg.model
        meta = cfg.ds_meta
        args.exp_tag=args.preset+args.preset_sub_model+args.exp_tag
        #last_layer_fetcher=getattr(cfg,'get_final_layer',None)
        factory_method = cfg.factory_method
    else:
        args.model_config=dict(**literal_eval(args.model_config))
        factory_method=getattr(models,args.model)
        meta = _DATASET_META_DATA[args.model_config.get('dataset',args.dataset)]

    #last_layer_fetcher = last_layer_fetcher or lambda m: list(m._modules.values())[-1]
    exp_start_time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    args.nclasses = meta.nclasses
    inp_shape = meta.shape
    mean = th.tensor(meta.mean, requires_grad=False).reshape((1, 3, 1, 1)).to(device)
    std = th.tensor(meta.std, requires_grad=False).reshape((1, 3, 1, 1)).to(device)
    input_stats = {'mean':mean,'std':std}

    smooth_loss=None
    cont_loss=None

    model = factory_method(**args.model_config)
    if args.ckt_path != '':
        model.load_state_dict(th.load(args.ckt_path,map_location='cpu')['state_dict'])
    model.to(device)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    print(f"number of parameters: {num_parameters}")

    ### todo: simplify by absorbing all bn parameters and returning the list of affine transformation parameters
    ## absorb batch normalization so layers statistics are normalized. we want to track the batch statistics for
    # the loss function for each iteration, setting momentum to 1 lets us access this information
    if args.calc_stats_loss:
        print('partially absorbing batch normalization parameters')
        search_absorbe_bn(model,remove_bn=False,keep_modifiers=True)

    if args.measure == '':
        # generate samples with a smaller size then the expected input, this is an additional image scaling prior
        if args.gen_resize_ratio != 1:
            gen_shape = list(inp_shape)
            gen_shape[1:] = [int(i / args.gen_resize_ratio + 0.5) for i in gen_shape[1:]]
            gen_shape = tuple(gen_shape)
        else:
            gen_shape = inp_shape

        dream_image = th.nn.Parameter(th.randn((args.batch_size,) + gen_shape, device=device))
        if args.SGD:
            print('using SGD')
            optimizer_im = optim.SGD([dream_image], lr=args.lr, momentum=0.9)
        else:
            print('using ADAM')
            optimizer_im = optim.Adam([dream_image], lr=args.lr, betas=args.betas)#,weight_decay=1e-8)

        if args.use_amp and optimizer_im:
            #print(f'using amp {amp_opt}')
            model, optimizer = amp.initialize(model, optimizer_im, opt_level='O1')

        if args.sync_bn:
            print('converting batch norms',args.sync_bn)
            model = convert_syncbn_model(model)

    freeze_params(model)
    model.eval()

    if args.calc_cont_loss:
        cont_loss=ContinuityLoss()
        cont_loss.to(device)
    if args.calc_smooth_loss:
        smooth_loss=GussianSmoothingLoss(args.smooth_sigma,args.smooth_kernel)
        smooth_loss.to(device)
    if args.batch_dup_factor > 0:
        b_aug = AugmentBatch(args.batch_dup_factor)
        b_aug.to(device)
    else:
        b_aug=None

    if len(device_id)>1:
        if args.calc_smooth_loss:
            smooth_loss=th.nn.DataParallel(smooth_loss,device_id)
        if args.calc_cont_loss:
            cont_loss=th.nn.DataParallel(cont_loss,device_id)
        b_aug=None if b_aug is None else th.nn.DataParallel(b_aug,device_id)
        model = nn.DataParallel(model, device_id)

    if args.use_stats_loss:
        args.calc_stats_loss=1
    stat_list = []
    if args.calc_stats_loss:
        for m in model.modules():
            if is_bn(m) or isinstance(m,SyncBatchNorm):
                m.register_forward_hook(layer_stats_hook(stat_list,device))

    if args.mixup:
        print('WARNING - running with INPUT MIXING!')
        mixer = MixUp()
        mixer.to(device)
    else:
        mixer = None

    if args.measure_fid != '':
        fid_ref_dataset = get_dataset(args.measure_fid, args.split_fid,
                                      transform=get_transform(args.measure_fid, augment=False),
                                      limit=args.measure_ds_limit)
        print('FID reference dataset', fid_ref_dataset)
        fid_ref_loader = th.utils.data.DataLoader(fid_ref_dataset,
                                                  batch_size=args.fid_batch_size, shuffle=True,
                                                  num_workers=8, pin_memory=False, drop_last=False)

        # tot_ref_examples=min(len(fid_ref_dataset), len(fid_ref_loader) * args.fid_batch_size)
        FID_ref_desc = f'fid:ref-{args.measure_fid}-{args.fid_split}_lim-{args.measure_ds_limit}'

        if args.fid_self == True:
            m_name = args.model + '-' + args.model_cfg.get('depth', '')
            FID_ref_desc = f'_source-{m_name}-{args.dataset}'
            FID = InceptionDistance(model=model, input_size=args.input_shape[1:], accumulate=True,
                                    ref_path=FID_ref_desc)
        else:
            FID_ref_desc += '_source-inceptionV3-imagenet'
            FID = InceptionDistance(input_size=(299, 299), accumulate=True, ref_path=FID_ref_desc)

        FID.to(device)
        FID.measure_or_load_ref_stats(fid_ref_loader)
    else:
        FID = None
    ## measure
    if args.measure != '':
        args.replay_latent=1

        #data loader MUST SHUFFLE the data for bnstat-measurement!
        th.random.manual_seed(args.measure_seed)
        ds=get_dataset(args.measure, args.split,
                       get_transform(args.measure, augment=False,input_size=inp_shape[1:]),
                       limit=args.measure_ds_limit)

        print('MEASURE ds:',ds)
        if args.measure_steps > 0:
            sampler = th.utils.data.RandomSampler(ds, replacement=True,
                                                     num_samples=args.measure_steps * args.batch_size)
        else:
            sampler = None
        train_loader = th.utils.data.DataLoader(
            ds, sampler=sampler,
            batch_size=args.batch_size, shuffle=True, #(sampler is None),
            num_workers=8, pin_memory=False, drop_last=False)
        if args.adversarial:
            adversarial_l=nn.CrossEntropyLoss()
            adversarial_l.to(device)
            forward(model, train_loader, inp_shape, args, device, None, None, None, smooth_loss, cont_loss,
                    stat_list,adversarial=adversarial_l,mixer=mixer,FID=FID)
        else:
            with th.no_grad():
                forward(model,train_loader,inp_shape,args,device,None,None,None,smooth_loss,cont_loss,stat_list,mixer=mixer,FID=FID)
        exit(0)

    #### gen
    #todo need to create a special iterator for generation
    exp_tag = f'v{_VERSION}'
    if args.SGD:
        exp_tag += f'_SGD'
    else:
        exp_tag += f'_adam'
    exp_tag += f'_lr{args.lr}_ba{args.batch_dup_factor}'
    if args.use_stats_loss:
        exp_tag += f'_stats-{args.stats_loss_mode}'
    if args.use_dd_loss:
        exp_tag += '_dd-{}_scale-{}'.format(args.dd_loss_mode, args.cls_scale)
    if args.use_prior_loss:
        if args.calc_smooth_loss:
            exp_tag += f'_smooth_k{args.smooth_kernel}_s{args.smooth_sigma}'
        if args.calc_cont_loss:
            exp_tag += '_cont_loss'
    if args.gen_resize_ratio !=1.0:
        exp_tag+= f'_gen_ration_{args.gen_resize_ratio}'

    save_path=os.path.join(args.result_root_dir,args.dataset,args.model+'{}'.format(args.model_config.get('depth','')),exp_tag,exp_start_time_stamp)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('save path',save_path)
    shutil.copy(sys.argv[0],os.path.join(save_path,'backup_gen.py'))
    print(args)
    log=ResultsLog(os.path.join(save_path,'results'),params=args)

    if args.replay_latent not in args.snapshots_replay:
        args.snapshots_replay.append(args.replay_latent)

    train_loader = GenLoader(dream_image,nclasses=args.nclasses,
                             target_mode='random' if args.use_dd_loss and args.batch_size < max(args.nclasses,64)
                             else 'running',
                             limit=int((args.n_samples_to_generate * args.nclasses) / args.batch_size)+0.5)

    forward(model, train_loader, inp_shape, args, device, b_aug, input_stats, optimizer_im, smooth_loss, cont_loss, stat_list,
            mixer=mixer, FID=FID,save_path=save_path,time_stamp=exp_start_time_stamp,log=log,use_amp=args.use_amp)

def save_batched_images(input_batch,label,root_path,prefix='',plot_sample_shape=None,plot_sample_name='sample'):
    with th.no_grad():
        for i, s in enumerate(input_batch):
            # save class to file
            buffer = s.clone()
            ndarr = buffer.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', th.uint8).numpy()
            im = Image.fromarray(ndarr)
            output_dirname = os.path.join(root_path,f'{label[i]:04}')
            filename = f'{prefix}.{i}.png'

            if not os.path.isdir(output_dirname):
                os.makedirs(output_dirname)
            im.save(os.path.join(output_dirname,filename))

        if plot_sample_shape is not None:
            sample = input_batch[: plot_sample_shape[0]*plot_sample_shape[1]]
            save_image(sample.detach().cpu(),
                       os.path.join(root_path,f'{plot_sample_name}.png'),
                       nrow=plot_sample_shape[0])

if __name__ == '__main__':
    print(sys.argv)
    main()

def merge(res_root = 'results/generator/cifar10/resnet',target_dirs = ['v11_no_dd_loss'],
          subdir_targets = [10,20,30,40,50,100,200,500,1000],
          output_dir = ['r44_cifar10_no_dd_loss_v11_merged']):
    import os
    import subprocess

    ops=[]
    for t, o in zip(target_dirs, output_dir):
        t_path = os.path.join(res_root, t)
        o_path = os.path.join(res_root, o)
        if not os.path.exists(o_path):
            os.makedirs(o_path)
        # filter
        for f_n in os.listdir(t_path):
            f_p = os.path.join(t_path, f_n)
            if os.path.isdir(f_p) and all([ os.path.isdir(os.path.join(f_p, f'r{s_t_n}')) for s_t_n in subdir_targets ]):
                print(f_p)
                for s_t_n in subdir_targets:
                    ops.append(subprocess.Popen('rsync -v -r --exclude snapshot*.png {} {}'.format(os.path.join(f_p, f'r{s_t_n}' ),o_path),shell=True))
        for i in ops:
            i.wait()
        ops.clear()
        for s_t_n in subdir_targets:
            final_dir=os.path.join(o_path+'_final',f'r{s_t_n}','train')
            os.popen('rm -rf {} && mkdir -p {} && mv {}/* {}'.format(final_dir,final_dir,os.path.join(o_path,f'r{s_t_n}'),final_dir))

def plot():
    res = {}
    files = [
        'final_mean_prediction_r18_q4w4a_imagenet_imaginet_dd-exp_kl_r1000imagenet_lim-0_batch_size-512_seed-0.pth',
        'final_mean_prediction_r18_q4w4a_imagenet_imaginet_no_dd_kl_r1000imagenet_lim-0_batch_size-512_seed-0.pth',
        'final_mean_prediction_r18_imagenetimagenet_lim-0_batch_size-512_seed-0.pth',
        'Summary_model-r18_imagenet_measure-imagine-imagenet-r18-no_dd_kl-r1000-train_bs-512_seed-0.pth']
    titles = ['QBNS_KL+I_val', 'QBNS_KL_val', 'reference_val', 'reference_BNS']
    # load summaries
    for f, n in zip(files, titles):
        res[n] = th.load(f)
    # sort according to soft mean prediction over BNS dataset (classes with weak predictions first)
    sorted_soft_BNS_ids = res['reference_BNS']['soft'].sort()[1]
    norm_diff = lambda x, y: (x - y) / y

    def calc_degradation(n_tail_sampels=200, verbose=False, only_degradation=True):
        soft_unrepresented_BNS_id = sorted_soft_BNS_ids[:n_tail_sampels]
        ref_under_acc = res['reference_val']['per_class_accuracy'][soft_unrepresented_BNS_id]
        BNS_I_under_acc = res['QBNS_KL+I_val']['per_class_accuracy'][soft_unrepresented_BNS_id]
        BNS_under_acc = res['QBNS_KL_val']['per_class_accuracy'][soft_unrepresented_BNS_id]
        BNS_normalized_diff = norm_diff(BNS_under_acc, ref_under_acc)
        BNS_I_normalized_diff = norm_diff(BNS_I_under_acc, ref_under_acc)
        if only_degradation:
            BNS_normalized_diff = BNS_normalized_diff[BNS_normalized_diff < 0]
            BNS_I_normalized_diff = BNS_I_normalized_diff[BNS_I_normalized_diff < 0]
        BNS_I_m, BNS_I_s, BNS_m, BNS_s = BNS_I_normalized_diff.mean(), \
                                         BNS_I_normalized_diff.std(), \
                                         BNS_normalized_diff.mean(), \
                                         BNS_I_normalized_diff.std()
        if verbose:
            print('BNS_I {:.4f} ({:.4f})\tBNS {:.4f} ({:.4f})'.format(BNS_I_m, BNS_I_s, BNS_m, BNS_s))
        return BNS_I_m, BNS_I_s, BNS_m, BNS_s

    plot_resulotion = 100
    x = []
    m = []
    std = []
    for i in range(plot_resulotion):
        x_ = (i + 1) * 1000 // plot_resulotion
        x.append(x_)
        BNS_I_m, BNS_I_s, BNS_m, BNS_s = calc_degradation(x_)
        m.append([BNS_I_m, BNS_m])
        std.append([BNS_I_s, BNS_s])

    import numpy as np
    from matplotlib import pyplot as plt

    x, m, s = np.array(x), np.array(m), np.array(std)
    plt.plot(x, m)
    plt.legend(['BNS_I', 'BNS'])
    plt.fill_between(x, m[:, 0] - s[:, 0], m[:, 0] + s[:, 0], alpha=0.5)
    plt.fill_between(x, m[:, 1] - s[:, 1], m[:, 1] + s[:, 1], alpha=0.5)
    # plt.ylim(-.1,0)
    plt.title('Tail Degradation')
    plt.ylabel('normalized (mean-per-class) accuracy degradation')
    plt.xlabel('number of tail classes (order by mean softmax output)')
    plt.show()