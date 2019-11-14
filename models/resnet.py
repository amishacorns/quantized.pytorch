import torch
import torch.nn as nn
FirstConv2d = torch.nn.Conv2d
FC = torch.nn.Linear
import torchvision.transforms as transforms
import math
from utils.regime import lr_drops,exp_decay_lr,ramp_up_lr,cosine_anneal_lr
from utils.partial_class import partial_class
from .modules.quantize import get_bn_folding_module
from .modules.se import SEBlock
from .modules.checkpoint import CheckpointModule
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.mixup import MixUp

__all__ = ['resnet', 'resnet_se']

_DEFUALT_A_NBITS = 8
_DEFUALT_W_NBITS = 4
_DEFUALT_G_NBITS = None
_DEFAULT_BIAS_QUANT = False

def conv3x3(in_planes, out_planes, stride=1, groups=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=bias)


def init_model(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d) and m.affine:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    for m in model.modules():
        if isinstance(m, Bottleneck) and isinstance(m.bn3, nn.BatchNorm2d):
                nn.init.constant_(m.bn3.weight, 0)
        elif isinstance(m, BasicBlock) and isinstance(m.bn2, nn.BatchNorm2d):
                nn.init.constant_(m.bn2.weight, 0)

    model.fc.weight.data.normal_(0, 0.01)
    model.fc.bias.data.zero_()


def weight_decay_config(value=1e-4, log=False):
    return {'name': 'WeightDecay',
            'value': value,
            'log': log,
            'filter': {'parameter_name': lambda n: not n.endswith('bias'),
                       'module': lambda m: not isinstance(m, nn.BatchNorm2d)}
            }


class OTFBottleneck(nn.Module):
    def __init__(self, inplanes, planes,  stride=1, expansion=4, downsample=None, groups=1, residual_block=None, dropout=0.):
        super(OTFBottleneck, self).__init__()
        CBN = get_bn_folding_module(nn.Conv2d,nn.BatchNorm2d)
        dropout = 0 if dropout is None else dropout
        self.conv1bn = CBN(inplanes, planes, kernel_size=1, bias=True)
        self.conv2bn = CBN(planes, planes, stride=stride, groups=groups,bias=True,kernel_size=3,padding=1)
        self.conv3bn = CBN(planes, planes * expansion, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion

    def forward(self, x):
        residual = x
        out = self.conv1bn(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3bn(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class OTFBasicBlock(nn.Module):
    def __init__(self, inplanes, planes,  stride=1, expansion=1,
                 downsample=None, groups=1, residual_block=None, dropout=0.):
        super(OTFBasicBlock, self).__init__()
        CBN = get_bn_folding_module(nn.Conv2d,nn.BatchNorm2d)
        dropout = 0 if dropout is None else dropout
        self.conv1 = CBN(inplanes, planes, stride=stride, groups=groups,bias=True,kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = CBN(planes, expansion * planes, groups=groups,bias=True,kernel_size=3,padding=1)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class OTFResNet(nn.Module):

    def __init__(self):
        super(OTFResNet, self).__init__()
        #self.flatten = Reshape(-1)
    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None, dropout=None, mixup=False):
        CBN = get_bn_folding_module(nn.Conv2d,nn.BatchNorm2d)
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(CBN(self.inplanes, out_planes,kernel_size=1, stride=stride, bias=True))

        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride, expansion=expansion,
                            downsample=downsample, groups=groups, residual_block=residual_block, dropout=dropout))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups,
                                residual_block=residual_block, dropout=dropout))
        if mixup:
            layers.append(MixUp())
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


class OTFResNet_imagenet(OTFResNet):
    def __init__(self, num_classes=1000, inplanes=64,
                 block=OTFBottleneck, residual_block=None, layers=[3, 4, 23, 3],
                 width=[64, 128, 256, 512], expansion=4, groups=[1, 1, 1, 1],
                 regime='normal', scale_lr=1, checkpoint_segments=0, mixup=False,absorb_bn=False,**kwargs):
        super(OTFResNet_imagenet, self).__init__()
        self.inplanes = inplanes
        if FirstConv2d == torch.nn.modules.conv.Conv2d:
            self.conv1 = FirstConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=absorb_bn)
            if absorb_bn:
                self.bn1 = lambda x: x
            else:
                self.bn1 = nn.BatchNorm2d(self.inplanes)
        else:
            CBNFirstConv2d = get_bn_folding_module(FirstConv2d,nn.BatchNorm2d)
            self.conv1 = CBNFirstConv2d (3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i in range(len(layers)):
            layer = self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion,
                                     stride=1 if i == 0 else 2, residual_block=residual_block, groups=groups[i],
                                     mixup=mixup)
            if checkpoint_segments > 0:
                layer_checkpoint_segments = min(checkpoint_segments, layers[i])
                layer = CheckpointModule(layer, layer_checkpoint_segments)
            setattr(self, 'layer%s' % str(i + 1), layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten=Reshape(-1)
        self.fc = nn.Linear(width[-1] * expansion, num_classes)

        init_model(self)

        def ramp_up_lr(lr0, lrT, T):
            rate = (lrT - lr0) / T
            return "lambda t: {'lr': %s + t * %s}" % (lr0, rate)
        if regime == 'normal':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(1e-4),
                 'step_lambda': ramp_up_lr(0.1, 0.1 * scale_lr, 5004 * 5 / scale_lr)},
                {'epoch': 5,  'lr': scale_lr * 1e-1},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': scale_lr * 1e-4}
            ]
        elif regime == 'fast':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(1e-4),
                 'step_lambda': ramp_up_lr(0.1, 0.1 * 4 * scale_lr, 5004 * 4 / (4 * scale_lr))},
                {'epoch': 4,  'lr': 4 * scale_lr * 1e-1},
                {'epoch': 18, 'lr': scale_lr * 1e-1},
                {'epoch': 21, 'lr': scale_lr * 1e-2},
                {'epoch': 35, 'lr': scale_lr * 1e-3},
                {'epoch': 43, 'lr': scale_lr * 1e-4},
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 256},
                {'epoch': 18, 'input_size': 224, 'batch_size': 64},
                {'epoch': 41, 'input_size': 288, 'batch_size': 32},
            ]
        elif 'small' in regime:
            if regime == 'small_half':
                bs_factor = 2
            else:
                bs_factor = 1
            scale_lr *= 4 * bs_factor
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'regularizer': weight_decay_config(1e-4),
                 'momentum': 0.9, 'lr': scale_lr * 1e-1},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': bs_factor * 1e-4}
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 256 * bs_factor},
                {'epoch': 80, 'input_size': 224, 'batch_size': 64 * bs_factor},
            ]
            self.data_eval_regime = [
                {'epoch': 0, 'input_size': 224, 'batch_size': 512 * bs_factor},
            ]
        elif 'qdistil_LARC2' == regime:
            lr_start = 1e-3
            trust_coef = 5e-2
            samples_per_epoch = 131072
            batch_size = 128
            steps_per_epoch = samples_per_epoch//batch_size - 1
            ramp_up_epochs=5
            self.regime = [
                #{'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, #'regularizer': weight_decay_config(1e-4),
                {'epoch': 0, 'optimizer': 'LARC','trust_coefficient':trust_coef, 'clip': False, #,'regularizer': weight_decay_config(1e-4),
                 #'step_lambda': ramp_up_lr(lr_start*1e-2, lr_start * scale_lr, steps_per_epoch * ramp_up_epochs / scale_lr)},
                'lr': scale_lr * lr_start },
                #{'epoch': ramp_up_epochs,  'lr': scale_lr * lr_start},
                 {'epoch': 40, 'lr': scale_lr * lr_start ,'clip': True, 'trust_coefficient':trust_coef / 5},
                 {'epoch': 70, 'lr': scale_lr * lr_start * 1e-1}, #'trust_coefficient':trust_coef / 5},
                # {'epoch': 90, 'lr': scale_lr * lr_start * 1e-2}, #'trust_coefficient':trust_coef / 10}
            ]
        elif 'qdistil_LARC' == regime:
            lr_start = 1e-2
            trust_coef = 1e-1
            samples_per_epoch = 131072
            batch_size = 128
            steps_per_epoch = samples_per_epoch//batch_size - 1
            ramp_up_epochs=5
            self.regime = [
                #{'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, #'regularizer': weight_decay_config(1e-4),
                {'epoch': 0, 'optimizer': 'LARC','trust_coefficient':trust_coef, 'clip': True, #,'regularizer': weight_decay_config(1e-4),
                 #'step_lambda': ramp_up_lr(lr_start*1e-2, lr_start * scale_lr, steps_per_epoch * ramp_up_epochs / scale_lr)},
                'lr': scale_lr * lr_start },
                #{'epoch': ramp_up_epochs,  'lr': scale_lr * lr_start},
                 {'epoch': 1, 'clip': False,'trust_coefficient': trust_coef / 4},
                 {'epoch': 40, 'lr': scale_lr * lr_start * 1e-1 ,'clip': True, 'trust_coefficient':trust_coef / 10},
                 {'epoch': 70, 'lr': scale_lr * lr_start * 1e-2}, #'trust_coefficient':trust_coef / 5},
                # {'epoch': 90, 'lr': scale_lr * lr_start * 1e-2}, #'trust_coefficient':trust_coef / 10}
            ]
        elif 'qdistil_sgd' == regime:
            lr_start = 5e-5
            trust_coef = 5e-2
            samples_per_epoch = 131072
            batch_size = 128
            steps_per_epoch = samples_per_epoch//batch_size - 1
            ramp_up_epochs=5
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.99, #'regularizer': weight_decay_config(1e-4),
                #{'epoch': 0, 'optimizer': 'LARC', 'momentum': 0.9, 'trust_coefficient':trust_coef, 'clip': False, #,'regularizer': weight_decay_config(1e-4),
                 #'step_lambda': ramp_up_lr(lr_start*1e-2, lr_start * scale_lr, steps_per_epoch * ramp_up_epochs / scale_lr)},
                 'lr': lr_start * scale_lr},
                #{'epoch': ramp_up_epochs,  'lr': scale_lr * lr_start},
                 #{'epoch': 40, 'lr': scale_lr * lr_start ,'clip': True, 'trust_coefficient':trust_coef / 10},
                 {'epoch': 70, 'lr': scale_lr * lr_start * 1e-1}, #'trust_coefficient':trust_coef / 5},
                # {'epoch': 90, 'lr': scale_lr * lr_start * 1e-2}, #'trust_coefficient':trust_coef / 10}
            ]

class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes,  stride=1, expansion=1,
                 downsample=None, groups=1, residual_block=None, dropout=0.,absorb_bn=False):
        super(BasicBlock, self).__init__()
        dropout = 0 if dropout is None else dropout
        self.conv1 = conv3x3(inplanes, planes, stride, groups=groups,bias=absorb_bn)
        if not absorb_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, expansion * planes, groups=groups,bias=absorb_bn)
        if not absorb_bn:
            self.bn2 = nn.BatchNorm2d(expansion * planes)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        self.dropout = nn.Dropout(dropout)

        if absorb_bn:
            self.bn1 = lambda x:x
            self.bn2 = lambda x:x

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes,  stride=1, expansion=4, downsample=None, groups=1, residual_block=None, dropout=0.,absorb_bn=False):
        super(Bottleneck, self).__init__()
        dropout = 0 if dropout is None else dropout
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=1, bias=absorb_bn)
        if not absorb_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=stride, groups=groups,bias=absorb_bn)
        if not absorb_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * expansion, kernel_size=1, bias=absorb_bn)
        if not absorb_bn:
            self.bn3 = nn.BatchNorm2d(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.downsample = downsample
        self.residual_block = residual_block
        self.stride = stride
        self.expansion = expansion
        if absorb_bn:
            self.bn1 = lambda x:x
            self.bn2 = lambda x:x
            self.bn3 = lambda x:x

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        if self.residual_block is not None:
            residual = self.residual_block(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()
        #self.flatten = Reshape(-1)
    def _make_layer(self, block, planes, blocks, expansion=1, stride=1, groups=1, residual_block=None, dropout=None, mixup=False,absorb_bn=False):
        downsample = None
        out_planes = planes * expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes,
                          kernel_size=1, stride=stride, bias=absorb_bn), )
            if not absorb_bn:
                downsample.add_module('1' ,nn.BatchNorm2d(planes * expansion))
        if residual_block is not None:
            residual_block = residual_block(out_planes)

        layers = []
        layers.append(block(self.inplanes, planes, stride, expansion=expansion,
                            downsample=downsample, groups=groups, residual_block=residual_block, dropout=dropout,absorb_bn=absorb_bn))
        self.inplanes = planes * expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, expansion=expansion, groups=groups,
                                residual_block=residual_block, dropout=dropout,absorb_bn=absorb_bn))
        if mixup:
            layers.append(MixUp())
        return nn.Sequential(*layers)

    def features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x
        #return x.view(x.size(0), -1)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


class ResNet_imagenet(ResNet):
    def __init__(self, num_classes=1000, inplanes=64,
                 block=Bottleneck, residual_block=None, layers=[3, 4, 23, 3],
                 width=[64, 128, 256, 512], expansion=4, groups=[1, 1, 1, 1],
                 regime='normal', scale_lr=1, checkpoint_segments=0, mixup=False,absorb_bn=False,**kwargs):
        super(ResNet_imagenet, self).__init__()
        self.inplanes = inplanes
        self.conv1 = FirstConv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=absorb_bn)
        if absorb_bn:
            self.bn1 = lambda x:x
        else:
            self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        for i in range(len(layers)):
            layer = self._make_layer(block=block, planes=width[i], blocks=layers[i], expansion=expansion,
                                     stride=1 if i == 0 else 2, residual_block=residual_block, groups=groups[i],
                                     mixup=mixup,absorb_bn=absorb_bn)
            if checkpoint_segments > 0:
                layer_checkpoint_segments = min(checkpoint_segments, layers[i])
                layer = CheckpointModule(layer, layer_checkpoint_segments)
            setattr(self, 'layer%s' % str(i + 1), layer)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten=Reshape(-1)
        self.fc = FC(width[-1] * expansion, num_classes)

        init_model(self)

        if regime == 'normal':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(1e-4),
                 'step_lambda': ramp_up_lr(0.1, 0.1 * scale_lr, 5004 * 5 / scale_lr)},
                {'epoch': 5,  'lr': scale_lr * 1e-1},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': scale_lr * 1e-4}
            ]
        elif regime == 'fast':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'regularizer': weight_decay_config(1e-4),
                 'step_lambda': ramp_up_lr(0.1, 0.1 * 4 * scale_lr, 5004 * 4 / (4 * scale_lr))},
                {'epoch': 4,  'lr': 4 * scale_lr * 1e-1},
                {'epoch': 18, 'lr': scale_lr * 1e-1},
                {'epoch': 21, 'lr': scale_lr * 1e-2},
                {'epoch': 35, 'lr': scale_lr * 1e-3},
                {'epoch': 43, 'lr': scale_lr * 1e-4},
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 256},
                {'epoch': 18, 'input_size': 224, 'batch_size': 64},
                {'epoch': 41, 'input_size': 288, 'batch_size': 32},
            ]
        elif 'small' in regime:
            if regime == 'small_half':
                bs_factor = 2
            else:
                bs_factor = 1
            scale_lr *= 4 * bs_factor
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'regularizer': weight_decay_config(1e-4),
                 'momentum': 0.9, 'lr': scale_lr * 1e-1},
                {'epoch': 30, 'lr': scale_lr * 1e-2},
                {'epoch': 60, 'lr': scale_lr * 1e-3},
                {'epoch': 80, 'lr': bs_factor * 1e-4}
            ]
            self.data_regime = [
                {'epoch': 0, 'input_size': 128, 'batch_size': 256 * bs_factor},
                {'epoch': 80, 'input_size': 224, 'batch_size': 64 * bs_factor},
            ]
            self.data_eval_regime = [
                {'epoch': 0, 'input_size': 224, 'batch_size': 512 * bs_factor},
            ]
        elif 'qdistil1' == regime:
            lr_start = 1e-3
            trust_coef = 5e-2
            samples_per_epoch = 131072
            batch_size = 128
            steps_per_epoch = samples_per_epoch//batch_size - 1
            ramp_up_epochs=5
            self.regime = [
                #{'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, #'regularizer': weight_decay_config(1e-4),
                {'epoch': 0, 'optimizer': 'LARC', 'momentum': 0.9, 'trust_coefficient':trust_coef, 'clip': False, #,'regularizer': weight_decay_config(1e-4),
                 #'step_lambda': ramp_up_lr(lr_start*1e-2, lr_start * scale_lr, steps_per_epoch * ramp_up_epochs / scale_lr)},
                 'lr': scale_lr * lr_start},
                #{'epoch': ramp_up_epochs,  'lr': scale_lr * lr_start},
                 {'epoch': 40, 'lr': scale_lr * lr_start ,'clip': True, 'trust_coefficient':trust_coef / 10},
                 {'epoch': 70, 'lr': scale_lr * lr_start * 1e-1}, #'trust_coefficient':trust_coef / 5},
                # {'epoch': 90, 'lr': scale_lr * lr_start * 1e-2}, #'trust_coefficient':trust_coef / 10}
            ]
        elif 'qdistil_absorbed_bn2' == regime:
            lr_start = 1e-4
            trust_coef = 1e-2
            tot_steps = 50000
            batch_size = 128
            #samples_per_epoch = 131072
            #steps_per_epoch = samples_per_epoch//batch_size - 1
            self.regime = [
                {'step': 0, 'optimizer': 'LARC', 'trust_coefficient':trust_coef, 'clip': True,
                 #'step_lambda': ramp_up_lr(lr_start*1e-2, lr_start * scale_lr, steps_per_epoch * ramp_up_epochs / scale_lr)},
                 'lr': scale_lr * lr_start},
                 #{'epoch': ramp_up_epochs,  'lr': scale_lr * lr_start},
                 {'step': 20000,'clip': False, 'trust_coefficient':trust_coef / 10},
                 {'step': 40000, 'trust_coefficient':trust_coef / 20},

            ]
        elif 'qdistil_fast_sgd' == regime:
            lr_start = 1e-3
            ## reference settings to fix training regime length in update steps
            sampels = 5200
            batch_size = 128
            max_replay = 10
            steps_per_epoch = int(max_replay*(sampels/batch_size))
            tot_steps = steps_per_epoch * 50
            self.quant_freeze_steps = 10000
            ramp_up_steps = 1000 #int(tot_steps * 0.05)
            # samples_per_epoch = 131072
            # self.regime = [
            #     {'step': 0, 'optimizer': 'SGD', 'step_lambda': ramp_up_lr(lr_start*1e-2, lr_start * scale_lr, ramp_up_steps // scale_lr),'momentum' : 0.85},
            #     {'step': ramp_up_steps,'lr' : lr_start * scale_lr},
            #     #{'step': 0, 'optimizer': 'SGD','lr' : lr_start * scale_lr },
            #     {'step': int(tot_steps * 0.25), 'lr' : lr_start * scale_lr * 1e-1},
            #     {'step': int(tot_steps * 0.7) , 'lr' : lr_start * scale_lr * 1e-2},
            # ]
            # self.regime = [
            #     {'step': 0, 'optimizer': 'SGD', 'step_lambda': ramp_up_lr(lr_start*1e-2, lr_start * scale_lr, ramp_up_steps // scale_lr),'momentum' : 0.8},
            #     {'step': ramp_up_steps,'lr' : lr_start * scale_lr},
            #     #{'step': 0, 'optimizer': 'SGD','lr' : lr_start * scale_lr },
            #     {'step': 4633, 'lr' : lr_start * scale_lr * 1e-2},
            #     {'step': 12000 , 'lr' : lr_start * scale_lr * 1e-3},
            # ]
            self.regime = [
                {'step': 0, 'optimizer': 'Adam', 'lr': lr_start * scale_lr, 'beta1':0.8},
                #{'step': ramp_up_steps,'lr' : lr_start * scale_lr},
                #{'step': 0, 'optimizer': 'SGD','lr' : lr_start * scale_lr },
                {'step': 4000, 'lr' : lr_start * scale_lr * 1e-1},
                # release q params at step 10000
                {'optimizer': 'SGD','step': 8000 , 'lr' : lr_start * scale_lr * 1e-2},
            ]
        elif 'qdistil_fast_sgd_hard' == regime:
            lr_start = 1e-3
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = 30000
            self.absorb_bn_step = 20000
            samples = 5200
            batch_size = 128
            max_replay = 10
            ramp_up_steps = 400
            self.regime = [
                #{'step': 0, 'optimizer': 'SGD', 'momentum' : 0.8,
                 {'step': 0, 'optimizer': 'Adam', 'beta1':0.8,
                #'lr' : lr_start * scale_lr},
                #'step_lambda': ramp_up_lr(lr_start * 1e-1, lr_start * scale_lr, ramp_up_steps // scale_lr)},
                'step_lambda': exp_decay_lr(lr_start * scale_lr,lr_start * scale_lr * 1e-3,0,15000)},
                #{'step': ramp_up_steps, 'step_lambda': exp_decay_lr(lr_start * scale_lr,lr_start * scale_lr * 1e-2,ramp_up_steps,8000)},
                #{'step': ramp_up_steps,'lr' : lr_start * scale_lr},
                #{'step': 4000, 'lr' : lr_start * scale_lr * 1e-1},
                # release q params at step 6000
                {'optimizer': 'SGD','step': 16000 , 'lr' : lr_start * scale_lr * 1e-3, 'momentum' : 0.9,'dampning':0.1},
            ]
        elif 'adam_exp_decay_1' == regime:
            lr_start = 1e-4
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = 99999999
            self.absorb_bn_step = 40*400
            samples = 5200
            batch_size = 128
            max_replay = 10
            ramp_up_steps = 400
            self.regime = [
                 {'step': 0, 'optimizer': 'Adam', 'beta1':0.8,
                'step_lambda': exp_decay_lr(lr_start * scale_lr,lr_start * scale_lr * 1e-3,0,60*400)},
                {'optimizer': 'SGD','step': 60*400 , 'lr' : lr_start * scale_lr * 5e-4, 'momentum' : 0.1,'dampning':0.9},
            ]
        elif 'adam_drop_1' == regime:
            lr_start = 7e-4
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = 9999999
            self.absorb_bn_step = 18000
            samples = 5200
            batch_size = 128
            max_replay = 10
            ramp_up_steps = 400
            self.regime = [
                {'step': 0, 'optimizer': 'Adam', 'beta1': 0.9,
                 'step_lambda': lr_drops(lr_start * scale_lr, lr_start * scale_lr * 1e-2, 0, 10000,2)},
                {'optimizer': 'SGD', 'step': 16000, 'lr': lr_start * scale_lr * 5e-3, 'momentum': 0.9,
                 'dampning': 0.1},
            ]
        elif 'adam_cos_sgd_1' == regime:
            lr_start = 1e-3
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = 9999999
            self.absorb_bn_step = 18000
            samples = 5200
            batch_size = 128
            max_replay = 10
            ramp_up_steps = 400
            self.regime = [
                {'step': 0, 'optimizer': 'Adam', 'beta1': 0.9,
                 'step_lambda': cosine_anneal_lr(lr_start * scale_lr, lr_start * scale_lr * 1e-2, 0, 25*400,24)},
                                #lr_drops(lr_start * scale_lr, lr_start * scale_lr * 1e-2, 0, 10000,2)},
                {'optimizer': 'SGD', 'step': 25*400, 'lr': lr_start * scale_lr * 1e-2, 'momentum': 0.1,
                 'dampning': 0.9},
            ]
        elif 'adam_cos_sgd_2' == regime:
            lr_start = 1e-3
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = 9999999
            self.absorb_bn_step = 18000
            samples = 5200
            batch_size = 128
            max_replay = 10
            ramp_up_steps = 400
            self.regime = [
                {'step': 0, 'optimizer': 'Adam', 'beta1': 0.9,
                 'step_lambda': cosine_anneal_lr(lr_start * scale_lr, lr_start * scale_lr * 1e-2, 0, 40 * 400, 39)},
                # lr_drops(lr_start * scale_lr, lr_start * scale_lr * 1e-2, 0, 10000,2)},
                {'optimizer': 'SGD', 'step': 40 * 400, 'lr': max(lr_start * scale_lr * 1e-2,1e-6), 'momentum': 0},
            ]
        elif 'sgd_drop_1' == regime:
            lr_start = 1e-4
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = 9999999
            self.absorb_bn_step = 30*400
            samples = 5200
            batch_size = 128
            max_replay = 10
            ramp_up_steps = 400
            self.regime = [
                {'step': 0, 'optimizer': 'SGD','momentum': 0.9, 'dampning': 0.1,
                 'step_lambda': lr_drops(lr_start * scale_lr, lr_start * scale_lr * 1e-2, 0, 45*400,2)},
                {'optimizer': 'SGD', 'step': 45*400, 'lr': lr_start * scale_lr * 5e-3, 'momentum': 0.1,
                 'dampning': 0.9}
            ]
        elif 'sgd_exp_decay_1' == regime:
            lr_start = 1e-4
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = 9999999
            self.absorb_bn_step = 40*400
            samples = 5200
            batch_size = 128
            max_replay = 10
            ramp_up_steps = 400
            self.regime = [
                {'step': 0, 'optimizer': 'SGD','momentum': 0.9, 'dampning': 0.1,
                 'step_lambda': exp_decay_lr(lr_start * scale_lr, lr_start * scale_lr * 1e-3, 0, 40*400)},
                {'optimizer': 'SGD', 'step': 40*400, 'lr': lr_start * scale_lr * 5e-4, 'momentum': 0.1,
                 'dampning': 0.9}
            ]
        elif 'IBM' == regime:
            lr_start = 0.0015
            lr_end = 1e-6
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = -1
            self.absorb_bn_step = -1
            self.regime_epochs = 110
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'dampning': 0.1,'epoch_lambda': exp_decay_lr(lr_start * scale_lr, lr_end * scale_lr,0,110)},
            ]
        elif 'IBM_short' == regime:
            lr_start = 0.0015
            lr_end = 1e-6
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = -1
            self.absorb_bn_step = -1
            self.regime_epochs = 110
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'dampning': 0.1,
                 'epoch_lambda': exp_decay_lr(lr_start * scale_lr, lr_end * scale_lr, 0, 110,n_drops=5)},
            ]


        elif 'sgd_cos_anneal_1' == regime:
            lr_start = 1e-3
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = 9999999
            self.absorb_bn_step = 400*40

            self.regime = [
                {'step': 0, 'optimizer': 'SGD','momentum': 0.9, 'dampning': 0.1,
                 'step_lambda': cosine_anneal_lr(lr_start * scale_lr, lr_start * scale_lr * 1e-2, 0, 40*400)},
                {'step': 40*400, 'lr': lr_start * scale_lr * 1e-2,
                 'momentum': 0,
                 #'dampning': 0.9
                 }
            ]
        elif 'sgd_cos_anneal_2' == regime:
            lr_start = 1e-3
            step_lambda_epochs = 40
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = -1
            self.absorb_bn_step = 40*400
            self.regime = [
                {'step': 0, 'optimizer': 'SGD','momentum': 0.9, 'dampning': 0.1,
                 'step_lambda': cosine_anneal_lr(lr_start * scale_lr, lr_start * scale_lr * 1e-2, 0, step_lambda_epochs*400,step_lambda_epochs-1)},
                {'step': step_lambda_epochs*400, 'lr': lr_start * scale_lr * 1e-2,
                 'momentum': 0,
                 #'dampning': 0.9
                 }
            ]
        elif 'sgd_cos_anneal_3' == regime:
            lr_start = 1e-3
            step_lambda_epochs = 60
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = -1
            self.absorb_bn_step =-1 # 40*400
            self.regime = [
                {'step': 0, 'optimizer': 'SGD','momentum': 0.9, 'dampning': 0.1,
                 'step_lambda': cosine_anneal_lr(lr_start * scale_lr, lr_start * scale_lr * 1e-3, 0, step_lambda_epochs*400)},
                {'step': step_lambda_epochs*400, 'lr': lr_start * scale_lr * 1e-3,
                 'momentum': 0,
                 #'dampning': 0.9
                 }
            ]
        elif 'sgd_cos_staggerd_1' == regime:
            self.regime_epochs = 120
            self.regime_steps_per_epoch = 400
            #start epoch,epochs,lr modifier
            warmup=(5,1e-8)
            cos_drops=[(95,1e-3)]
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = -1
            self.absorb_bn_step = -1
            self.regime = []
            lr_start = 1e-3 * scale_lr
            epoch_start=0
            if warmup:
                ramp_up_epochs,warmup_scale=warmup
                self.regime += [{'step': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'dampning': 0.1,
                                 'step_lambda': cosine_anneal_lr(lr_start*warmup_scale, lr_start,0, ramp_up_epochs*self.regime_steps_per_epoch)}]
                epoch_start+=ramp_up_epochs
            for epochs,lr_md in cos_drops:
                lr_end=lr_start*lr_md
                epoch_end=epoch_start+epochs
                self.regime +=[{'step': epoch_start * self.regime_steps_per_epoch, 'optimizer': 'SGD','momentum': 0.9, 'dampning': 0.1,
                 'step_lambda': cosine_anneal_lr(lr_start, lr_end, epoch_start*self.regime_steps_per_epoch, epoch_end*self.regime_steps_per_epoch,epochs-1)}]
                lr_start=lr_end
                epoch_start=epoch_end

            self.regime+=[{'step': epoch_start * self.regime_steps_per_epoch, 'optimizer': 'SGD', 'momentum': 0.9, 'dampning': 0.1, 'lr': lr_start}]

        elif 'qdistil_absorbed_bn' == regime:
            lr_start = 1e-2
            trust_coef = 1e-1
            batch_size = 128
            self.regime = [
                {'epoch': 0, 'optimizer': 'LARC', 'trust_coefficient': trust_coef, 'clip': True,
                 # ,'regularizer': weight_decay_config(1e-4),
                 # 'step_lambda': ramp_up_lr(lr_start*1e-2, lr_start * scale_lr, steps_per_epoch * ramp_up_epochs / scale_lr)},
                 'lr': scale_lr * lr_start},
                # {'epoch': ramp_up_epochs,  'lr': scale_lr * lr_start},
                {'epoch': 1, 'clip': False, 'trust_coefficient': trust_coef / 4},
                {'epoch': 40, 'lr': scale_lr * lr_start * 1e-1, 'clip': True, 'trust_coefficient': trust_coef / 10},
                {'epoch': 70, 'lr': scale_lr * lr_start * 1e-2},  # 'trust_coefficient':trust_coef / 5},
            # {'epoch': 90, 'lr': scale_lr * lr_start * 1e-2}, #'trust_coefficient':trust_coef / 10}
            ]
        elif 'quant_ft' == regime:
            lr_start = 1e-4
            trust_coef = 5e-2
            #samples_per_epoch = 131072
            #batch_size = 1024
            #steps_per_epoch = samples_per_epoch//batch_size - 1
            #ramp_up_epochs=5
            self.regime = [
                #{'epoch': 0, 'optimizer': 'SGD', 'momentum': 0.99, 'clip': False,
                 #'step_lambda': ramp_up_lr(lr_start, lr_start * scale_lr, 5004 * 5 / scale_lr)},
                {'epoch': 0, 'lr': scale_lr * lr_start},
                {'epoch': 6, 'lr': scale_lr * lr_start * 1e-1},
                {'epoch': 20, 'lr': scale_lr * lr_start * 1e-2}
            ]


class ResNet_cifar(ResNet):

    def __init__(self, num_classes=10, inplanes=16,
                 block=BasicBlock, depth=18, width=[16, 32, 64],
                 groups=[1, 1, 1], residual_block=None, regime='normal', dropout=None, mixup=False,absorb_bn = False,scale_lr=1.0,**kwargs):
        super(ResNet_cifar, self).__init__()
        self.inplanes = inplanes
        n = int((depth - 2) / 6)
        self.conv1 = FirstConv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        if absorb_bn:
            self.bn1 = lambda x: x
        else:
            self.bn1 = nn.BatchNorm2d(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = lambda x: x

        self.layer1 = self._make_layer(block, width[0], n, groups=groups[
                                       0], residual_block=residual_block, dropout=dropout, mixup=mixup,absorb_bn=absorb_bn)
        self.layer2 = self._make_layer(
            block, width[1], n, stride=2, groups=groups[1], residual_block=residual_block, dropout=dropout, mixup=mixup,absorb_bn=absorb_bn)
        self.layer3 = self._make_layer(
            block, width[2], n, stride=2, groups=groups[2], residual_block=residual_block, dropout=dropout, mixup=mixup,absorb_bn=absorb_bn)
        self.layer4 = lambda x: x
        self.avgpool = nn.AvgPool2d(8)
        self.flatten=Reshape(-1)
        self.fc = FC(width[-1], num_classes)

        init_model(self)
        if regime == 'normal':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9,
                 'regularizer': weight_decay_config(1e-4)},
                {'epoch': 81, 'lr': 1e-2},
                {'epoch': 122, 'lr': 1e-3},
                {'epoch': 164, 'lr': 1e-4}
            ]
        elif regime == 'wide-resnet':
            self.regime = [
                {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1, 'momentum': 0.9,
                 'regularizer': weight_decay_config(5e-4)},
                {'epoch': 60, 'lr': 2e-2},
                {'epoch': 120, 'lr': 4e-3},
                {'epoch': 160, 'lr': 8e-4}
            ]
        elif 'sgd_cos_staggerd_1' == regime:
            self.regime_epochs = 80
            self.regime_steps_per_epoch = 200
            #start epoch,epochs,lr modifier
            warmup=(10,1e-8)
            cos_drops=[(15,1),(40,1e-2)]
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = -1
            self.absorb_bn_step = -1
            self.regime = []
            lr_start = 1e-3 * scale_lr
            epoch_start=0
            if warmup:
                ramp_up_epochs,warmup_scale=warmup
                self.regime += [{'step': 0, 'optimizer': 'SGD', 'momentum': 0.9, 'dampning': 0.1,
                                 'step_lambda': cosine_anneal_lr(lr_start*warmup_scale, lr_start,0, ramp_up_epochs*self.regime_steps_per_epoch)}]
                epoch_start+=ramp_up_epochs
            for epochs,lr_md in cos_drops:
                lr_end=lr_start*lr_md
                epoch_end=epoch_start+epochs
                self.regime +=[{'step': epoch_start * self.regime_steps_per_epoch, 'optimizer': 'SGD','momentum': 0.9, 'dampning': 0.1,
                 'step_lambda': cosine_anneal_lr(lr_start, lr_end, epoch_start*self.regime_steps_per_epoch, epoch_end*self.regime_steps_per_epoch,epochs-1)}]
                lr_start=lr_end
                epoch_start=epoch_end

            self.regime+=[{'step': epoch_start * self.regime_steps_per_epoch, 'optimizer': 'SGD', 'momentum': 0.9, 'dampning': 0.1, 'lr': lr_start}]

        elif 'sgd_cos_1' == regime:
            self.regime_epochs = 80
            steps_per_epoch = 400
            # start epoch,epochs,lr modifier
            cos_drops = [(40, 1e-2)]
            ## reference settings to fix training regime length in update steps
            self.quant_freeze_steps = -1
            self.absorb_bn_step = -1
            self.regime = []
            lr_start = 1e-3 * scale_lr
            epoch_start = 0
            for epochs, lr_md in cos_drops:
                lr_end = lr_start * lr_md
                epoch_end = epoch_start + epochs

                self.regime += [
                    {'step': epoch_start * steps_per_epoch, 'optimizer': 'SGD', 'momentum': 0.9, 'dampning': 0.1,
                     'step_lambda': cosine_anneal_lr(lr_start, lr_end, 0, epochs * steps_per_epoch, epochs - 1)}]

                lr_start = lr_end
                epoch_start = epoch_end
            self.regime += [{'step': epoch_start * steps_per_epoch, 'lr': lr_start}]

def resnet(**config):
    dataset = config.get('dataset', 'imagenet')
    if config.get('quantize', False):
        global FirstConv2d,FC
        from .modules.quantize import QConv2d, QLinear, RangeBN
        activation_numbit = config.get('activations_numbits', _DEFUALT_A_NBITS )
        weights_numbits = config.get('weights_numbits', _DEFUALT_W_NBITS )
        gradient_numbits = config.get('grad_numbits', _DEFUALT_G_NBITS )
        bias_quant = config.get('bias_quant', _DEFAULT_BIAS_QUANT)

        FirstConv2d = config.get('conv1',partial_class(QConv2d, num_bits=activation_numbit, num_bits_weight=weights_numbits,
                                        num_bits_grad=gradient_numbits,bias_quant=bias_quant))
        if FirstConv2d == 'f32':
            FirstConv2d = torch.nn.Conv2d
            print(f'conv1 is f32')
        elif isinstance(FirstConv2d,dict):
            a = FirstConv2d.get('a',activation_numbit)
            w = FirstConv2d.get('w',weights_numbits)
            print(f'conv1 is w{w}a{a}')
            FirstConv2d = partial_class(QConv2d, num_bits=a, num_bits_weight=w,
                                        num_bits_grad=gradient_numbits,bias_quant=bias_quant)

        FC = config.get('fc',partial_class(QLinear,num_bits=activation_numbit,num_bits_weight=weights_numbits,num_bits_grad=gradient_numbits))
        if FC == 'f32':
            FC = torch.nn.Linear
            print(f'fc is fc')
        elif isinstance(FC,dict):
            a = FC.get('a',activation_numbit)
            w = FC.get('w',weights_numbits)
            print(f'fc is w{w}a{a}')
            FC = partial_class(QLinear, num_bits=a, num_bits_weight=w,
                               num_bits_grad=gradient_numbits,bias_quant=bias_quant)

        # replace all conv2d/linear layers to quantized version
        torch.nn.Conv2d = partial_class(QConv2d, num_bits=activation_numbit, num_bits_weight=weights_numbits,
                                        num_bits_grad=gradient_numbits,bias_quant=bias_quant)
        torch.nn.Linear = partial_class(QLinear, num_bits=activation_numbit, num_bits_weight=weights_numbits,
                                        num_bits_grad=gradient_numbits,bias_quant=bias_quant)
    else:
        torch.nn.Conv2d = torch.nn.modules.conv.Conv2d
        torch.nn.Linear = torch.nn.modules.Linear
        FirstConv2d = torch.nn.modules.conv.Conv2d
        FC = torch.nn.modules.Linear



    bn_norm = config.get('bn_norm', None)
    if bn_norm is not None:
        from .modules.lp_norm import L1BatchNorm2d, TopkBatchNorm2d
        if bn_norm == 'L1':
            torch.nn.BatchNorm2d = L1BatchNorm2d
        if bn_norm == 'TopK':
            torch.nn.BatchNorm2d = TopkBatchNorm2d
        if bn_norm == 'range':
            torch.nn.BatchNorm2d = RangeBN

    if 'imagenet' in dataset:
        config.setdefault('num_classes', 1000)
        depth = config.get('depth', 50)
        if depth == 18:
            config.update(dict(block=BasicBlock,
                               layers=[2, 2, 2, 2],
                               expansion=1))
        if depth == 34:
            config.update(dict(block=BasicBlock,
                               layers=[3, 4, 6, 3],
                               expansion=1))
        if depth == 50:
            config.update(dict(layers=[3, 4, 6, 3]))
        if depth == 101:
            config.update(dict(layers=[3, 4, 23, 3]))
        if depth == 152:
            config.update(dict(layers=[3, 8, 36, 3]))
        if depth == 200:
            config.update(dict(layers=[3, 24, 36, 3]))

        if config.get('OTF',False):
            if config['block'] == BasicBlock:
                config.update({'block' : OTFBasicBlock})
            elif config['block'] == Bottleneck:
                config.update({'block' : OTFBottleneck})
            model = OTFResNet_imagenet(**config)
        else:
            model = ResNet_imagenet(**config)
        return model

    elif dataset in ['cifar10', 'mnist_3c','SVHN','stl10']:
        config.setdefault('num_classes', 10)
        config.setdefault('depth', 44)
        return ResNet_cifar(block=BasicBlock, **config)

    elif dataset == 'cifar100':
        config.setdefault('num_classes', 100)
        config.setdefault('depth', 44)
        return ResNet_cifar(block=BasicBlock, **config)


def resnet_se(**config):
    config['residual_block'] = SEBlock
    return resnet(**config)

class Reshape(nn.Module):
    def __init__(self,*args):
        super(Reshape,self).__init__()
        if type(args) == int:
            args = (args,)
        self.shape = args

    def forward(self,x):
        shape = (x.size(0),) + self.shape
        return x.view(*shape)

class PassThrough(nn.Module):
    def __init__(self):
        super(Reshape,self).__init__()

    def forward(self, *input):
        return input