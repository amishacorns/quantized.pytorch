import torch
from torch.autograd.function import InplaceFunction, Function
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from utils.absorb_bn import absorb_bn,absorb_bn_step
import pdb
## DEBUG FLAGS
_DEBUG_BN_PLOT = 0

## DEFAULT VALUES:
_DEFAULT_ABSORB_BN_START_STEPS = -1 #1001
_DEFAULT_START_BN_MOMENTUM = 0.1
_DEFAULT_BN_FOLD_ITER = 1
_DEFAULT_ABSORBING_BN_MOMENTUM = 1
_DEFAULT_ABSORBED_BN_W_UPTADE_MOMENTUM = 1

def _mean(p, dim):
    """Computes the mean over all dimensions except dim"""
    if dim is None:
        return p.mean()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).mean(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).mean(dim=0).view(*output_size)
    else:
        return _mean(p.transpose(0, dim), 0).transpose(0, dim)


class UniformQuantize(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None,
                stochastic=False, inplace=False, enforce_true_zero=True, num_chunks=None, out_half=False):

        num_chunks = input.shape[
            0] if num_chunks is None else num_chunks
        if min_value is None or max_value is None:
            B = input.shape[0]
            y = input.view(max(B // num_chunks,1), -1)
        if min_value is None:
            min_value = y.min(-1)[0].mean(-1)  # C
            #min_value = float(input.view(input.size(0), -1).min(-1)[0].mean())
        if max_value is None:
            #max_value = float(input.view(input.size(0), -1).max(-1)[0].mean())
            max_value = y.max(-1)[0].mean(-1)  # C
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        qmin = 0.
        qmax = 2.**num_bits - 1.
        #import pdb; pdb.set_trace()
        scale = (max_value - min_value) / (qmax - qmin)
        if torch.is_tensor(scale):
            scale.clamp_(1e-8)
        else:
            scale = max(scale, 1e-8)

        if enforce_true_zero:
            initial_zero_point = qmin - min_value / scale
            zero_point = 0.
            # make zero exactly represented
            if initial_zero_point < qmin:
                zero_point = qmin
            elif initial_zero_point > qmax:
                zero_point = qmax
            else:
                zero_point = initial_zero_point
            zero_point = int(zero_point)
            output.div_(scale).add_(zero_point)
        else:
            output.add_(-min_value).div_(scale).add_(qmin)

        if ctx.stochastic:
            noise = output.new(output.shape).uniform_(-0.5, 0.5)
            output.add_(noise)
        output.clamp_(qmin, qmax).round_()  # quantize

        if enforce_true_zero:
            output.add_(-zero_point).mul_(scale)  # dequantize
        else:
            output.add_(-qmin).mul_(scale).add_(min_value)  # dequantize
        if out_half and num_bits <= 16:
            output = output.half()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None, None, None


class UniformQuantizeGrad(InplaceFunction):

    @classmethod
    def forward(cls, ctx, input, num_bits=8, min_value=None, max_value=None, stochastic=True, inplace=False):
        ctx.inplace = inplace
        ctx.num_bits = num_bits
        ctx.min_value = min_value
        ctx.max_value = max_value
        ctx.stochastic = stochastic
        return input

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.min_value is None:
            min_value = float(grad_output.min())
            # min_value = float(grad_output.view(
            # grad_output.size(0), -1).min(-1)[0].mean())
        else:
            min_value = ctx.min_value
        if ctx.max_value is None:
            max_value = float(grad_output.max())
            # max_value = float(grad_output.view(
            # grad_output.size(0), -1).max(-1)[0].mean())
        else:
            max_value = ctx.max_value
        grad_input = UniformQuantize().apply(grad_output, ctx.num_bits,
                                             min_value, max_value, ctx.stochastic, ctx.inplace)
        return grad_input, None, None, None, None, None


def conv2d_biprec(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, num_bits_grad=None):
    out1 = F.conv2d(input.detach(), weight, bias,
                    stride, padding, dilation, groups)
    out2 = F.conv2d(input, weight.detach(), bias.detach() if bias is not None else None,
                    stride, padding, dilation, groups)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


def linear_biprec(input, weight, bias=None, num_bits_grad=None):
    out1 = F.linear(input.detach(), weight, bias)
    out2 = F.linear(input, weight.detach(), bias.detach()
                    if bias is not None else None)
    out2 = quantize_grad(out2, num_bits=num_bits_grad)
    return out1 + out2 - out1.detach()


def quantize(x, num_bits=8, min_value=None, max_value=None, num_chunks=None, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, num_bits, min_value, max_value, num_chunks, stochastic, inplace)


def quantize_grad(x, num_bits=8, min_value=None, max_value=None, stochastic=True, inplace=False):
    return UniformQuantizeGrad().apply(x, num_bits, min_value, max_value, stochastic, inplace)


# This class can be used to implement measure mode for all Qmodules via multiple inheritance
# example:
# class Qclass(InheritFrom_nn.Module,QuantNode):
#     def __init__(*args,**kwargs):
#       InheritFrom_nn.Module.__init__(self,*args,**kwargs)
#       QuantNode.__init__(self)
#
# now Qclass has set_measure_mode method and enable_quant attribute.
# Note that forward method should now use quant_enabled to allow for normal forward when it is set to false

class QuantNode():
    def __init__(self):
        self.enable_quant = True
        self.freeze_param_dyn_range = False

    def set_measure_mode(self,measure,momentum=None):
        self.enable_quant = not measure
        if momentum and isinstance(self,QuantMeasure):
            self.momentum = momentum
        else:
            if isinstance(self,nn.Module):
                for q in self._modules.values():
                    if isinstance(q,QuantNode):
                        q.set_measure_mode(measure,momentum=momentum)

    def overwrite_params(self,logging=None):
        if isinstance(self,nn.Module):
            for q in self._modules.values():
                if isinstance(q,QuantNode):
                    q.overwrite_params(logging)

class QuantMeasure(nn.Module,QuantNode):
    """docstring for QuantMeasure."""
    _QMEASURE_SUPPORTED_METHODS = ['avg', 'aciq']

    def __init__(self, num_bits=8, momentum=None,method='avg'):
        super(QuantMeasure, self).__init__()
        QuantNode.__init__(self)
        assert method in QuantMeasure._QMEASURE_SUPPORTED_METHODS
        self.register_buffer('running_min', torch.zeros(1))
        self.register_buffer('running_max', torch.zeros(1))
        self.register_buffer('num_measurements', torch.zeros(1))
        self.register_buffer('running_var', torch.ones(1))
        self.register_buffer('running_mean', torch.zeros(1))
        self.momentum = momentum
        self.num_bits = num_bits
        self.method = method
        self.laplace_alpha = {2: 2.83, 3: 3.89, 4: 5.03, 5: 6.2, 6: 7.41, 7: 8.64, 8: 9.89}.get(num_bits)

    def _momentum_update_stat(self,new_value,running_stat,momentum=None):
        momentum = momentum or self.momentum or self.num_measurements/(self.num_measurements+1)
        running_stat.mul_(momentum).add_(
                new_value * (1 - momentum))

    def forward(self, input):
        # todo
        input_ = input.detach()

        if self.training:
            min_value = input_.view(
                input_.size(0), -1).min(-1)[0].mean()
            self._momentum_update_stat(min_value,self.running_min)
            max_value = input_.view(
                input_.size(0), -1).max(-1)[0].mean()
            self._momentum_update_stat(max_value,self.running_max)
            mean = input_.mean()
            std = input_.std(unbiased=True)
            self._momentum_update_stat(mean, self.running_mean)
            self._momentum_update_stat(std, self.running_var)
            self.num_measurements += 1

            if self.method == 'aciq':
                min_value, max_value = self._get_aciq_range(std,mean,min_value,max_value)

        else:
            if self.method == 'aciq':
                min_value, max_value = self._get_aciq_range(self.running_var,self.running_mean,self.running_min,self.running_max)
            else:
                min_value = self.running_min
                max_value = self.running_max

        if self.enable_quant:
            return quantize(input, self.num_bits, min_value=float(min_value), max_value=float(max_value))
        else:
            return input

    def _get_measured_range(self):
        return float(self.running_min), float(self.running_max)

    def _get_aciq_range(self,std,mean,tmin,tmax):
        with torch.no_grad():
            std += 1e-8
            assert self.laplace_alpha, 'aciq not supported for module num bits'
            clip_val = std * self.laplace_alpha
            assert clip_val > 0, 'invalid clip value!'
            max_range = tmax - tmin
            clip_val = min(max_range / 2,clip_val)

            min_value = max(tmin, mean - clip_val)
            max_value = min(tmax, mean + clip_val)

        return min_value,max_value


class QConv2d(nn.Conv2d,QuantNode):
    """docstring for QConv2d."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None, biprecision=False,bias_quant=True,per_channel=True):
        super(QConv2d,self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        QuantNode.__init__(self)

        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits)
        self.biprecision = biprecision
        self.bias_quant = bias_quant and bias
        self.per_channel = per_channel

        if self.per_channel:
            n_channels = self.weight.size(0)
            dim = self.weight.dim()
            self.scale_shape = (n_channels,) + (1,) * (dim - 1)
            self.register_buffer('weight_min', self.weight.flatten(1).min(-1)[0].view(self.scale_shape))
            self.register_buffer('weight_max', self.weight.flatten(1).max(-1)[0].view(self.scale_shape))
        else:
            self.register_buffer('weight_min', self.weight.min())
            self.register_buffer('weight_max', self.weight.max())
        if self.bias_quant:
            self.register_buffer('bias_min', self.bias.min())
            self.register_buffer('bias_max', self.bias.max())

    def overwrite_params(self,logging=None):
        sd = self.state_dict()
        if logging:
            logging.debug(f'quantizing parameters for {super(QConv2d,self).__str__()}')
        sd.update({'weight':quantize(self.weight, num_bits=self.num_bits_weight,
                 min_value=self.weight_min,
                 max_value=self.weight_max)})
        if self.bias_quant:
            sd.update({'bias': quantize(self.bias, min_value=self.bias_min, max_value=self.bias_max,
                             num_bits=self.num_bits_weight)})
        self.load_state_dict(sd)

    def forward(self, input):
        input_ = self.quantize_input(input)
        if self.enable_quant:
            if not self.freeze_param_dyn_range:
                if self.per_channel:
                    n_channels = self.weight.size(0)
                    dim = self.weight.dim()
                    scale_shape = (n_channels,) + (1,) * (dim-1)
                    self.weight_min = self.weight.flatten(1).min(-1)[0].view(scale_shape)
                    self.weight_max = self.weight.flatten(1).max(-1)[0].view(scale_shape)
                else:
                    self.weight_min = float(self.weight.min())
                    self.weight_max = float(self.weight.max())

                if self.bias is not None:
                    self.bias_min = self.bias.min()
                    self.bias_max = self.bias.max()

            qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                               min_value=self.weight_min,
                               max_value=self.weight_max)

            if self.bias_quant:
                qbias = quantize(self.bias, min_value=self.bias_min, max_value=self.bias_max,
                                 num_bits=self.num_bits_weight)
            else:
                qbias = self.bias

            if not self.biprecision or self.num_bits_grad is None:
                output = F.conv2d(input_, qweight, qbias, self.stride,
                                  self.padding, self.dilation, self.groups)
                if self.num_bits_grad is not None:
                    output = quantize_grad(output, num_bits=self.num_bits_grad)
            else:
                output = conv2d_biprec(input_, qweight, qbias, self.stride,
                                       self.padding, self.dilation, self.groups, num_bits_grad=self.num_bits_grad)
        else:
            output = F.conv2d(input, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

        return output


class QLinear(nn.Linear,QuantNode):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, num_bits=8, num_bits_weight=None, num_bits_grad=None, biprecision=False,bias_quant= True,per_channel = True):
        super(QLinear,self).__init__(in_features, out_features, bias)
        QuantNode.__init__(self)
        self.num_bits = num_bits
        self.num_bits_weight = num_bits_weight or num_bits
        self.num_bits_grad = num_bits_grad
        self.biprecision = biprecision
        self.quantize_input = QuantMeasure(self.num_bits)
        self.bias_quant = bias_quant and bias
        self.per_channel = per_channel

        if self.per_channel:
            n_channels = self.weight.size(0)
            dim = self.weight.dim()
            self.scale_shape = (n_channels,) + (1,) * (dim - 1)
            self.register_buffer('weight_min', self.weight.flatten(1).min(-1)[0].view(self.scale_shape))
            self.register_buffer('weight_max', self.weight.flatten(1).max(-1)[0].view(self.scale_shape))
        else:
            self.register_buffer('weight_min', self.weight.min())
            self.register_buffer('weight_max', self.weight.max())

        if self.bias_quant:
            self.register_buffer('bias_min', self.bias.min())
            self.register_buffer('bias_max', self.bias.max())

    ##todo refactor geter with overwrite flag+qparam update flag
    def overwrite_params(self,logging=None):
        sd = self.state_dict()
        if logging:
            logging.debug(f'quantizing parameters for {super(QLinear,self).__str__()}')
        sd.update({'weight':quantize(self.weight, num_bits=self.num_bits_weight,
                 min_value=self.weight_min,
                 max_value=self.weight_max)})
        if self.bias_quant:
            sd.update({'bias': quantize(self.bias, min_value=self.bias_min, max_value=self.bias_max,
                             num_bits=self.num_bits_weight)})
        self.load_state_dict(sd)

    def forward(self, input):
        input_ = self.quantize_input(input)
        if self.enable_quant:
            if not self.freeze_param_dyn_range:
                if self.per_channel:
                    self.weight_min = self.weight.flatten(1).min(-1)[0].view(self.scale_shape)
                    self.weight_max = self.weight.flatten(1).max(-1)[0].view(self.scale_shape)
                else:
                    self.weight_min = float(self.weight.min())
                    self.weight_max = float(self.weight.max())

                if self.bias is not None:
                    self.bias_min = self.bias.min()
                    self.bias_max = self.bias.max()

            qweight = quantize(self.weight, num_bits=self.num_bits_weight,
                               min_value=self.weight_min,
                               max_value=self.weight_max)

            if self.bias_quant:
                qbias = quantize(self.bias, min_value=self.bias_min, max_value=self.bias_max,
                                 num_bits=self.num_bits_weight)
            else:
                qbias = self.bias

            if not self.biprecision or self.num_bits_grad is None:
                output = F.linear(input_, qweight, qbias)
                if self.num_bits_grad is not None:
                    output = quantize_grad(output, num_bits=self.num_bits_grad)
            else:
                output = linear_biprec(input_, qweight, qbias, self.num_bits_grad)
        else:
            output = F.linear(input, self.weight, self.bias)

        return output


class RangeBN(nn.Module):
    # this is normalized RangeBN

    def __init__(self, num_features, dim=1, momentum=0.1, affine=True, num_chunks=16, eps=1e-5, num_bits=8, num_bits_grad=8):
        super(RangeBN, self).__init__()
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.zeros(num_features))

        self.momentum = momentum
        self.dim = dim
        if affine:
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight = nn.Parameter(torch.Tensor(num_features))
        self.num_bits = num_bits
        self.num_bits_grad = num_bits_grad
        self.quantize_input = QuantMeasure(self.num_bits)
        self.eps = eps
        self.num_chunks = num_chunks
        self.reset_params()

    def reset_params(self):
        if self.weight is not None:
            self.weight.data.uniform_()
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        x = self.quantize_input(x)
        if x.dim() == 2:  # 1d
            x = x.unsqueeze(-1,).unsqueeze(-1)

        if self.training:
            B, C, H, W = x.shape
            y = x.transpose(0, 1).contiguous()  # C x B x H x W
            y = y.view(C, self.num_chunks, B * H * W // self.num_chunks)
            mean_max = y.max(-1)[0].mean(-1)  # C
            mean_min = y.min(-1)[0].mean(-1)  # C
            mean = y.view(C, -1).mean(-1)  # C
            scale_fix = (0.5 * 0.35) * (1 + (math.pi * math.log(4)) **
                                        0.5) / ((2 * math.log(y.size(-1))) ** 0.5)

            scale = 1 / ((mean_max - mean_min) * scale_fix + self.eps)

            self.running_mean.detach().mul_(self.momentum).add_(
                mean * (1 - self.momentum))

            self.running_var.detach().mul_(self.momentum).add_(
                scale * (1 - self.momentum))
        else:
            mean = self.running_mean
            scale = self.running_var
        scale = quantize(scale, num_bits=self.num_bits, min_value=float(
            scale.min()), max_value=float(scale.max()))
        out = (x - mean.view(1, mean.size(0), 1, 1)) * \
            scale.view(1, scale.size(0), 1, 1)

        if self.weight is not None:
            qweight = quantize(self.weight, num_bits=self.num_bits,
                               min_value=float(self.weight.min()),
                               max_value=float(self.weight.max()))
            out = out * qweight.view(1, qweight.size(0), 1, 1)

        if self.bias is not None:
            qbias = quantize(self.bias, num_bits=self.num_bits)
            out = out + qbias.view(1, qbias.size(0), 1, 1)
        if self.num_bits_grad is not None:
            out = quantize_grad(out, num_bits=self.num_bits_grad)

        if out.size(3) == 1 and out.size(2) == 1:
            out = out.squeeze(-1).squeeze(-1)
        return out


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_quant(m):
    #todo fix when Q modules are all subclasses of QNode
    return isinstance(m, QuantNode) or isinstance(m, QLinear) or isinstance(m, QConv2d)


def recursive_apply(model,func,*args):
    for m in model.children():
        func(m,*args)
        recursive_apply(m,func,*args)


def set_bn_is_train(model,train,logger=None):
    def func(m,*args):
        if is_bn(m):
            if logger:
                logger.debug('{} set to {}'.format(m,'eval' if train else 'train'))
            m.train(train)

    recursive_apply(model,func)


def set_measure_mode(model,measure,momentum=None,logger=None):
    def func(m,*args):
        if is_bn(m):
            if logger:
                logger.debug('{} set to {}'.format(m,'eval' if measure else 'train'))
            m.train(not measure)
        elif is_quant(m):
            if logger:
                logger.debug('{} set to {}'.format(m, 'float' if measure else 'quant'))
            m.set_measure_mode(measure,momentum=momentum)

    recursive_apply(model, func)


def set_quant_mode(model,quant,logger=None):
    def func(m,*args):
        if is_quant(m):
            if logger:
                logger.debug('{} set to {}'.format(m, 'float' if not quant else 'quant'))
            m.enable_quant=quant

    recursive_apply(model, func)


def overwrite_params(model,logger = None):
    def func(m,*args):
        if is_quant(m):
            m.overwrite_params(logger)

    recursive_apply(model, func)


def freeze_quant_params(model,freeze=True,include_param_dyn_range=True,momentum='same',logger = None):
    def func(m,*args):
        if isinstance(m,QuantMeasure):
            if logger:
                logger.debug('{} set to {}'.format(m, 'eval' if freeze else 'train'))
            m.train(not freeze)
            if momentum!= 'same':
                if logger:
                    logger.debug('setting momentum to {}'.format(momentum))
                m.momentum=momentum
        if include_param_dyn_range and isinstance(m,QuantNode):
            m.freeze_param_dyn_rang = freeze

    recursive_apply(model, func)


def distill_set_train(model,train):
    model.train(train)
    if train:
        freeze_quant_params(model)
        set_bn_is_train(model, False)


def set_global_quantization_method(model,method='aciq',logger = None):
    assert method in QuantMeasure._QMEASURE_SUPPORTED_METHODS
    def func(m,*args):
        if isinstance(m,QuantMeasure):
            if logger:
                logger.debug('{} set to {}'.format(m, method))
            m.method = method

    recursive_apply(model, func)

# EXPERIMENTAL use this method to generate classes that can fold batchnorms on the fly
def get_bn_folding_module(base_module,bn_module,
                          start_folding_steps=_DEFAULT_ABSORB_BN_START_STEPS,
                          bn_momentum_start_period = _DEFAULT_START_BN_MOMENTUM,
                          fold_iter=_DEFAULT_BN_FOLD_ITER,
                          bn_momentum_post_start =_DEFAULT_ABSORBING_BN_MOMENTUM,
                          lr=_DEFAULT_ABSORBED_BN_W_UPTADE_MOMENTUM):

    class QFold(base_module):
        def __init__(self,*args,**kwargs):
            super(QFold,self).__init__(*args,**kwargs)
            if isinstance(self,nn.Linear):
                out_channels = self.out_features
            else:
                out_channels = self.out_channels
            if hasattr(self,'bias'):
                self.bias.requires_grad = False

            self.absorb_iter = fold_iter
            self.start_steps = start_folding_steps
            self.bn_momentum_post_start = bn_momentum_post_start
            self.bn = bn_module(out_channels,momentum = bn_momentum_start_period if self.start_steps > 0 else self.bn_momentum_post_start
                                ,affine=True,eps=1e-5)
            self._weight_update_lr = lr

        def forward(self,input):
            if -1 < self.bn.num_batches_tracked < self.start_steps :
                out = super(QFold, self).forward(input)
                out = self.bn(out)
                return out
            elif -1 > self.start_steps:
                print('starting bn folding for Qfold layer with shape {}'.format(self.weight.size))
                self.start_steps = -1
                self.bn.momentum = self.bn_momentum_post_start

            if self.training and self.enable_quant:
                with torch.no_grad():
                    assert self.bn.training
                    self.enable_quant = False
                    out_c = super(QFold,self).forward(input)
                    out_ = self.bn(out_c)
                    self.enable_quant = True

                if self.bn.num_batches_tracked >= self.absorb_iter:
                    absorb_bn_step(self, self.bn, remove_bn=False,keep_modifiers=True,lr=self._weight_update_lr)

            out = super(QFold, self).forward(input)
            out = self.bn(out)

            if _DEBUG_BN_PLOT:
                plt.hist(out.detach().cpu().flatten(), 500, label='quant absorbed', alpha=0.3)
                plt.hist(out_c.detach().cpu().flatten(), 500, label='float pre-bn absorbed', alpha=0.3)
                plt.hist(out_.detach().cpu().flatten(), 500, label='float post-bn', alpha=0.3)
                plt.title(self.weight.shape.__str__())
                plt.legend(loc='upper right')
                plt.show()

            return out

    return QFold