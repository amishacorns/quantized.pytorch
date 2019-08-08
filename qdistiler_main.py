import argparse
import os
import subprocess
import time
import logging
import torch
import shutil
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from utils.mixup import MixUp
import models
from data import get_dataset
from torchvision.transforms import Compose

from preprocess import get_transform,RandomNoise,Cutout,ImgGhosting
from utils.log import setup_logging, ResultsLog, save_checkpoint
from utils.meters import AverageMeter, accuracy
from utils.optim import OptimRegime
from utils.misc import torch_dtypes,CosineSimilarityChannelWiseLoss
from datetime import datetime
from ast import literal_eval
from models.modules.quantize import set_measure_mode,set_bn_is_train,freeze_quant_params,\
    set_global_quantization_method,QuantMeasure,is_bn,overwrite_params,set_quant_mode
_DEFUALT_W_NBITS = 4
_DEFUALT_A_NBITS = 8

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')
###GENERAL
parser.add_argument('--args-from-file', default=None,
                    help='load run arguments from file')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--exp-group', default=None,
                    help='use a shared file to collect a group of experiment results')
parser.add_argument('--print-freq', '-pf', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--ckpt-freq', '-cf', default=10, type=int,
                    metavar='N', help='save checkpoint frequency (default: 10)')
parser.add_argument('--seed', default=123, type=int,
                    help='random seed (default: 123)')
####OP MOD
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--device', default='cuda',
                    help='device assignment ("cpu" or "cuda")')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='device ids assignment (e.g 0 1 2 3')
parser.add_argument('--dtype', default='float',
                    help='type of tensor: ' +
                         ' | '.join(torch_dtypes.keys()) +
                         ' (default: float)')
###MODEL
parser.add_argument('--model', '-a', metavar='MODEL', default='alexnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--teacher', type=str, metavar='FILE',
                    help='path to teacher model checkpoint FILE')
####BN-MOD
parser.add_argument('--absorb-bn-step', default=None, type=int,
                    help='limit training steps')
parser.add_argument('--absorb-bn', action='store_true',
                    help='student model absorbs batchnorm before distillation')
parser.add_argument('--fresh-bn', action='store_true',
                    help='student model absorbs batchnorm running mean and var before distillation but leaves bn layers with affine parameters')
parser.add_argument('--otf', action='store_true',
                    help='use on the fly absorbing batchnorm layers')
###DATA
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--dataset', metavar='DATASET', default='imagenet',
                    help='dataset name or folder')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--dist-set-size', default=None, type=int,
                    help='limit number of examples per class for distilation training (default: None, use entire ds)')
parser.add_argument('--distill-aug', nargs='+', type=str,help='use intermediate layer loss',choices=['cutout','ghost','normal'],default=None)
parser.add_argument('--mixup', action='store_true',
                    help='use training examples mixup')
parser.add_argument('--mixup_rate', default=0.5,
                    help='mixup distribution parameter')
parser.add_argument('--mix-target', action='store_true',
                    help='use target mixup')
###OPT
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--steps-limit', default=None, type=int,
                    help='limit training steps')
parser.add_argument('--steps-per-epoch', default=None, type=int,
                    help='number of steps per epoch, value greater than 0'
                         ' will cause training iterator to sample with replacement')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-r','--overwrite-regime',default=None, help='rewrite regime with external list of dicts "[{},{}...]"')
####MISC
parser.add_argument('--pretrain', action='store_true',
                    help='preform layerwise pretraining session before full network distillation')
parser.add_argument('--train-first-conv', action='store_true',
                    help='allow first conv to train')
parser.add_argument('--use-softmax-scale', action='store_true',
                    help='use trainable temperature parameter')

####Quant
parser.add_argument('--q-method', default='avg',choices=QuantMeasure._QMEASURE_SUPPORTED_METHODS,
                    help='which quantization method to use')
parser.add_argument('--calibration-resample', action='store_true',
                    help='resample calibration dataset examples')
parser.add_argument('--quant-freeze-steps', default=None, type=int,
                    help='number of steps untill releasing qparams')
parser.add_argument('--free-w-range', action='store_true',
                    help='do not freeze weight dynamic range during training')
parser.add_argument('--quant-once', action='store_true',
                    help='debug regime mode, model params are quantized only once before first iteration the rest of the compute is float')
####Loss
parser.add_argument('--order-weighted-loss', action='store_true',
                    help='loss is proportional to the teacher ordering')
parser.add_argument('--ranking-loss', action='store_true',
                    help='use top1 ranking loss')
parser.add_argument('--aux', choices=['mse','kld','cos','smoothl1'],default=None,
                    help='use intermediate layer loss')
parser.add_argument('--loss', default='mse',choices=['mse','kld','smoothl1'],
                    help='specify main loss criterion')
parser.add_argument('--aux-loss-scale',default=1.0,type=float,
                    help='overwrite aux loss scale')
parser.add_argument('--loss-scale',default=1.0,type=float,
                    help='overwrite loss scale')

####Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of distributed processes')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='rank of distributed processes')
parser.add_argument('--dist-init', default='env://', type=str,
                    help='init used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

def main():
    global args, best_prec1, dtype
    best_prec1 = 0
    best_val_loss = 9999.9
    args = parser.parse_args()
    if args.args_from_file:
        import json
        with open(args.args_from_file,'r') as f :
            args_l=json.load(f)
            for key,value in args_l.items():
                if key in ['save','device_id']:
                    continue
                assert key in args, f'loaded argument {key} does not exist in current version'
                setattr(args,key,value)
            print(args)
    dtype = torch_dtypes.get(args.dtype)
    torch.manual_seed(args.seed)
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    distributed = args.local_rank >= 0 or args.world_size > 1
    is_not_master = distributed and args.local_rank > 0
    if distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_init,
                                world_size=args.world_size, rank=args.local_rank)
        args.local_rank = dist.get_rank()
        args.world_size = dist.get_world_size()
        if args.dist_backend == 'mpi':
            # If using MPI, select all visible devices
            args.device_ids = list(range(torch.cuda.device_count()))
        else:
            args.device_ids = [args.local_rank]

    # create model config
    logging.info("creating model %s", args.model)
    model_builder = models.__dict__[args.model]
    if args.dataset in ['imaginet', 'randomnet']:
        model_ds_config='imagenet'
    elif args.dataset in ['cifar10-raw','imagine-cifar10-r44']:
        model_ds_config='cifar10'
    else:
        model_ds_config=args.dataset
    model_config = {'input_size': args.input_size, 'dataset':  model_ds_config}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        conv1,fc = '', ''
        if model_config.get('conv1'):
            conv1 = '_conv1_{}'.format(model_config['conv1'].__str__().strip('\{\}').replace('\'','').replace(': ','').replace(', ',''))
        if model_config.get('fc'):
            fc = '_fc_{}'.format(model_config['fc'].__str__().strip('\{\}').replace('\'','').replace(': ','').replace(', ',''))
        fl_layers = conv1 + fc
        # save all experiments with same setup in the same root dir including calibration checkpoint
        args.save = '{net}{spec}_{gw}w{ga}a{fl}{bn_mod}'.format(
            net=args.model,spec=model_config['depth'],
            gw=model_config['weights_numbits'],
            ga=model_config['activations_numbits'],
            fl=fl_layers,
            bn_mod=('_OTF' if args.otf else ('_absorb_bn' if args.absorb_bn else '_fresh_bn' if args.fresh_bn else '_bn')))
        # specific optimizations are stored per experiment
        regime_name = '_' + model_config.get('regime', '')

        if args.overwrite_regime:
            opt = '_custom'+ regime_name
        elif regime_name == '_':
            opt = '_default'
        else:
            opt = regime_name

        opt += f'_loss-{args.loss}'
        if args.pretrain:
            opt += '_pretrain'
        if args.aux:
            opt += f'_aux-{args.aux}'
        if args.mixup:
            opt += '_mixup'
            if args.mix_target:
                opt += '_w_targets'
        if args.order_weighted_loss:
            opt+='_order_scale'
        if args.ranking_loss:
            opt+='_ranking_loss'
        if args.quant_once:
            opt += '_float_opt'
        elif args.quant_freeze_steps and args.free_w_range:
            opt += '_free_weights'
        if args.use_softmax_scale:
            opt += '_tau'
        if args.dist_set_size:
            opt += f'_cls_lim_{args.dist_set_size}'

    #args.results_dir = os.path.join(os.environ['HOME'],'experiment_results','quantized.pytorch.results')

    save_calibrated_path = os.path.join(args.results_dir,'distiller',args.dataset,args.save)
    if args.exp_group:
        exp_group_path = os.path.join(save_calibrated_path,args.exp_group)

    save_path = os.path.join(save_calibrated_path, time_stamp + opt)
    if not os.path.exists(save_path) and not is_not_master:
        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'),
                  resume=args.resume is not '',
                  dummy=is_not_master)

    results_path = os.path.join(save_path, 'results')
    if not is_not_master:
        results = ResultsLog(
            results_path, title='Training Results - %s' % opt,resume=args.resume,params=args)

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.device and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.set_device(args.device_ids[0])
        cudnn.benchmark = True
    else:
        args.device_ids = None

    logging.info('rewriting calibration resample flag to True')
    args.calibration_resample = True

    teacher = model_builder(**model_config)
    logging.info("created teacher with configuration: %s", model_config)
    logging.debug(teacher)
    logging.info(f"loading teacher checkpoint {args.teacher}")
    teacher_checkpoint = torch.load(args.teacher,map_location='cpu')
    if 'state_dict' in teacher_checkpoint:
        logging.info(f"reference top 1 score {teacher_checkpoint['best_prec1']}")
        teacher_checkpoint = teacher_checkpoint['state_dict']
    teacher.load_state_dict(teacher_checkpoint)
    q_model_config=model_config.copy()
    q_model_config.update({'quantize': True})

    if 'weights_numbits' not in q_model_config:
        q_model_config.update({'weights_numbits':_DEFUALT_W_NBITS})
    if 'activations_numbits' not in q_model_config:
        q_model_config.update({'activations_numbits':_DEFUALT_A_NBITS})
    if args.absorb_bn or args.otf or args.fresh_bn:
        assert not (args.absorb_bn and args.otf or args.absorb_bn and  args.fresh_bn or args.otf and  args.fresh_bn)
        from utils.absorb_bn import search_absorbe_bn
        logging.info('absorbing teacher batch normalization')
        search_absorbe_bn(teacher,verbose=True,remove_bn=not args.fresh_bn,keep_modifiers=args.fresh_bn)
        if not args.fresh_bn:
            model_config.update({'absorb_bn': True})
            teacher_nobn = model_builder(**model_config)
            teacher_nobn.load_state_dict(teacher.state_dict())
            teacher = teacher_nobn
            teacher.eval()
            q_model_config.update({'absorb_bn': True})
    q_model_config.update({'OTF': args.otf})

    logging.info("creating apprentice model with configuration: %s", q_model_config)
    model = model_builder(**q_model_config)
    logging.debug(model)

    regime = literal_eval(args.overwrite_regime) if args.overwrite_regime else getattr(model, 'regime', [{'epoch': 0,
                                                                                                          'optimizer': 'SGD',
                                                                                                          'lr': 0.1,
                                                                                                          'momentum': 0.9,
                                                                                                          'weight_decay': 1e-4}])
    if args.absorb_bn and not args.otf:
        logging.info('freezing remaining batch normalization in student model')
        set_bn_is_train(model,False,logger=logging)
    logging.info(f'overwriting quantization method with {args.q_method}')
    set_global_quantization_method(model,args.q_method)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    # todo mharoush: add distillation specific transforms
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    # if args.distill_aug:
    #     trans = transform['train']
    #     if 'normal' in args.distill_aug:
    #         trans = Compose([trans,RandomNoise('normal', 0.05)])
    #     if 'ghost' in args.distill_aug:
    #         trans = Compose([trans, ImgGhosting()])
    #     if 'cutout' in args.distill_aug:
    #         trans = Compose([trans, Cutout()])
    #     if 'mixup' in args.distill_aug:
    #
    transform.update({'train' : Compose([transform['train'], Cutout()])})
    if args.mixup:
        mixer = MixUp()
        mixer.to(args.device)
    else:
        mixer = None

    train_data = get_dataset(args.dataset, 'train', transform['train'])

    if args.dist_set_size:
        # per samples size
        # ims = []
        # samples_per_class = args.dist_set_size
        # num_classes = len(train_data.classes)
        # for jj in range(num_classes):
        #     tmpp = []
        #     for s in train_data.samples:
        #         if s[1] == jj:
        #             tmpp.append(s)
        #     #print('class', jj, len(tmpp))
        #     els = torch.randperm(len(tmpp))[:samples_per_class].numpy().tolist()
        #     ims += [tmpp[kk] for kk in els]
        #
        #
        # train_data.imgs = ims
        # train_data.samples = train_data.imgs
        train_data=limitDS(train_data,args.dist_set_size)
        logging.info(f'total samples in train dataset {len(train_data.imgs)}')
        if distributed:
            print('verify distributed samples are the same!', train_data.imgs[:100])

    if is_not_master and args.steps_per_epoch:
        ## this ensures that all procesees work on the same sampled sub set data but with different samples per batch
        logging.info('setting different seed per worker for random sampler use in distributed mode')
        torch.manual_seed(args.seed * (1+args.local_rank))

    # todo p3:
    #   2. in-batch augmentation
    if args.steps_per_epoch > 0:
        logging.info(f'total steps per epoch {args.steps_per_epoch}')
        logging.info('setting random sampler with replacement for training dataset')
        sampler = torch.utils.data.RandomSampler(train_data,replacement=True,num_samples=args.steps_per_epoch*args.batch_size)
    else:
        sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_data,sampler=sampler,
        batch_size=args.batch_size, shuffle=(sampler is None),
        num_workers=args.workers, pin_memory=False,drop_last=True)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size*4, shuffle=False,
        num_workers=2, pin_memory=False,drop_last=False)

    repeat = 1
    if args.steps_per_epoch < 0:
        if len(train_loader) / len(val_loader) < 1:
            repeat = len(val_loader) // len(train_loader)

        args.steps_per_epoch=len(train_loader)*repeat
        logging.info(f'total steps per epoch {args.steps_per_epoch}')
    if hasattr(model, 'regime_epochs'):
        args.epochs = model.regime_epochs
    if args.steps_limit:
        args.epochs = min(1+int(args.steps_limit / args.steps_per_epoch),args.epochs)
    logging.info(f'total epochs for training {args.epochs}')

    #pre_train_criterion = nn.MSELoss()
    pre_train_criterion = nn.KLDivLoss(reduction='mean')
    pre_train_criterion.to(args.device)
    loss_scale = args.loss_scale
    aux_loss_scale = args.aux_loss_scale

    if args.loss=='kld':
        criterion = nn.KLDivLoss(reduction='mean')
    elif args.loss == 'smoothl1':
        criterion = nn.SmoothL1Loss()
    else:
        assert(args.loss=='mse')
        criterion = nn.MSELoss()

    if args.aux:
        if args.aux == 'kld' :
            aux = nn.KLDivLoss(reduction='mean')
        elif args.aux == 'cos':
            aux = CosineSimilarityChannelWiseLoss()
        elif args.aux == 'mse':
            aux = nn.MSELoss()
        elif args.aux == 'smoothl1':
            aux = nn.SmoothL1Loss()
        aux.to(args.device)

    else:
        aux = None

    if args.order_weighted_loss:
        loss_scale *= 1000

    criterion.to(args.device, dtype)
    # define loss function (criterion) and optimizer
    # valid_criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    # valid_criterion.to(args.device,dtype)
    valid_criterion = criterion
    teacher.eval()
    teacher.to(args.device, dtype)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
        validate(val_loader, model, valid_criterion, 0,teacher=None)
        return
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            if args.fresh_bn:
                search_absorbe_bn(model, remove_bn=False)
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    elif os.path.isfile(os.path.join(save_calibrated_path,'calibrated_checkpoint.pth.tar')):
        student_checkpoint = torch.load(os.path.join(save_calibrated_path,'calibrated_checkpoint.pth.tar'),'cpu')
        logging.info(f"loading pre-calibrated quantized model from {save_calibrated_path}, reported top 1 score {student_checkpoint['best_prec1']}")
        if args.fresh_bn:
            search_absorbe_bn(model,remove_bn=False)
        model.load_state_dict(student_checkpoint['state_dict'],strict=False)
    else:
        # no checkpoint for model, calibrate quant measure nodes and freeze bn
        logging.info("initializing apprentice model with teacher parameters: %s", q_model_config)
        if args.fresh_bn:
            search_absorbe_bn(model,remove_bn=False)
        model.load_state_dict(teacher.state_dict(), strict=False)
        model.to(args.device, dtype)
        model,_,acc = calibrate(model,args.dataset,transform,val_loader=val_loader,logging=logging,resample=200,sample_per_class=args.dist_set_size)

        student_checkpoint= teacher_checkpoint.copy()
        student_checkpoint.update({'config': q_model_config, 'state_dict': model.state_dict(),
                                   'epoch': 0,'regime':None, 'best_prec1': acc})
        logging.info("saving apprentice checkpoint")
        save_checkpoint(student_checkpoint, path=save_calibrated_path,filename='calibrated_checkpoint')
    #regime[0].update({'use_float_copy':True})
    #model
    if args.use_softmax_scale:
        tau=torch.nn.Parameter(torch.tensor((1.0,), requires_grad=True))
        model.register_parameter('tau',tau)

    if args.quant_freeze_steps is None:
        args.quant_freeze_steps=getattr(model,'quant_freeze_steps',0)
    if args.quant_freeze_steps>-1:
        logging.info(f'quant params will be released at step {args.quant_freeze_steps}')

    if not args.absorb_bn:
        if args.absorb_bn_step is None:
            args.absorb_bn_step = getattr(model, 'absorb_bn_step', -1)
        if args.absorb_bn_step>-1:
            logging.info(f'bn will be absorbed at step {args.absorb_bn_step}')

    optimizer = OptimRegime(model, regime)
    logging.info('start training with regime-\n'+('{}\n'*len(regime)).format(*[p for p in regime]))
    if args.otf:
        logging.info('updating batch norm learnable modifiers in student model')
        state = model.state_dict().copy()
        teacher_params = []
        for k, v in teacher_checkpoint.items():
            if k not in model.state_dict() and ('weight' in k or 'bias' in k):
                teacher_params.append(v)
        from models.modules.quantize import QuantNode
        if not isinstance(model.conv1,QuantNode):
            teacher_params=teacher_params[2:]
        for k,v in model.state_dict().items():
            if 'bn' in k and ('weight' in k or 'bias' in k):
                state[k] = teacher_params.pop(0)
        model.load_state_dict(state)
    model.to(args.device, dtype)
    if args.pretrain:
        ## layerwise training freeze all previous layers first
        #pretrain(model,teacher,train_loader,optimizer,pre_train_criterion,True,4)
        ## fine tune
        pretrain(model, teacher, train_loader, optimizer, pre_train_criterion,False,3, aux = aux,loss_scale = loss_scale)

    if args.quant_once:
        with torch.no_grad():
            overwrite_params(model,logging)
            set_quant_mode(model,False,logging)
            #set_bn_is_train(model, False, logging)
            pass

    for epoch in range(args.start_epoch , args.epochs):
        # train for one epoch
        # if not args.absorb_bn and -1 < args.absorb_bn_step == args.steps_per_epoch*epoch:
        #     logging.info(f'step {epoch*len(train_loader)} absorbing batchnorm layers')
        #     if epoch>0:
        #         #update weights since we are using a master copy in float
        #         overwrite_params(model, logging)
        #     search_absorbe_bn(model)
        #     q_model_config.update({'absorb_bn': True})
        #     no_bn_model = model_builder(**q_model_config)
        #     no_bn_model.load_state_dict(model.state_dict())
        #     model=no_bn_model
        #     model.to(args.device)
        #     if epoch > 0:
        #         model,_,_=calibrate(model,args.dataset,transform,valid_criterion,val_loader=val_loader,logging=logging)
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer,teacher,
            aux=aux,loss_scale=loss_scale,aux_loss_scale=aux_loss_scale,mixer=mixer,quant_freeze_steps=args.quant_freeze_steps,
            dr_weight_freeze=not args.free_w_range,distributed=distributed)

        if (epoch +1) % repeat == 0:
            # evaluate on validation set
            if is_not_master:
                logging.debug('local rank {} done training'.format(args.local_rank))
                continue
            val_loss, val_prec1, val_prec5 = validate(
                val_loader, model, valid_criterion, epoch,teacher=teacher)
            if distributed:
                logging.debug('local rank {} is now saving'.format(args.local_rank))
            timer_save=time.time()
            # remember best prec@1 and save checkpoint
            is_val_best=best_val_loss > val_loss
            if is_val_best:
                best_val_loss=val_loss
                best_loss_epoch=epoch
                best_loss_top1=val_prec1
                best_loss_train=train_loss
            is_best = val_prec1 > best_prec1
            if is_best:
                best_epoch=epoch
                val_best=val_loss
                train_best=train_loss
            best_prec1 = max(val_prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'model': args.model,
                'config': q_model_config,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'regime': regime
            }, is_best, path=save_path,save_freq=args.ckpt_freq)

            logging.info('\n Epoch: {0}\t'
                         'Training Loss {train_loss:.4f} \t'
                         'Training Prec@1 {train_prec1:.3f} \t'
                         'Training Prec@5 {train_prec5:.3f} \t'
                         'Validation Loss {val_loss:.4f} \t'
                         'Validation Prec@1 {val_prec1:.3f} \t'
                         'Validation Prec@5 {val_prec5:.3f} \n'
                         .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                                 train_prec1=train_prec1, val_prec1=val_prec1,
                                 train_prec5=train_prec5, val_prec5=val_prec5))

            results.add(epoch=epoch + 1, train_loss=train_loss, val_loss=val_loss,
                        train_error1=100 - train_prec1, val_error1=100 - val_prec1,
                        train_error5=100 - train_prec5, val_error5=100 - val_prec5)
            results.plot(x='epoch', y=['train_loss', 'val_loss'],
                         legend=['training', 'validation'],
                         title='Loss', ylabel='loss')
            results.plot(x='epoch', y=['train_error1', 'val_error1'],
                         legend=['training', 'validation'],
                         title='Error@1', ylabel='error %')
            results.plot(x='epoch', y=['train_error5', 'val_error5'],
                         legend=['training', 'validation'],
                         title='Error@5', ylabel='error %')
            results.save()
            # clear log file from PIL junk logging(blocking)
            subprocess.call(f"sed -i \"/\bSTREAM\b/d\" {save_path}/log.txt", shell=True)
            if distributed:
                logging.debug('local rank {} done saving. save time: {}'.format(args.local_rank,time.time()-timer_save))
    logging.info('Training-Summary:')
    logging.info(f'best-top1:      {best_prec1:.2f}\tval-loss {val_best:.4f}\ttrain-loss {train_best:.4f}\tepoch {best_epoch}')
    logging.info(f'best-loss-top1: {best_loss_top1:.2f}\tval-loss {best_val_loss:.4f}\ttrain-loss {best_loss_train:.4f}\tepoch {best_loss_epoch}')
    logging.info('regime-\n'+('{}\n'*len(regime)).format(*[p for p in regime]))
    save_path=shutil.move(save_path, save_path + f'_top1_{best_prec1:.2f}_loss_{val_best:.4f}_e{best_epoch}')
    logging.info(f'logdir-{save_path}')
    if args.exp_group:
        logging.info(f'appending experiment result summary to {args.exp_group} experiment')
        exp_summary = ResultsLog(exp_group_path, title='Result Summary, Experiment Group: %s' % args.exp_group,
                                 resume=1,params=None)
        summary={'best_acc_top1': best_prec1, 'best_acc_val': val_best, 'best_acc_train': train_best,
         'best_acc_epoch': best_epoch,
         'best_loss_top1': best_loss_top1, 'best_loss_val': best_val_loss, 'best_loss_train': best_loss_train,
         'best_loss_epoch': best_loss_epoch,'save_path':save_path}
        summary.update(dict(args._get_kwargs()))
        exp_summary.add(**summary)
        exp_summary.save()
        # results.plot(x='epoch', y=['train_loss', 'val_loss'],
        #              legend=['training', 'validation'],
        #              title='Loss', ylabel='loss')
        # os.popen(f'cat \"{save_path}, {regime}, {best_prec1:.2f}, {val_best:.4f}, {train_best:.4f}, {best_epoch:03},'
        #          f' {best_loss_top1:.2f}, {best_val_loss:.4f}, {best_loss_train:.4f}, {best_loss_epoch}\" >> {}.csv')
    os.popen(f'firefox {save_path}/results.html &')

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None,teacher=None,aux=None,aux_start=0,loss_scale = 1.0,aux_loss_scale=1.0,quant_freeze_steps=0,mixer=None,distributed=False):
    if aux:
        model = SubModules(model)
        teacher = SubModules(teacher) if teacher else None

    modules = model._modules
    if distributed:
        model = nn.parallel.DistributedDataParallel(model,
                                                     device_ids=args.device_ids,
                                                     output_device=args.device_ids[0])
        teacher = nn.parallel.DistributedDataParallel(teacher,
                                                     device_ids=args.device_ids,
                                                     output_device=args.device_ids[0]) if teacher else None
        mixer = nn.parallel.DistributedDataParallel(mixer,
                                                     device_ids=args.device_ids,
                                                     output_device=args.device_ids[0]) if mixer else None
    elif args.device_ids and len(args.device_ids) > 1 and not isinstance(model,nn.DataParallel):
        #aux = torch.nn.DataParallel(aux) if aux else None
        #criterion = torch.nn.DataParallel(criterion)
        model = torch.nn.DataParallel(model, args.device_ids)
        teacher = torch.nn.DataParallel(teacher, args.device_ids) if teacher else None
        mixer = torch.nn.DataParallel(mixer, args.device_ids) if mixer else None

    if aux:
        # print('trainable params')
        for r,(k, m) in enumerate(modules.items()):
            for n, p in m.named_parameters():
                if p.requires_grad:
                    # print(f'{k}.{n} shape {p.shape}')
                    if aux_start == -1 and not is_bn(m):
                        logging.debug(f'aux loss will start at {r} for module {k} output')
                        aux_start = r

    regularizer = getattr(model, 'regularization', None)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    aux_loss_mtr = AverageMeter()
    ranking_loss_mtr = AverageMeter()

    end = time.time()
    _once = 1
    if hasattr(data_loader.sampler,'num_samples'):
        steps_per_epoch = data_loader.sampler.num_samples//data_loader.batch_size
    else:
        steps_per_epoch = len(data_loader)

    for i, (inputs, label) in enumerate(data_loader):
        if training:
            steps = epoch * steps_per_epoch + i
            if -1 < quant_freeze_steps < steps and _once:
                logging.info('releasing model quant parameters')
                freeze_quant_params(model, freeze=False, include_param_dyn_range=True, momentum=0.9999,logger=logging)
                _once = 0
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.to(args.device, dtype=dtype)
        label = label.to(args.device)
        if mixer:
            with torch.no_grad():
                inputs = mixer(inputs,[args.mixup_rate,inputs.size(0),True])
        output_ = model(inputs)
        loss = torch.tensor(0.0).to(args.device)
        if teacher:
            with torch.no_grad():
                target_ = teacher(inputs)
                if mixer and args.mix_target:
                    target_ = mixer.mix_target(target_)
            if aux:
                num_outputs_for_aux=(len(output_) - aux_start - 1)
                aux_outputs, aux_targets=output_[aux_start:-1],target_[aux_start:-1]

                for k,(output__,target__) in enumerate(zip(aux_outputs,aux_targets)):
                    if isinstance(aux,nn.KLDivLoss) or isinstance(aux,nn.DataParallel) and isinstance(aux._modules['module'],nn.KLDivLoss):
                        with torch.no_grad():
                            ## divide by temp factor to increase entropy todo register as model learnable param
                            if args.use_softmax_scale:
                                if k == len(output_) - 1:
                                    target__ /= model.tau
                            a_t = F.softmax(target__,-1)
                        a_o = F.log_softmax(output__,-1)
                    else:
                        a_o = output__
                        a_t = target__

                    #mean aux loss over participating layers, deeper layers are weighed less to reduce gradient accumulation
                    #loss = loss + 1/(k-aux_start +1) * aux(output,target)/(len(output_) - aux_start -1)

                    loss = loss + aux(a_o,a_t)* 2*(k-aux_start +1)/(num_outputs_for_aux**2+num_outputs_for_aux)

                #keep last module output for final loss
                output_ = output_[-1]
                target_ = target_[-1]

            ## normal distilation extract target
            if isinstance(criterion, nn.KLDivLoss):
                if args.use_softmax_scale:
                    target_ /= model.tau
                with torch.no_grad():
                    target = F.softmax(target_, -1)
                output = F.log_softmax(output_, -1)
            else:
                target = target_
                output = output_
        else:
            # use real labels as targets
            target = label
            output = output_

        if mixer and args.mix_target:
            with torch.no_grad():
                target = mixer.mix_target(target)

        if args.order_weighted_loss and training:
            with torch.no_grad():
                target,ids = torch.sort(target,descending=True)
                ids_ = torch.cat([s + k * target.size(1) for k, s in enumerate(ids)])
            output_flat = output.flatten()
            output = output_flat[ids_].reshape((target.size(0),target.size(1)))
            # using 1 / ni**2 scaling where ni is the ranking of the element i
            # normalization with pi**2 / 6
            with torch.no_grad():
                #normalizing_sorting_scale=torch.sqrt(0.607927/(torch.arange(1,1001).to(target.device)**2).float()).unsqueeze(0)
                normalizing_sorting_scale = torch.sqrt(
                  1 / (torch.arange(1, 1001,dtype=torch.float)* torch.log(torch.tensor([target.size(1)],dtype=torch.float)))

                ).unsqueeze(0).to(target.device)
                target = torch.mul(target,normalizing_sorting_scale)
            output = torch.mul(output,normalizing_sorting_scale)
        aux_loss_mtr.update(loss.item())
        loss = loss*aux_loss_scale + criterion(output, target) * loss_scale

        if args.ranking_loss and training:
            topk=5
            with torch.no_grad():
                _,ids = torch.sort(target,descending=True)
            output_flat = output.flatten()
            x1,x2=None,None
            for k in range(topk):
                with torch.no_grad():
                    ids_top1= ids[:,k:k+1]
                    ids_rest= ids[:,k+1:]
                    #calculate flat ids for slicing
                    ids_top1_= torch.cat([s + r * target.size(1) for r, s in enumerate(ids_top1)])
                    ids_rest_ = torch.cat([s + r * target.size(1) for r, s in enumerate(ids_rest)])
                if x1 is None:
                    x1 = output_flat[ids_top1_].unsqueeze(1).repeat(1,target.size(1)-1)
                    x2 = output_flat[ids_rest_].reshape((target.size(0),-1))
                else:
                    x1 = torch.cat(x1,output_flat[ids_top1_].unsqueeze(1).repeat(1, target.size(1) - k - 1))
                    x2 = torch.cat(x2,output_flat[ids_rest_].reshape((target.size(0), -1)))
            gt = torch.ones_like(x2)
            ranking_loss = nn.MarginRankingLoss()(x1,x2,gt)
            ranking_loss_mtr.update(ranking_loss.item())
            loss = loss + ranking_loss

        if regularizer is not None:
            loss += regularizer(model)

        # measure accuracy and record loss
        try:
            prec1, prec5 = accuracy(output.detach(), label, topk=(1, 5))
            top1.update(float(prec1), inputs.size(0))
            top5.update(float(prec5), inputs.size(0))
        except:
            pass
        losses.update(float(loss), inputs.size(0))

        if training:
            optimizer.update(epoch, steps)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # elif teacher and i == 0:
        #     compare_activations(model,teacher,inputs[:64])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]  \t{steps}'
                         'Time ({batch_time.avg:.3f}) {batch_time.var:.3f}\t'
                         'Data ({data_time.avg:.3f}) {data_time.var:.3f}\t'
                         'Loss ({loss.avg:.4f}) {loss.var:.3f} \t'
                         'Prec@1 ({top1.avg:.3f}) {top1.var:.3f} \t'
                         'Prec@5 ({top5.avg:.3f} {top5.var:.3f} )'.format(
                epoch, i, len(data_loader),
                phase='TRAINING' if training else 'EVALUATING',
                steps=f'Train steps: {steps}\t' if training else '',
                batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5)+
                         f'\taux_loss {aux_loss_mtr.avg:0.4f}({aux_loss_mtr.var:0.3f})'
                         f'\tranking loss {ranking_loss_mtr.avg:0.4f}({ranking_loss_mtr.var:0.3f})')

    return losses.avg, top1.avg, top5.avg

def train(data_loader, model, criterion, epoch, optimizer,teacher=None,aux=None,aux_start = 0,loss_scale=1.0,aux_loss_scale=1.0,quant_freeze_steps=-1,mixer=None,dr_weight_freeze=True,distributed=False):
    # switch to train mode
    model.train()
    if hasattr(data_loader.sampler, 'num_samples'):
        steps_per_epoch = data_loader.sampler.num_samples // data_loader.batch_size
    else:
        steps_per_epoch = len(data_loader)
    if epoch * steps_per_epoch < quant_freeze_steps or quant_freeze_steps==-1:
        logging.info('freezing quant params{}'.format('' if dr_weight_freeze else ' NOT INCL WEIGHTS'))
        freeze_quant_params(model, freeze=True, include_param_dyn_range=dr_weight_freeze, logger=logging)

    if teacher:
        if args.absorb_bn and not args.otf:
            logging.info('freezing remaining batch normalization')
            set_bn_is_train(model,False)

        if not args.train_first_conv:
            if isinstance(model,nn.DataParallel):
                modules_list= list(model._modules['module']._modules.values())
            else:
                modules_list = list(model._modules.values())
            conv_1_module = modules_list[0]
            bn_1_module = modules_list[1]
            if is_bn(bn_1_module):
                bn_1_module.eval()
            # freeze first layer training
            for p in conv_1_module.parameters():
                p.requires_grad = False

    return forward(data_loader, model, criterion, epoch, training=True, optimizer=optimizer,teacher=teacher,
                   aux=aux,aux_start=aux_start,loss_scale=loss_scale,aux_loss_scale=aux_loss_scale,quant_freeze_steps=quant_freeze_steps,
                   mixer=mixer,distributed=distributed)


def validate(data_loader, model, criterion, epoch,teacher=None):
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        return forward(data_loader, model, criterion, epoch,
                       training=False, optimizer=None,teacher=teacher)


def compare_activations(src,tgt,inputs):
    inp = inputs
    with torch.no_grad():
        if isinstance(src,nn.DataParallel):
            src = src._modules['module']
            tgt = tgt._modules['module']
        for tv, sv in zip(src._modules.values(), tgt._modules.values()):
            m_out = sv(inp)
            t_out = tv(inp)
            print((m_out - t_out).abs().mean(), m_out.shape)
            inp = m_out.squeeze()


def pretrain(model,teacher,data,optimizer,criterion,freeze_prev=True,epochs=5,aux=None,loss_scale=10.0):
    logging.info('freezing batchnorms')
    set_bn_is_train(model,False,logging)
    aux = aux if not freeze_prev else None
    mod = nn.Sequential()
    t_mod = nn.Sequential()

    mod.to(args.device)
    t_mod.to(args.device)
    aux_start=0
    defrost_list = []
    for i,(sv, tv) in enumerate(zip(model._modules.values(), teacher._modules.values())):
        # switch to eval mode
        if freeze_prev:
            for m in mod.modules():
                m.eval()
                for p in m.parameters():
                    if p.requires_grad:
                        p.requires_grad = False
                        defrost_list.append(p)

        t_mod.add_module(str(i), tv)
        t_mod.eval()
        mod.add_module(str(i), sv)
        # 'first and last layers stay frozen'
        if i==0 or i == len(model._modules)-1 or is_bn(sv):
            for p in mod.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                    defrost_list.append(p)
            logging.info(f'skipping module {sv}')
            continue
        if all([p.requires_grad == False for p in mod.parameters(True)]) or len([p for p in sv.parameters(True)]) == 0:
            logging.info(f'skipping module {sv}')
            continue
        else:
            logging.info(f'tuning params for module {sv.__str__()}')

        for e in range(epochs):
            train_loss, _ ,_ = train(data, mod, criterion, 0, optimizer=optimizer, teacher=t_mod,aux=aux,aux_start = aux_start,loss_scale=loss_scale)
            logging.info('\nPre-training Module {} - Epoch: {}\tTraining Loss {train_loss:.5f}'.format(i,e + 1, train_loss=train_loss))
    # defrost model
    # model.train()
    for p in defrost_list:
        p.requires_grad = True

# sequential model with intermidiate output collection, usefull when using aux loss and runing data parallel model
class SubModules(nn.Sequential):
    def __init__(self,model):
        super(SubModules,self).__init__(model._modules)

    def forward(self, input):
        output = []
        for module in self._modules.values():
            input = module(input)
            output.append(input)
        return output

def limitDS(dataset,samples_per_class):
    ims = []
    num_classes = len(dataset.classes)
    samp_reg_per_class=[[]]*num_classes

    for s in dataset.samples:
        samp_reg_per_class[s[1]].append(s)
    for jj in range(num_classes):
        ims += samp_reg_per_class[jj][:samples_per_class]
    dataset.imgs = ims
    dataset.samples = dataset.imgs
    return dataset

def calibrate(model,dataset,transform,calib_criterion=None,resample=200,batch_size=256,workers=4,val_loader=None,sample_per_class=-1,logging=None):
    if logging:
        logging.info("set measure mode")
    # set_bn_is_train(model,False)
    set_measure_mode(model, True, logger=logging)
    if logging:
        logging.info("calibrating model to get quant params")
    calibration_data = get_dataset(dataset, 'train', transform['train'])
    calibration_data = limitDS(calibration_data, sample_per_class)
    if resample>0:
        cal_sampler = torch.utils.data.RandomSampler(calibration_data, replacement=True,
                                                     num_samples=resample * batch_size)
    else:
        cal_sampler = None

    calibration_loader = torch.utils.data.DataLoader(
        calibration_data, sampler=cal_sampler,
        batch_size=batch_size, shuffle=cal_sampler is None,
        num_workers=workers, pin_memory=False, drop_last=cal_sampler is None)
    calib_criterion = calib_criterion or getattr(model, 'criterion', nn.CrossEntropyLoss)()
    calib_criterion.to(args.device,dtype)
    with torch.no_grad():
        losses_avg, top1_avg, top5_avg = forward(calibration_loader, model, calib_criterion, 0, training=False,
                                                 optimizer=None)
    if logging:
        logging.info('Measured float resutls on calibration data:\nLoss {loss:.4f}\t'
                     'Prec@1 {top1:.3f}\t'
                     'Prec@5 {top5:.3f}'.format(loss=losses_avg, top1=top1_avg, top5=top5_avg))
    set_measure_mode(model, False, logger=logging)
    if val_loader:
        if logging:
            logging.info("testing model accuracy")
        losses_avg, top1_avg, top5_avg = validate(val_loader, model, calib_criterion, 0, teacher=None)
        if logging:
            logging.info('Quantized validation results:\nLoss {loss:.4f}\t'
                     'Prec@1 {top1:.3f}\t'
                     'Prec@5 {top5:.3f}'.format(loss=losses_avg, top1=top1_avg, top5=top5_avg))

        return model,losses_avg,top1_avg

    return model
if __name__ == '__main__':
    main()
