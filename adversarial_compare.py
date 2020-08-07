import torch as th
import torchvision as tv
from matplotlib import pyplot as plt
import models
from data import get_dataset,limit_ds
from preprocess import get_transform
import os
import tqdm
import logging
from utils.log import setup_logging
from utils.absorb_bn import get_bn_params
from dataclasses import dataclass
import inspect
from utils.misc import Recorder
from utils.meters import MeterDict, OnlineMeter,AverageMeter,accuracy
from functools import partial
@dataclass
class Settings:
    def __repr__(self):
        return str(self.__class__.__name__) + str({attr:getattr(self,attr) for attr in self.__dir__()
                                                   if type(attr) == str and not attr.startswith('__')})

    def __init__(self,
                 model:str,
                 model_cfg: dict,
                 batch_size: int = 1000,
                 recompute: bool = False,
                 augment_measure: bool = True,
                 augment_test :bool = False,
                 device: str = 'cuda',
                 dataset: str = f'cats_vs_dogs',
                 ckt_path: str = '/home/mharoush/myprojects/convNet.pytorch/results/r18_cats_N_dogs/checkpoint.pth.tar',
                 collector_device : str = 'same', #use cpu or cuda:# if gpu runs OOM
                 limit_test : int = None,
                 limit_measure : int = None,
                 test_split : str = 'val',
                 num_classes=2,
                 right_sided_fisher_pvalue = True,
                 transform_dataset = None
                 ):

        arg_names, _,_, local_vars= inspect.getargvalues(inspect.currentframe())
        for name in arg_names[1:]:
            setattr(self,name,local_vars[name])

def Gaussian_KL(mu1, var1, mu2, var2, epsilon=1e-5):
    var1 = var1.clamp(min=epsilon)
    var2 = var2.clamp(min=epsilon)
    return 1 / 2 * (-1 + th.log(var2 / var1) + (var1 + (mu1 - mu2).pow(2)) / var2)


def Gaussian_sym_KL(mu1, sigma1, mu2, sigma2, epsilon=1e-5):
    return 0.5 * (Gaussian_KL(mu1, sigma1, mu2, sigma2, epsilon) + Gaussian_KL(mu2, sigma2, mu1, sigma1, epsilon))


'''
calculate statistics loss
@inputs:
ref_stat_dict - the reference statistics we want to compare against typically we just pass the model state_dict
stat_dict - stats per layer in form {key,(mean,var)}
running_dict - stats per layer in form {key,(running_sum,running_square_sum,num_samples)}
raw_act_dict - calc stats directly from activations {key,act}
'''


def calc_stats_loss(ref_stat_dict=None, stat_dict=None, running_dict=None, raw_act_dict=None, mode='sym', epsilon=1e-8,
                    pre_bn=True, reduce=True):
    # used to aggregate running stats from multiple devices
    def _get_stats_from_running_dict():
        batch_statistics = {}
        for k, v in running_dict.items():
            mean = v[0] / v[2]
            ## var = E(x^2)-(EX)^2: sum_p2/n -sum/n
            var = v[1] / v[2] - mean.pow(2)
            batch_statistics[k] = (mean, var)
        running_dict.clear()
        return batch_statistics

    def _get_stats_from_acts_dict():
        batch_statistics = {}
        for k, act in raw_act_dict.items():
            mean = act.mean((0, 2, 3))
            var = act.var((0, 2, 3))
            batch_statistics[k] = (mean, var)
        return batch_statistics

    ## target statistics provided externally in mu,var form
    if stat_dict:
        batch_statistics = stat_dict
    ## compute mu and var from a running moment accumolation (statistics collected from multi-gpu)
    elif running_dict:
        batch_statistics = _get_stats_from_running_dict()
    # compute mu and var directly from reference activations
    elif raw_act_dict:
        batch_statistics = _get_stats_from_acts_dict()
    else:
        assert 0

    if pre_bn:
        target_mean_key, target_var_key = 'running_mean', 'running_var'
    else:
        target_mean_key, target_var_key = 'bias', 'weight'

    if mode == 'mse':
        calc_stats = lambda m1, m2, v1, v2: m1.sub(m2).pow(2) + v1.sub(v2).pow(2)
    elif mode == 'l1':
        calc_stats = lambda m1, m2, v1, v2: m1.sub(m2).abs() + v1.sub(v2).abs()
    elif mode == 'exp':
        calc_stats = lambda m1, m2, v1, v2: th.exp(m1.sub(m2).abs() + v1.sub(v2).abs())
    elif mode == 'kl':
        calc_stats = lambda m1, m2, v1, v2: Gaussian_KL(m2, v2, m1, v1, epsilon)
    elif mode == 'sym':
        calc_stats = lambda m1, m2, v1, v2: Gaussian_sym_KL(m1, v1, m2, v2, epsilon)
    else:
        assert 0

    # calculate states per layer key in stats dict
    loss_stat = {}
    for i, (k, (m, v)) in enumerate(batch_statistics.items()):
        # collect reference statistics from dictionary
        if ref_stat_dict and k in ref_stat_dict:
                #statistics are in a param dictionary
            ref_dict = ref_stat_dict[k]
            if target_mean_key in ref_dict:
                m_ref, v_ref = ref_dict[target_mean_key], ref_dict[target_var_key]
            elif 'mean:0' in ref_dict:
                m_ref, v_ref = ref_dict[ f'mean:0'], th.diag(ref_dict[ f'cov:0'])
            else:
                assert 0, 'unsuported reference stat dict structure'
        else:
            # assume normal distribution reference
            m_ref, v_ref = th.zeros_like(m), th.ones_like(v)
        m_ref, v_ref = m_ref.to(m.device), v_ref.to(v.device)

        moments_distance_ = calc_stats(m_ref, m, v_ref, v)

        if reduce:
            moments_distance = moments_distance_.mean()
        else:
            moments_distance = moments_distance_

        # if verbose > 0:
        #     with th.no_grad():
        #         zero_sigma = (v < 1e-5).sum()
        #         if zero_sigma > 0 or moments_distance > 5*loss_stat/i:
        #             print(f'high divergence in layer {k}: {moments_distance}'
        #                   f'\nmu:{m.mean():0.4f}<s:r>{m_ref.mean():0.4f}\tsigma:{v.mean():0.4f}<s:r>{v_ref.mean():0.4f}\tsmall sigmas:{zero_sigma}/{len(v)}')
        #
        loss_stat[k] = moments_distance

    if reduce:
        ret_val = 0
        for k, loss in loss_stat:
            ret_val = ret_val + loss
        ret_val = ret_val / len(batch_statistics)
        return ret_val

    return loss_stat

# todo broken for now
def plot(clean_act, fgsm_act, layer_key, reference_stats=None, nbins=256, max_ratio=True, mode='sym',
         rank_by_stats_loss=False):
    if rank_by_stats_loss:
        fgsm_distance = \
        calc_stats_loss(ref_stat_dict=reference_stats, raw_act_dict={layer_key: fgsm_act}, reduce=False, epsilon=1e-8,
                        mode=mode)[layer_key]
        clean_distance = \
        calc_stats_loss(ref_stat_dict=reference_stats, raw_act_dict={layer_key: clean_act}, reduce=False, epsilon=1e-8,
                        mode=mode)[layer_key]
        if max_ratio:
            # normalized ratio to make sure we detect channles that are stable for clean data
            ratio, ids = (fgsm_distance / clean_distance).sort()
        else:
            div, ids = fgsm_distance.sort()

    max_channels = 12
    ncols = 4
    fig, axs = plt.subplots(nrows=max(max_channels // ncols, 1), ncols=ncols)
    for i, channel_id in enumerate(ids[-max_channels:]):
        ax = axs[i // ncols, i % ncols]
        # plt.figure()
        for e, acts in enumerate([clean_act, fgsm_act]):
            ax.hist(acts[:, i].transpose(1, 0).reshape(-1).detach().numpy(), nbins, density=True, alpha=0.3, label=['clean', 'adv'][e])
            ax.tick_params(axis='both', labelsize=5)
            ax.autoscale()

        ax.set_title(f'C-{channel_id}|div-{clean_distance[channel_id]:0.3f}|adv_div-{fgsm_distance[channel_id]:0.3f}',
                     fontdict={'fontsize': 7})
    fig.suptitle(f'{layer_key} sorted by {mode} KL divergence' + (' ratio (adv/clean)' if max_ratio else ''))
    fig.legend()
    fig.set_size_inches(19.2, 10.8)
    fig.show()
    # fig.waitforbuttonpress()
    pass

def gen_simes_reducer_fn(ref_stats_dict):
    def _batch_calc_simes(trace_name, m, inputs):
        class_specific_stats = []
        for class_stat_dict in ref_stats_dict:
            reduction_specific_record = {}
            for reduction_name, reduction_stat in class_stat_dict[trace_name[:-8]].items():
                pval_per_input = []
                for e, per_input_stat in enumerate(reduction_stat):
                    reduced = per_input_stat.reduction_fn(inputs[e])
                    pval = per_input_stat.pval_matcher(reduced)
                    ## for now just reduce channles using simes
                    pval_per_input.append(calc_simes(pval))
                reduction_specific_record[reduction_name] = pval_per_input
            class_specific_stats.append(reduction_specific_record)
        return class_specific_stats
    return _batch_calc_simes


class PvalueMatcher():
    def __init__(self,percentiles,quantiles, two_side=True, right_side=False):
        self.percentiles = percentiles
        self.quantiles = quantiles.t().unsqueeze(0)
        self.num_percentiles = percentiles.shape[0]
        self.two_side = two_side
        self.right_side = right_side
        assert not (self.two_side and self.right_side)

    ## todo document!
    def __call__(self, x):
        if x.device != self.percentiles.device:
            self.percentiles=self.percentiles.to(x.device)
            self.quantiles = self.quantiles.to(x.device)
        stat_layer = x.unsqueeze(-1).expand(x.shape[0], x.shape[1], self.num_percentiles)
        quant_layer = self.quantiles.expand(stat_layer.shape[0], stat_layer.shape[1], self.num_percentiles)

        ### find p-values based on quantiles
        temp_location = self.num_percentiles - th.sum(stat_layer < quant_layer, -1)
        upper_quant_ind = temp_location > (self.num_percentiles // 2)
        temp_location[upper_quant_ind] += -1
        matching_percentiles = self.percentiles[temp_location]
        if self.two_side:
            matching_percentiles[upper_quant_ind] = 1 - matching_percentiles[upper_quant_ind]
            return matching_percentiles * 2
        if self.right_side:
            return 1-matching_percentiles
        return matching_percentiles

def extract_output_distribution(all_class_ref_stats,target_percentiles=th.tensor([0.001, 0.005, 0.025, 0.05, 0.1, 0.15,
                                                                                  0.2, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9,
                                                                                  0.95, 0.975, 0.995, 0.999]), right_sided_fisher_pvalue = False):
    # this function should return pvalues in the format of (Batch x num_classes)
    per_class_record = []
    # reduce all layers (e.g. fisher)
    for class_stats_per_layer_dict in all_class_ref_stats:
        sum_pval_per_reduction = {}
        for layer_stats_per_input_dict in class_stats_per_layer_dict.values():
            for reduction_name, record_per_input in layer_stats_per_input_dict.items():
                for record in record_per_input:
                    if reduction_name in sum_pval_per_reduction:
                        sum_pval_per_reduction[reduction_name] -= 2*th.log(record.simes_pval)
                    else:
                        sum_pval_per_reduction[reduction_name] = -2*th.log(record.simes_pval)
                    if record.pval_matcher is None:
                        record.pval_matcher = PvalueMatcher(*record.meter.get_distribution_histogram())
        # update fisher pvalue per reduction
        fisher_pvals_per_reduction = {}
        for reduction_name, sum_pval in sum_pval_per_reduction.items():
            meter = OnlineMeter(batched=True, track_percentiles=True,
                                target_percentiles= target_percentiles.clamp_(
                                1/sum_pval.shape[0],1-1/sum_pval.shape[0]).unique(),
                                                                per_channel=False, number_edge_samples=70,
                                                                track_cov=False)
            meter.update(sum_pval)
            # use right tail pvalue since we don't care about fisher "normal" looking pvalues that are closer to 0
            kwargs = {}
            if right_sided_fisher_pvalue:
                kwargs = {'two_side':False,'right_side':True}
            fisher_pvals_per_reduction[reduction_name] = PvalueMatcher(*meter.get_distribution_histogram(),**kwargs)

        per_class_record.append(fisher_pvals_per_reduction)

    return per_class_record

class OODDetector():
    def __init__(self, model, num_classes,channle_reducer_fn, output_pval_matcher):
        self.output_pval_matcher = output_pval_matcher
        self.num_classes = num_classes
        self.stats_recorder = Recorder(model, recording_mode=[Recorder._RECORD_INPUT_MODE[1]],
                                       include_matcher_fn=lambda n, m: isinstance(m,th.nn.BatchNorm2d) or
                                                                       isinstance(m, th.nn.Linear),
                                       input_fn=channle_reducer_fn,
                                       recursive=True, device_modifier='same')

    # helper function to convert per class per reduction to per reduction per class dictionary
    def _gen_output_dict(self,per_class_per_reduction_record):
        # prepare a dict with pvalues per reduction per sample per class i.e. {reduction_name : (BxC)}
        reduction_stats_collection = {}
        for class_stats in per_class_per_reduction_record:
            for reduction_name, pval in class_stats.items():
                if reduction_name in reduction_stats_collection:
                    reduction_stats_collection[reduction_name] = th.cat([reduction_stats_collection[reduction_name],
                                                                         pval.unsqueeze(1)],1)
                else:
                    reduction_stats_collection[reduction_name] = pval.unsqueeze(1)

        return reduction_stats_collection

    # this function should return pvalues in the format of (Batch x num_classes)
    def get_fisher(self):
        per_class_record = []
        # reduce all layers (e.g. fisher)
        for class_id in range(self.num_classes):
            sum_pval_per_reduction = {}
            # iterate over recorder to aggragate per layer statistics for the new batch
            for layer_pvalues_per_class in self.stats_recorder.record.values():
                for reduction_name,pval_per_input in layer_pvalues_per_class[class_id].items():
                    for pval in pval_per_input:
                        if reduction_name in sum_pval_per_reduction:
                            sum_pval_per_reduction[reduction_name] -= 2*th.log(pval)
                        else:
                            sum_pval_per_reduction[reduction_name] = -2*th.log(pval)

            # update fisher pvalue per reduction
            fisher_pvals_per_reduction = {}
            for reduction_name, sum_pval in sum_pval_per_reduction.items():
                fisher_pvals_per_reduction[reduction_name] = self.output_pval_matcher[class_id][reduction_name](sum_pval)

            per_class_record.append(fisher_pvals_per_reduction)

        self.stats_recorder.record.clear()
        return self._gen_output_dict(per_class_record)

# clac Simes per batch element (samples x variables)
def calc_simes(pval):
    pval, _ = th.sort(pval, 1)
    rank = th.arange(1, pval.shape[1] + 1,device=pval.device).repeat(pval.shape[0], 1)
    simes_pval, _ = th.min(pval.shape[1] * pval / rank, 1)
    return simes_pval.unsqueeze(1)

def spatial_mean(x):
    return x.mean(tuple(range(2, x.dim()))) if x.dim() > 2 else x

# k will ignore k-1 most extreme values, todo choose value to match percentile ()
def spatial_edges(x,k=1,is_max=True,):
    x_ = x
    if x.dim() < 3:
        return x

    if x.dim()>3:
        x_ = x.view(x.shape[0], x.shape[1], -1)

    ret = x_.topk(k, -1, is_max)[0]
    if is_max:
        return ret[:,:,0]

    return ret[:,:,k-1]

def spatial_min(x,k=1):
    return spatial_edges(x,k,False)

def spatial_max(x,k=1):
    return spatial_edges(x,k,True)

def spatial_margin(x,k=1):
    return spatial_max(x,k)-spatial_min(x,k)




## auxilary data containers
@dataclass()
class BatchStatsCollectorRet():
    def __init__(self, reduction_name: str,
                 reduction_fn = lambda x: x,
                 cov: th.Tensor = None,
                 num_observations: int = 0,
                 simes_pval: th.Tensor = None,
                 meter :AverageMeter = None,
                 pval_matcher : PvalueMatcher = None):
        self.reduction_name = reduction_name
        self.cov = cov
        self.num_observations = num_observations
        self.simes_pval = simes_pval
        self.meter = meter
        self.reduction_fn = reduction_fn
        self.pval_matcher = pval_matcher

@dataclass()
class BatchStatsCollectorCfg():
    cov_off : bool = True
    _track_cov: bool = False
    partial_stats: bool = False
    update_tracker: bool = True
    find_simes: bool = False
    reduction_dictionary = {
        'spatial-mean': spatial_mean,
        'spatial-max': spatial_max,
        'spatial-min': spatial_min,
        'spatial-margin': partial(spatial_margin,k=1)
    }
    target_percentiles = [0.001, 0.005, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5,
               0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.995, 0.999]
    num_edge_samples: int = 70

    def __init__(self,batch_size):
        # adjust percentiles to the specified batch size
        if self.target_percentiles:
            self.target_percentiles = th.tensor(self.target_percentiles).clamp_(1/batch_size,1-1/batch_size).unique()

def measure_data_statistics(loader, model, epochs=5, model_device='cuda', collector_device='same', batch_size=1000,
                            measure_settings : BatchStatsCollectorCfg = None):

    measure_settings = measure_settings or BatchStatsCollectorCfg(batch_size)
    compute_cov_on_partial_stats = measure_settings.partial_stats and not measure_settings.cov_off
    ## bypass the simple recorder dictionary with a meter dictionary to track statistics
    tracker = MeterDict(meter_factory=lambda k, v: OnlineMeter(batched=True, track_percentiles=True,
                                                               target_percentiles=measure_settings.target_percentiles,
                                                               per_channel=True,number_edge_samples=measure_settings.num_edge_samples,
                                                               track_cov=compute_cov_on_partial_stats))

    # function collects statistics of a batched tensors, return the collected statistics per input tensor
    def _batch_stats_collector(trace_name, m, inputs):
        stats_per_input = []
        for e, i in enumerate(inputs):
            reduction_specific_record = []
            for reduction_name, reduction_fn in measure_settings.reduction_dictionary.items():
                tracker_name = f'{trace_name}_{reduction_name}:{e}'
                ## make sure input is a 2d tensor [batch, nchannels]
                i_ = reduction_fn(i)
                if collector_device != 'same' and collector_device != model_device:
                    i_ = i_.to(collector_device)

                num_observations, channels = i_.shape
                reduction_ret_obj = BatchStatsCollectorRet(reduction_name,reduction_fn,num_observations=num_observations)

                # we dont always want to update the tracker, particularly when we call the method again to collect
                # statistics that depends on previous values that are computed on the entire measure dataset
                if measure_settings.update_tracker:
                    # tracker keeps track statistics such as number of samples seen, mean, var, percentiles and potentially
                    # running mean covariance
                    tracker.update({tracker_name: i_})
                # save a reference to the meter for convenience
                reduction_ret_obj.meter = tracker[tracker_name]

                if measure_settings.find_simes:
                    reduction_ret_obj.pval_matcher = PvalueMatcher(
                        *reduction_ret_obj.meter.get_distribution_histogram())
                    pval = reduction_ret_obj.pval_matcher(i_)
                    reduction_ret_obj.simes_pval = calc_simes(pval)

                # calculate covariance, can be used to compute covariance with global mean instead of running mean
                if measure_settings._track_cov:
                    _i_mean = reduction_ret_obj.meter.mean
                    _i_centered = i_ - _i_mean
                    reduction_ret_obj.cov = _i_centered.transpose(1, 0).matmul(_i_centered) / (num_observations)

                reduction_specific_record.append(reduction_ret_obj)

            stats_per_input.append(reduction_specific_record)

        return stats_per_input

    # this functionality is used to calculate a more accurate covariance estimate
    def _batch_stats_reducer(old_record, new_entry):
        stats_per_input = []
        for input_id, reduction_stats_record_n in enumerate(new_entry):
            reductions_per_input = []
            for reduction_id, new_reduction_ret_obj in enumerate(reduction_stats_record_n):
                reduction_ret_obj = old_record[input_id][reduction_id]
                assert reduction_ret_obj.reduction_name == new_reduction_ret_obj.reduction_name
                # compute global mean covariance update
                if new_reduction_ret_obj.cov is not None:
                    if reduction_ret_obj.cov is None:
                        reduction_ret_obj.cov = new_reduction_ret_obj.cov
                        reduction_ret_obj.num_observations = new_reduction_ret_obj.num_observations
                    else:
                        reduction_ret_obj.num_observations += new_reduction_ret_obj.num_observations
                        scale = new_reduction_ret_obj.num_observations / reduction_ret_obj.num_observations
                        # delta
                        delta = new_reduction_ret_obj.cov.sub(reduction_ret_obj.cov)
                        # update mean covariance
                        reduction_ret_obj.cov.add_(delta.mul_(scale))

                # Simes aggregated (for now we collect all of the observed values and process them later for simplicity)
                if new_reduction_ret_obj.simes_pval is not None:
                    if reduction_ret_obj.simes_pval is None:
                        reduction_ret_obj.simes_pval=new_reduction_ret_obj.simes_pval
                    else:
                        reduction_ret_obj.simes_pval = th.cat([reduction_ret_obj.simes_pval, new_reduction_ret_obj.simes_pval])

                reductions_per_input.append(reduction_ret_obj)
            stats_per_input.append(reductions_per_input)
        return stats_per_input

    #simple loop over measure data to collect statistics
    def _loop_over_data():
        model.eval()
        with th.no_grad():
            for _ in tqdm.trange(epochs):
                for d, l in loader:
                    _ = model(d.to(model_device))

    model.to(model_device)
    r = Recorder(model, recording_mode=[Recorder._RECORD_INPUT_MODE[1]],
                 include_matcher_fn=lambda n, m: isinstance(m, th.nn.BatchNorm2d) or isinstance(m, th.nn.Linear),
                 input_fn=_batch_stats_collector,
                 activation_reducer_fn=_batch_stats_reducer, recursive=True, device_modifier='same')

    # if compute_cov_on_partial_stats:
    #     # todo compare aginst meter cov
    #     measure_settings._track_cov = True

    logging.info('measuring mean and percentiles')
    _loop_over_data()
    logging.info('calculating covariance and Simes pvalues using measured mean and quantiles')
    measure_settings.update_tracker = False
    measure_settings.find_simes = True
    measure_settings._track_cov = not (measure_settings._track_cov or measure_settings.cov_off)
    _loop_over_data()

    ## build reference dictionary
    ret_stat_dict = {}
    for k in r.tracked_modules.keys():
        ret_stat_dict[k] = {}
        for kk, stats_per_input in r.record.items():
            if kk.startswith(k):
                for inp_id, reduction_records in enumerate(stats_per_input):
                    for reduction_record in reduction_records:
                        assert isinstance(reduction_record,BatchStatsCollectorRet)
                        if reduction_record.reduction_name in ret_stat_dict[k]:
                            ret_stat_dict[k][reduction_record.reduction_name] += [reduction_record]
                        else:
                            ret_stat_dict[k][reduction_record.reduction_name] = [reduction_record]
    r.record.clear()
    r.remove_model_hooks()
    return ret_stat_dict


def evaluate_data(loader,model, detector,model_device,alpha = 0.001,class_conditional=True,report_accuracy=False):
    model.eval()
    model.to(model_device)
    top1 = AverageMeter()
    top5 = AverageMeter()
    rejected = {'spatial-mean':AverageMeter(),
                'spatial-max':AverageMeter(),
                'spatial-min':AverageMeter(),
                'spatial-margin':AverageMeter()}
    with th.no_grad():
        for d, l in tqdm.tqdm(loader, total=len(loader)):
            out = model(d.to(model_device))
            if report_accuracy:
                t1, t5 = accuracy(out, l, (1, 5))
                top1.update(t1, out.shape[0])
                top5.update(t5, out.shape[0])
                logging.info(f'\nPrec@1 {top1.avg:.3f} ({top1.std:.3f}) \t'
                             f'Prec@5 {top5.avg:.3f} ({top5.std:.3f})')
            pvalues_dict = detector.get_fisher()
            for reduction_name,pvalues in pvalues_dict.items():
                #aggragate pvalues or return per reduction score
                # todo:
                #  1. split rejection statistics between correct and incorrect model predictions
                #  2. check rejection rate according to all class statistics (i.e. reject only if pval is unlikely
                #  under all class reference)
                #  3. compute accuracy according to maximum likelihood class from 2.
                best_class_pval, best_class_pval_id = pvalues.squeeze().max(1)
                if class_conditional:
                    predicted_class_pval = pvalues[th.where(out.max(1,keepdims=True)[0] == out)]
                else:
                    predicted_class_pval = best_class_pval
                # we typically evaluate only in/out dist data at a time so we only care about rejection rate
                # (instead of accuracy)
                rejected[reduction_name].update((predicted_class_pval < alpha).float().mean(),out.shape[0])
                if report_accuracy:
                    predicted = out.max(1)[1].cpu()
                    t1_likely,t5_likely = accuracy(pvalues.squeeze(),l,(1,5))
                    agreement = (best_class_pval_id.cpu() == predicted)
                    agreement_true = (agreement == (predicted == l))
                    agreement_false =  pvalues[predicted != l]
                    logging.info(f'{reduction_name}: rejected: {rejected[reduction_name].avg:0.3f}\t'
                                 f'Prec@1 {t1_likely:0.3f}, Prec@5 {t5_likely:0.3f}\n'
                                 f'\tagreement: {agreement.float().mean():0.3f},\t'
                                 f'agreement on true: {agreement_true.float().mean():0.3f},\t'
                                 f'agreement on false {agreement_false.float().mean():0.3f}')
                # we typically evaluate only in/out dist data at a time so we only care about rejection rate
                # (instead of accuracy)

        for reduction_name, rejected_p in rejected.items():
            logging.info(f'{reduction_name} rejected: {rejected_p.avg:0.3f}')

    # Important! recorder hooks should be removed when done

if __name__ == '__main__':
    # args = Settings(
    #     model = 'resnet',
    #     dataset = 'cats_vs_dogs',
    #     model_cfg={'dataset': 'imagenet', 'depth': 18, 'num_classes': 2},
    #     ckt_path = '/home/mharoush/myprojects/convNet.pytorch/results/r18_cats_N_dogs/checkpoint.pth.tar'
    # )
    #
    args = Settings(
        model='ResNet34',
        dataset='SVHN', #'cifar10',
        transform_dataset = 'cifar10',
        num_classes = 10,
        model_cfg={'num_c': 10},
        ckt_path='/home/mharoush/myprojects/Residual-Flow/pre_trained/resnet_cifar10.pth',
        limit_measure=None,
        limit_test=None,
        test_split='val',
        augment_measure=False,
        right_sided_fisher_pvalue=True
    )
    # args = Settings(
    #     model='ResNet34',
    #     dataset='cifar100',
    #     num_classes=100,
    #     model_cfg={'num_c': 100},
    #     batch_size = 1000,
    #     ckt_path='/home/mharoush/myprojects/Residual-Flow/pre_trained/resnet_cifar100.pth',
    #     device='cuda:1',
    #     augment_measure=False,
    #     limit_test =None,
    #     test_split='val'
    # )
    # args = Settings(
    #     model='ResNet34',
    #     dataset='svhn',
    #     num_classes=10,
    #     model_cfg={'num_c': 10},
    #     batch_size = 1000,
    #     ckt_path='/home/mharoush/myprojects/Residual-Flow/pre_trained/resnet_svhn.pth',
    #     device='cuda:1',
    #     augment_measure=False,
    #     limit_test =1000,
    #     test_split='val',
    # )
    # args = Settings(
    #     model='DenseNet3',
    #     dataset='cifar10',
    #     num_classes = 10,
    #     model_cfg={'num_classes': 10,'depth':100},
    #     ckt_path='densenet_cifar10_ported.pth',
    #     # device='cuda:2',
    #     # limit_measure=None,
    #     # limit_test=1000,
    #     # test_split='val',
    #     augment_measure=False
    # )
    # args = Settings(
    #     model='DenseNet3',
    #     dataset='svhn',
    #     num_classes = 10,
    #     model_cfg={'num_classes': 10,'depth':100},
    #     ckt_path='/home/mharoush/myprojects/Residual-Flow/pre_trained/densenet_svhn.pth',
    #     # device='cuda:2',
    #     # limit_measure=None,
    #     # limit_test=1000,
    #     # test_split='val',
    #     augment_measure=False
    # )
    # args = Settings(
    #     model='DenseNet3',
    #     dataset='cifar100',
    #     num_classes = 100,
    #     model_cfg={'num_classes': 100,'depth':100},
    #     ckt_path='densenet_cifar100_ported.pth',
    #     device='cuda:1',
    #     batch_size=500,
    #     collector_device='cpu',
    #     # limit_measure=None,
    #     # limit_test=1000,
    #     # test_split='val',
    #     augment_measure=False
    #     )

    setup_logging()
    logging.info(args)
    model = getattr(models,args.model)(**(args.model_cfg))
    checkpoint = th.load(args.ckt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        checkpoint=checkpoint['state_dict']

    model.load_state_dict(checkpoint)

    if args.augment_measure:
        epochs = 5
    else:
        epochs = 1

    expected_transform_measure = get_transform(args.transform_dataset or args.dataset, augment=args.augment_measure)
    expected_transform_test = get_transform(args.transform_dataset or args.dataset, augment=args.augment_test)
    calibrated_path = f'measured_stats_per_class-{args.model}-{args.dataset}-{"augment" if args.augment_measure else "no_augment"}.pth'
    if not args.recompute and os.path.exists(calibrated_path):
        all_class_ref_stats = th.load(calibrated_path,map_location=lambda storage, loc: storage)
    else:
        ds = get_dataset(args.dataset, 'train', expected_transform_measure)
        classes = ds.classes if hasattr(ds,'classes') else range(args.num_classes)
        if args.limit_measure:
            ds=limit_ds(ds,args.limit_measure,per_class=True)
        all_class_ref_stats=[]
        targets = th.tensor(ds.targets) if hasattr(ds,'targets') else  th.tensor(ds.labels)
        for class_id,class_name in enumerate(classes):
            logging.info(f'collecting stats for class {class_name}')
            sampler = th.utils.data.SubsetRandomSampler(th.where(targets==class_id)[0])
            train_loader = th.utils.data.DataLoader(
                    ds, sampler=sampler,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=8, pin_memory=True, drop_last=True)

            all_class_ref_stats.append(measure_data_statistics(train_loader, model, epochs=epochs, model_device=args.device,
                                                collector_device=args.collector_device, batch_size=args.batch_size))
        logging.info('saving reference stats dict')
        th.save(all_class_ref_stats, calibrated_path)

    val_ds = get_dataset(args.dataset, args.test_split ,expected_transform_test)
    if args.limit_test:
        val_ds = limit_ds(val_ds,args.limit_test,per_class=False)
    # get the pvalue estimators per class per reduction for the channel and final output functions
    # here we can implement subsample channles and other layer reductions exept Fisher
    # todo add Specs to fuse the pvalue matcher and OODDetector as they are tightly coupled.
    output_pval_matcher = extract_output_distribution(all_class_ref_stats,right_sided_fisher_pvalue=args.right_sided_fisher_pvalue)
    detector = OODDetector(model,args.num_classes,gen_simes_reducer_fn(all_class_ref_stats),output_pval_matcher)

    # todo replace all targets to in or out of distribution?
    # todo add adversarial samples test
    # run in-dist data evaluate per class to simplify analysis
    logging.info(f'evaluating inliers')
    #for class_id,class_name in enumerate(val_ds.classes):
    sampler = None #th.utils.data.SubsetRandomSampler(th.where(targets==class_id)[0]) #th.utils.data.RandomSampler(ds, replacement=True,num_samples=5000)
    val_loader = th.utils.data.DataLoader(
            val_ds, sampler=sampler,
            batch_size=args.batch_size, shuffle=False,
            num_workers=8, pin_memory=True, drop_last=True)
    evaluate_data(val_loader, model, detector,args.device,0.05,report_accuracy=True)

    logging.info(f'evaluating outliers')
    ood_datasets = ['folder-LSUN_resize','folder-Imagenet_resize','SVHN']
    for ood_dataset in ood_datasets:
        ood_ds = get_dataset(ood_dataset, 'val',expected_transform_test)
        if args.limit_test:
            ood_ds = limit_ds(ood_ds,args.limit_test,per_class=False)
        ood_loader = th.utils.data.DataLoader(
            ood_ds, sampler=None,
            batch_size=args.batch_size, shuffle=False,
            num_workers=8, pin_memory=False, drop_last=True)
        logging.info(f'evaluating {ood_dataset}')
        evaluate_data(ood_loader, model, detector, args.device, 0.05)

pass

# record = th.load(f'record-{args.dataset}.pth')
# adv_tag = 'FGSM_0.1'
# layer_inputs = recorded[layer_name]
# layer_inputs_fgsm = recorded[f'{layer_name}-@{adv_tag}']
# def _maybe_slice(tensor, nsamples=-1):
#     if nsamples > 0:
#         return tensor[0:nsamples]
#     return tensor
#
# for layer_name in args.plot_layer_names:
#     clean_act = _maybe_slice(record[layer_name + '_forward_input:0'])
#     fgsm_act = _maybe_slice(record[layer_name + f'_forward_input:0-@{adv_tag}'])
#
#     plot(clean_act, fgsm_act, layer_name, reference_stats=ref_stats, rank_by_stats_loss=True, max_ratio=False)
# plt.waitforbuttonpress()
