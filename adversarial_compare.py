import torch as th
import torchvision as tv
from matplotlib import pyplot as plt
import models
from data import get_dataset
from preprocess import get_transform
import os
import tqdm
from utils.absorb_bn import get_bn_params
from dataclasses import dataclass
import inspect


@dataclass
class Settings:
    def __repr__(self):
        return str(self.__class__.__name__) + str({attr:getattr(self,attr) for attr in self.__dir__()
                                                   if type(attr) == str and not attr.startswith('__')})

    def __init__(self,
                 layer_names: list,
                 model_cfg: dict,
                 measure_stats: bool = True,
                 recompute: bool = True,
                 augment: bool = False,
                 device: str = 'cuda',
                 dataset: str = f'cats_vs_dogs',
                 super_category: str = '', # 'cats'  # dogs
                 ckt_path: str = '/home/mharoush/myprojects/convNet.pytorch/results/r18_cats_N_dogs/checkpoint.pth.tar',
                 collector_device : str = 'cpu'):

        arg_names, _,_, local_vars= inspect.getargvalues(inspect.currentframe())
        for name in arg_names[1:]:
            setattr(self,name,local_vars[name])
        if self.super_category:
            self.dataset += '-' + self.super_category

args = Settings(
    recompute=False,
    augment=False,
    device='cuda:2',
    collector_device='cuda:3',
    super_category='dogs',
    layer_names=['layer1.0.bn1', 'layer2.0.downsample.1', 'layer2.1.bn2',
                             'layer4.0.downsample.1','layer4.1.bn2'],
    model_cfg={'dataset': 'imagenet', 'depth': 18, 'num_classes': 2})


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


def measure_data_statistics(loader, model, epochs=10, model_device='cuda:2', collector_device='cuda:3', stats_dict={},
                            collect_pecentiles=th.tensor([0, 0.005,0.001, 0.025, 0.05, 0.5, 0.995, 0.975, 0.999, 0.995, 1]), num_edge_samples=5):
    from utils.misc import Recorder
    from utils.meters import MeterDict,OnlineMeter
    ## this tracks simplified statistics
    tracker = MeterDict(meter_factory=lambda k,v: OnlineMeter(batched=True))

    @dataclass()
    class measure_cfg():
        compute_cov_on_partial_stats :bool = False
        update_tracker :bool = True

    measure_settings = measure_cfg()
    reduction_dictionary = {
        'spatial-mean': lambda x: x.mean(tuple(range(2, x.dim()))) if x.dim() > 2 else x,
        'spatial-max': lambda x: x.view(x.shape[0],x.shape[1],-1).max(-1)[0] if x.dim() > 2 else x,
        'spatial-min': lambda x: x.view(x.shape[0],x.shape[1],-1).min(-1)[0] if x.dim() > 2 else x,
    }

    # function collects statistics of a batched tensors, return the collected statistics per input tensor
    def _batch_stats_collector(trace_name, m, inputs):
        stats_per_input = []
        for e, i in enumerate(inputs):
            reduction_specific_record=[]
            for reduction_name, reduction_fn in reduction_dictionary.items():
                tracker_name_pattern = f'{trace_name}_{reduction_name}_@@:{e}'
                tracker_mean_name = tracker_name_pattern.replace("@@","mean")
                tracker_cent_name = tracker_name_pattern.replace("@@","cent")
                ## make sure input is a 2d tensor [batch, nchannels]
                i_ = reduction_fn(i)
                if collector_device != model_device:
                    i_ = i_.to(collector_device)
                ## find tensor percentile per channel
                num_observations, channels = i_.shape

                # sort tensor and slice percentiles from the sorted tensor per channel
                sorted = i_.sort(0)[0]
                if measure_settings.update_tracker:
                    # compute ids for each percentile
                    percentile_ids = th.repeat_interleave(
                        th.round(collect_pecentiles * (num_observations - 1)).long()[:, None], channels, 1)
                    percentile_values = sorted.gather(0, percentile_ids.to(i_.device))
                    tracker.update({tracker_cent_name: percentile_values.unsqueeze(0), tracker_mean_name: i_})

                if not (measure_settings.compute_cov_on_partial_stats or tracker_mean_name in stats_dict):
                    continue

                # calculate covariance
                if tracker_mean_name in stats_dict:
                    # provided reference dictionary we can use a more accurate measure for the per-channel mean
                    # (typically we will first calculate the global mean then use it to get a better covariance estimator)
                    _i_mean = stats_dict[tracker_mean_name]
                else:
                    # use tracked mean from previous steps
                    _i_mean = tracker[tracker_mean_name]

                if isinstance(_i_mean, OnlineMeter):
                    _i_mean = _i_mean.mean

                _i_centered = i_ - _i_mean
                cov_n = _i_centered.transpose(1, 0).matmul(_i_centered)/(num_observations)

                min_obs = sorted[:num_edge_samples]
                max_obs = sorted[-num_edge_samples:]

                reduction_specific_record.append([reduction_name,cov_n, num_observations, min_obs, max_obs])
            stats_per_input.append(reduction_specific_record)

        return stats_per_input

    # this functionality is used to calculate a more accurate covariance estimate
    def _batch_stats_reducer(old_record, new_entry):
        stats_per_input = []
        for input_id, reduction_stats_record_n in enumerate(new_entry):
            reductions_per_input=[]
            for reduction_id,(reduction_name_n,cov_n, obs_n,min_obs_n, max_obs_n) in enumerate(reduction_stats_record_n):
                reduction_name,cov, obs,min_obs, max_obs = old_record[input_id][reduction_id]
                assert reduction_name==reduction_name_n
                # update worst case observations
                max_obs = th.cat([max_obs,max_obs_n]).sort(0)[0][-len(max_obs):]
                min_obs = th.cat([min_obs,min_obs_n]).sort(0)[0][:len(min_obs)]
                #compute new covariance
                tot_obs = obs + obs_n
                scale = obs_n / tot_obs
                #delta
                delta = cov_n.sub(cov)
                #update mean covariance
                cov.add_(delta.mul_(scale))
                reductions_per_input.append([reduction_name, cov, tot_obs, min_obs, max_obs])
            stats_per_input.append(reductions_per_input)
        return stats_per_input

    def _loop_over_data():
        model.eval()
        with th.no_grad():
            for e in range(epochs):
                print(f'measure statistics - epoch {e}')
                for d, l in tqdm.tqdm(loader,total=len(loader)):
                    _ = model(d.to(model_device))

    model.to(model_device)
    r = Recorder(model, recording_mode=[Recorder._RECORD_INPUT_MODE[1]],
                 include_matcher_fn=lambda n, m: isinstance(m, th.nn.BatchNorm2d) or isinstance(m,th.nn.Linear), input_fn=_batch_stats_collector,
                 activation_reducer_fn=_batch_stats_reducer,recursive=True,device_modifier='same')

    _loop_over_data()
    if not stats_dict:
        print('calculating covariance using measured mean')
        r.record.clear()
        stats_dict.update(tracker)
        measure_settings.update_tracker=False
        _loop_over_data()
    ## build reference dictionary
    ret_stat_dict = {}
    for k in r.tracked_modules.keys():
        ret_stat_dict[k] = {}
        for kk, stats_per_input in r.record.items():
            if kk.startswith(k):
                for inp_id,reduction_records in enumerate(stats_per_input):
                    for reduction_record in reduction_records:
                        reduction_name = reduction_record[0]
                        ret_stat_dict[k][reduction_name]={
                            f'cov:{inp_id}'     : reduction_record[1],
                            f'count:{inp_id}'   : reduction_record[2],
                            f'min_obs:{inp_id}' : reduction_record[3],
                            f'max_obs:{inp_id}' : reduction_record[4]
                        }

        for kk, v in tracker.items():
            if kk.startswith(k):
                reduction_name,stat_name=kk.split('_')[-2:]
                ret_stat_dict[k][reduction_name][stat_name]= v.mean
        # ret_stat_dict.update({k[:-len('input_fn')]+f'min_obs:0':v[0][2] for kk,v in r.record.items()})
        # ret_stat_dict.update({k[:-len('input_fn')]+f'max_obs:0':v[0][3] for k,v in r.record.items()})
        # ret_stat_dict.update({k:v.mean for k,v in tracker.items()})
    return ret_stat_dict

model = models.resnet(**(args.model_cfg))
model.load_state_dict(th.load(args.ckt_path)['state_dict'])

if args.augment:
    epochs = 5
else:
    epochs = 1

if args.measure_stats:
    calibrated_path = f'record-measured_stats-{args.super_category}.pth'
    if not args.recompute and os.path.exists(calibrated_path):
        ref_stats = th.load(calibrated_path,map_location=lambda storage, loc: storage)
    else:
        ds = get_dataset(args.dataset, 'train',
                           get_transform('imagenet', augment=args.augment),limit=None)
        sampler = None #th.utils.data.RandomSampler(ds, replacement=True,num_samples=5000)
        train_loader = th.utils.data.DataLoader(
                ds, sampler=sampler,
                batch_size=1000, shuffle=True, #(sampler is None),
                num_workers=8, pin_memory=False, drop_last=True)
        # calibrate_bn to get single class statistics per layer
        ref_stats = measure_data_statistics(train_loader, model, epochs=epochs, model_device=args.device,
                                            collector_device=args.collector_device)
        print('saving reference stats dict')
        th.save(ref_stats, calibrated_path)
else:
    ref_stats = get_bn_params(model)
record = th.load(f'record-{args.dataset}.pth')
# ref_stats.keys()
adv_tag = 'FGSM_0.1'

# layer_inputs = recorded[layer_name]
# layer_inputs_fgsm = recorded[f'{layer_name}-@{adv_tag}']

def _maybe_slice(tensor, nsamples=-1):
    if nsamples > 0:
        return tensor[0:nsamples]
    return tensor


for layer_name in args.layer_names:
    clean_act = _maybe_slice(record[layer_name + '_forward_input:0'])
    fgsm_act = _maybe_slice(record[layer_name + f'_forward_input:0-@{adv_tag}'])

    plot(clean_act, fgsm_act, layer_name, reference_stats=ref_stats, rank_by_stats_loss=True, max_ratio=False)

plt.waitforbuttonpress()
