#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import itertools
import json
import logging
import math
import os
import sys

from chainer.datasets import TransformDataset
from chainer import reporter as reporter_module
from chainer import training
from chainer.training import extensions
from chainer.training import StandardUpdater
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch.nn.parallel import data_parallel
from espnet.asr.asr_utils import freeze_parameters
from espnet.asr.asr_utils import adadelta_eps_decay
from espnet.asr.asr_utils import sgd_lr_decay
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import CompareValueTrigger
from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import plot_spectrogram
from espnet.asr.asr_utils import restore_snapshot
from espnet.asr.asr_utils import snapshot_object
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import torch_resume
from espnet.asr.asr_utils import torch_snapshot
from espnet.asr.pytorch_backend.asr import CustomEvaluator
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
import espnet.lm.pytorch_backend.extlm as extlm_pytorch
from espnet.nets.asr_interface import ASRInterface
from espnet.nets.tts_interface import TTSInterface
from espnet.nets.pytorch_backend.e2e_asr import pad_list
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.nets.pytorch_backend.streaming.segment import SegmentStreamingE2E
from espnet.nets.pytorch_backend.streaming.window import WindowStreamingE2E
from espnet.transform.spectrogram import IStft
from espnet.transform.transformation import Transformation
from espnet.utils.cli_writers import file_writer_helper
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import
from espnet.utils.io_utils import LoadInputsAndTargets
from espnet.utils.training.batchfy import make_batchset
from espnet.utils.training.evaluator import BaseEvaluator
from espnet.utils.training.iterators import ShufflingEnabler
from espnet.utils.training.iterators import ToggleableShufflingMultiprocessIterator
from espnet.utils.training.iterators import ToggleableShufflingSerialIterator
from espnet.utils.training.tensorboard_logger import TensorboardLogger
from espnet.utils.training.train_utils import check_early_stop
from espnet.utils.training.train_utils import set_early_stop

import matplotlib
matplotlib.use('Agg')

if sys.version_info[0] == 2:
    from itertools import izip_longest as zip_longest
else:
    from itertools import zip_longest as zip_longest

def merge_batchsets(train_paired, train_unpaired, repeat=False):
    import math
    ndata = max(len(train_unpaired), len(train_paired))
    ratio = math.trunc(len(train_unpaired) / len(train_paired))
    train = []
    for i in range(ndata):
        if repeat:
            try:
                train.append(train_paired[i])
            except:
                train.append(train_paired[i+1-i])
            try:
                train.append(train_unpaired[i])
            except:
                logging.info('all unpaired data is loaded')
        else:
            if len(train) >= int(ndata+len(train_paired)-1):
                logging.info('all paired and unpaired data is loaded')
                break
            else:
                try:
                    train.append(train_paired[i])
                except:
                    logging.info('all paired data is loaded')
                    # train.append(train_paired[i+1-i])
                try:
                    for k in range(ratio):
                        try:
                            train.append(train_unpaired[i+k])
                        except:
                            logging.info('unpaired data at %s loaded', k)
                except:
                    logging.info('all unpaired data is loaded')

    return train, ratio

#encoder.embed.out.0.weight
#decoder.embed.0.weight
#decoder.output_layer.weight 
#decoder.output_layer.bias
#ctc.ctc_lo.weight
#ctc.ctc_lo.bias 

def mask_by_length_and_multiply(xs, length, fill=0, msize=1):
    assert xs.size(0) == len(length)
    ret = xs.data.new(xs.size(0) * msize, xs.size(1), xs.size(2)).fill_(fill)
    k = 0
    new_length = length.new(len(length) * msize)
    for i, l in enumerate(length):
        for j in range(msize):
            ret[k, :l] = xs[i, :l]
            new_length[k] = length[i]
            k += 1
    return ret, new_length

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
    nets (network list)   -- a list of networks
    requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

class CustomConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        subsampling_factor (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, subsampling_factor=1, dtype=torch.float32):
        self.subsampling_factor = subsampling_factor
        self.ignore_id = -1
        self.dtype = dtype

    def __call__(self, batch, device):
        """Transforms a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be located in list
        assert len(batch) == 1
        try:
            xs, spembs, ys = batch[0]
        except:
            xs, ys = batch[0]
            spembs = None

        # perform subsampling
        if self.subsampling_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        if xs[0].dtype.kind == 'c':
            xs_pad_real = pad_list(
                [torch.from_numpy(x.real).float() for x in xs], 0).to(device, dtype=self.dtype)
            xs_pad_imag = pad_list(
                [torch.from_numpy(x.imag).float() for x in xs], 0).to(device, dtype=self.dtype)
            # Note(kamo):
            # {'real': ..., 'imag': ...} will be changed to ComplexTensor in E2E.
            # Don't create ComplexTensor and give it E2E here
            # because torch.nn.DataParellel can't handle it.
            xs_pad = {'real': xs_pad_real, 'imag': xs_pad_imag}
        else:
            xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device, dtype=self.dtype)

        ilens = torch.from_numpy(ilens).to(device)
        # NOTE: this is for multi-task learning (e.g., speech translation)
        ys_pad = pad_list([torch.from_numpy(np.array(y[0]) if isinstance(y, tuple) else y).long()
                           for y in ys], self.ignore_id).to(device)
        if spembs:
            return xs_pad, ilens, ys_pad, torch.tensor(spembs).to(device)
        else:
            return xs_pad, ilens, ys_pad

class CycleUpdater(StandardUpdater):
    """Custom Updater for Pytorch.

    Args:
        model (torch.nn.Module): The model to update.
        grad_clip_threshold (float): The gradient clipping value to use.
        train_iter (chainer.dataset.Iterator): The training iterator.
        optimizer (torch.optim.optimizer): The training optimizer.

        converter (espnet.asr.pytorch_backend.asr.CustomConverter): Converter
            function to build input arrays. Each batch extracted by the main
            iterator and the ``device`` option are passed to this function.
            :func:`chainer.dataset.concat_examples` is used by default.

        device (torch.device): The device to use.
        ngpu (int): The number of gpus to use.
        use_apex (bool): The flag to use Apex in backprop.

    """

    def __init__(self, *args, **kwargs):
        self.model, self.tts_model = kwargs.pop('models')
        params = kwargs.pop('params')
        #converter = kwargs.pop('converter')
        super(CycleUpdater, self).__init__(*args, **kwargs)
        self.alpha = params['alpha']
        self.tts = params['tts']
        self.char_list = params['char_list']
        self.beam_size = params['beam_size']
        self.nbest = params['nbest']
        self.grad_clip_threshold = params['grad_clip_threshold']
        self.converter = params['converter']
        self.device = params['device']
        self.ngpu = params['ngpu']
        self.accum_grad = params['accum_grad']
        self.forward_count = 0
        self.grad_noise = params['grad_noise']
        self.iteration = 0
        self.use_apex = params['use_apex']
        self.update_tts = params['update_tts']
        self.speech_only = params['speech_only']
        self.text_only = params['text_only']

    def policy_rewards(self, log_probs, rewards):
        gamma = 0.99
        R = 0
        eps = np.finfo(np.float32).eps.item()
        returns = rewards
        policy_loss = []
        #for r in rewards:
        #    R = r + gamma * R
        #    returns.insert(0, R)
        #returns = torch.tensor(returns)
        returns = (returns - returns.mean()) # / (returns.std() + eps)
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(log_prob * R)
        policy_loss = torch.stack(policy_loss)
        return policy_loss

    def random_sampler(self, hyps, xlens, xs, spembs):
        # convert hyps to xs, xlens to ylens, ys to xs
        # separate yseq from dictionary of nbest_hyps
        ys = hyps
        #for i, y_hat in enumerate(ys):
            #seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
            #seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
            #logging.info("prediction[%d]: " % i + seq_hat_text)
        xlens_tts = torch.from_numpy(np.array([y.shape[0] for y in ys])).long().to(self.device)
        xlens_tts = sorted(xlens_tts, reverse=True)
        xs_tts = pad_list([y.long() for y in ys], 0).to(self.device)
        xs, xlens = mask_by_length_and_multiply(xs, xlens, 0, self.nbest)
        onelens = np.fromiter((1 for xx in spembs), dtype=np.int64)
        spembs, _ = mask_by_length_and_multiply(spembs.unsqueeze(1), torch.tensor(onelens), 0, self.nbest)
        spembs = spembs.squeeze(1)
        ylens_tts = torch.Tensor([ torch.max(xlens) for _ in range(len(xlens)) ]).type(xlens.dtype)
        ys_tts = xs
        labels = ys_tts.new_zeros(ys_tts.size(0), ys_tts.size(1))
        for i, l in enumerate(ylens_tts):
            labels[i, l - 1:] = 1.0

        return xs_tts, xlens_tts, ys_tts, labels, ylens_tts, spembs

    def asr_to_tts(self, hyps, xlens, xs):
        # convert hyps to xs, xlens to ylens, ys to xs
        # separate yseq from dictionary of nbest_hyps
        ys = [ torch.tensor(y['yseq']) for x in hyps for y in x]
        for i, y_hat in enumerate(ys):
            seq_hat = [self.char_list[int(idx)] for idx in y_hat if int(idx) != -1]
            seq_hat_text = "".join(seq_hat).replace('<space>', ' ')
            logging.info("prediction[%d]: " % i + seq_hat_text)

        xlens_tts = torch.from_numpy(np.array([y.shape[0] for y in ys])).long().to(self.device)
        xlens_tts = sorted(xlens_tts, reverse=True)
        xs_tts = pad_list([y.long() for y in ys], 0).to(self.device)
        reduced_best = len(hyps[0])
        logging.info("nbest is %d", reduced_best)
        xs, xlens = mask_by_length_and_multiply(xs, xlens, 0, reduced_best)
        ylens_tts = xlens
        ys_tts = xs
        labels = ys_tts.new_zeros(ys_tts.size(0), ys_tts.size(1))
        for i, l in enumerate(ylens_tts):
            labels[i, l - 1:] = 1.0

        return xs_tts, xlens_tts, ys_tts, labels, ylens_tts

    def loss_fn_tts(self, after_x, before_x, logits, labels, real_x):
        l1_loss = F.l1_loss(after_x, real_x, reduction='none') + F.l1_loss(before_x, real_x, reduction='none')
        mse_loss = F.mse_loss(after_x, real_x, reduction='none') + F.mse_loss(before_x, real_x, reduction='none')
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=torch.tensor(1.0, device=self.device), reduction='none')
        loss = (l1_loss.mean(2) + mse_loss.mean(2) + bce_loss).mean(1)
        loss = (mse_loss.mean(2)).mean(1)
        return loss

    def loss_fn_asr(self, best_x):
        loss_nll = torch.nn.NLLLoss()
        asr_loss = []
        ys = [torch.tensor(y['yseq']).long() for x in best_x for y in x]
        ys_asr = pad_list([y for y in ys], -1).to(self.device)
        batch = int(ys_asr.size(0) / self.nbest)
        ys_asr = ys_asr.view(batch, self.nbest, -1)
        char_scores = torch.stack(best_x[0][0]['char_score'])
        score = char_scores.mean(0).view(-1)
        return score

    def loss_fn_reinforce(self, asr_loss, tts_loss):
        #loss = 0
        #R = torch.zeros(1, 1)
        # entropy = logp
        entropy = -asr_loss
        reward = tts_loss
        loss = reward * entropy
        #for i in reversed(range(len(reward))):
        #    R = gamma * R + rewards[i]
        #    loss = loss - (asr_loss[i] * (reward) - (0.0001*entropy[i]))
        #loss = loss / len(rewards)
        return loss.mean()

    #@profile
    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        """Main update routine of the CustomUpdater."""
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        asr_optimizer = self.get_optimizer('main')
        tts_optimizer = self.get_optimizer('tts_opt')

        # Get the next batch ( a list of json files)
        batch = train_iter.next()
        #x = self.converter(batch, self.device)
        # Compute the loss at this time step and accumulate it
        if self.ngpu == 0:
            asr_loss = self.model(x).mean() / self.accum_grad
        else:
            # apex does not support torch.nn.DataParallel
            #if (batch[0][1][0][0:5] == np.array([1,1,1,1,1])).all():
            if len(batch[0]) == 3:
                xs_pad, ilens, ys_pad, spembs  = self.converter(batch, self.device)
                x = (xs_pad, ilens, ys_pad)
                if 'espnet.nets.pytorch_backend.e2e_asr_transformer' in self.model.__class__.__module__:
                    fake_loss, best_hyps = data_parallel(self.model, x+(self.iteration, True,), range(self.ngpu))
                else:
                    fake_loss, best_hyps = data_parallel(self.model, x+(True,), range(self.ngpu))
                    if self.text_only:
                        ttsasr_loss = data_parallel(self.model, x+(False,True,), range(self.ngpu)).mean() / self.accum_grad
                # calculate no of nbest and repeat based on it
                #set_requires_grad(self.tts_model, False)
                if self.tts:
                    x_tts = self.random_sampler(best_hyps, ilens, xs_pad, spembs)
                    #tts_loss, after_outs, before_outs, logits, att_ws = self.tts_model(*x_tts+(None,True,))
                    tts_loss, after_outs, before_outs, logits, att_ws = self.tts_model(*x_tts+(True,))
                    #tts_loss = self.loss_fn_tts(after_outs, before_outs, logits, x_tts[4], x_tts[2])
                    #comparison with orig hyp
                    #x_tts_orig = self.random_sampler(x[2], x[1], x[0], spembs)
                    #x_tts_orig[0][x_tts_orig[0] == -1] = 0
                    #tts_loss_j, after_outs_j, before_outs_j, logits_j, att_ws_j = self.tts_model(x_tts_orig[0], x_tts_orig[1], x_tts_orig[2], x_tts_orig[3], x_tts_orig[4], x_tts_orig[5], True)
                    #tts_loss_j = self.loss_fn_tts(after_outs_j, before_outs_j, logits_j, x_tts_orig[3], x_tts_orig[2])
                    #logging.info("true loss is: " + str(tts_loss_j.mean()))
                    logging.info("fake loss is: " + str(fake_loss.mean()))
                    policy_loss = self.policy_rewards(fake_loss, tts_loss)
                    logging.info('tts_loss: ' + str(float(tts_loss.mean())))
                    logging.info('policy_loss: ' + str(float(policy_loss.mean())))
                    asr_loss = policy_loss.mean() / self.accum_grad
                    # asr_loss = tts_loss.mean() / self.accum_grad
                    if self.text_only:
                        asr_loss  = asr_loss + ttsasr_loss
                else:
                    asr_loss = fake_loss.mean() / self.accum_grad
                    logging.info('asr_loss: ' + str(float(asr_loss)))
            else:
                xs_pad, ilens, ys_pad  = self.converter(batch, self.device)
                x = (xs_pad, ilens, ys_pad)
                asr_loss = data_parallel(self.model, x, range(self.ngpu)).mean() / self.accum_grad
                logging.info('asr_sup_loss: ' + str(float(asr_loss)))
        if self.use_apex:
            from apex import amp
            # NOTE: for a compatibility with noam optimizer
            opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
        else:
            asr_loss.backward()
        # gradient noise injection
        if self.grad_noise:
            from espnet.asr.asr_utils import add_gradient_noise
            add_gradient_noise(self.model, self.iteration, duration=100, eta=1.0, scale_factor=0.55)
        asr_loss.detach()  # Truncate the graph

        # update parameters
        self.forward_count += 1
        if self.forward_count != self.accum_grad:
            return
        self.forward_count = 0
        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip_threshold)
        logging.info('ASR grad norm={}'.format(grad_norm))

        #if (batch[0][1][0][0:5] == np.array([1,1,1,1,1])).all():
        if len(batch[0]) == 3:
            tts_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.tts_model.parameters(), self.grad_clip_threshold)
            logging.info('TTS grad norm={}'.format(tts_grad_norm))
            if math.isnan(tts_grad_norm):
                logging.warning('TTS grad norm is nan. Do not update model.')
            else:
                if self.update_tts:
                    tts_optimizer.step()
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            asr_optimizer.step()
        asr_optimizer.zero_grad()
        #if (batch[0][1][0][0:5] == np.array([1,1,1,1,1])).all(): # cheap trick by BMK
        if len(batch[0]) == 3: # cheap trick by BMK
            tts_optimizer.zero_grad()

    def update(self):
        self.update_core()
        # #iterations with accum_grad > 1
        # Ref.: https://github.com/espnet/espnet/issues/777
        if self.forward_count == 0:
            self.iteration += 1


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """
    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][-1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][-1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    if args.enc_init is not None or args.dec_init is not None:
        model = load_trained_modules(idim, odim, args)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(idim, odim, args)
        if args.asr_init is not None:
            #model, asr_args = load_trained_model(args.asr_init)
            asr_idim, asr_odim, asr_args = get_model_conf(
            args.asr_init, os.path.join(os.path.dirname(args.asr_init), 'model.json'))
            logging.info('reading model parameters from ' + args.asr_init)
            torch_load(args.asr_init, model)
            #if args.freeze == "ctc":
            #    model, size = freeze_parameters(model, 16)
            #    logging.info("no of parameters frozen are: " + str(size))

    if args.tts_init is not None:
        tts_model, tts_args = load_trained_model(args.tts_init)
    assert isinstance(model, ASRInterface)
    assert isinstance(tts_model, TTSInterface)

    subsampling_factor = model.subsample[0]

    if args.rnnlm is not None:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        model.rnnlm = rnnlm

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
        model_conf = args.outdir + '/model.json'
    else:
        model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)),
                           indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter
    tts_reporter = tts_model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        tts_model = torch.nn.DataParallel(tts_model, device_ids=list(range(args.ngpu)))
        if args.batch_size != 0:
            logging.info('batch size is automatically increased (%d -> %d)' % (
                args.batch_size, args.batch_size * args.ngpu))
            args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    tts_model = tts_model.to(device=device, dtype=dtype)

    # Setup an optimizer
    #params = [asr_model.parameters(), tts_model.parameters()]
    if args.opt == 'adadelta':
        asr_optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
        tts_optimizer = torch.optim.Adadelta(tts_model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        asr_optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0,
                                        dampening=0, weight_decay=args.weight_decay, nesterov=False)
        tts_optimizer = torch.optim.SGD(tts_model.parameters(), lr=0.05, momentum=0,
                                        dampening=0, weight_decay=args.weight_decay, nesterov=False)
    elif args.opt == 'adam':
        asr_optimizer = torch.optim.Adam(model.parameters(),
            weight_decay=args.weight_decay)
        tts_optimizer = torch.optim.Adam(tts_model.parameters(),
            weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        asr_optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
        tts_optimizer = torch.optim.Adadelta(tts_model.parameters(), rho=0.95, eps=args.eps,
            weight_decay=args.weight_decay)

    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # setup apex.amp
    if args.train_dtype in ("O0", "O1", "O2", "O3"):
        try:
            from apex import amp
        except ImportError as e:
            logging.error(f"You need to install apex for --train-dtype {args.train_dtype}. "
                          "See https://github.com/NVIDIA/apex#linux")
            raise e
        if args.opt == 'noam':
            model, optimizer.optimizer = amp.initialize(model, optimizer.optimizer, opt_level=args.train_dtype)
        else:
            model, optimizer = amp.initialize(model, optimizer, opt_level=args.train_dtype)
        use_apex = True
    else:
        use_apex = False

    # FIXME: TOO DIRTY HACK
    setattr(asr_optimizer, "target", reporter)
    setattr(tts_optimizer, "target", tts_reporter)
    setattr(asr_optimizer, "serialize", lambda s: reporter.serialize(s))
    setattr(tts_optimizer, "serialize", lambda s: tts_reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(subsampling_factor=subsampling_factor, dtype=dtype)

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.train_unpaired_json, 'rb') as f:
        train_unpaired_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    use_sortagrad = args.sortagrad == -1 or args.sortagrad > 0
    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=use_sortagrad,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          iaxis=0, oaxis=0)
    train_unp = make_batchset(train_unpaired_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          shortest_first=use_sortagrad,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          iaxis=0, oaxis=0)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches,
                          min_batch_size=args.ngpu if args.ngpu > 1 else 1,
                          count=args.batch_count,
                          batch_bins=args.batch_bins,
                          batch_frames_in=args.batch_frames_in,
                          batch_frames_out=args.batch_frames_out,
                          batch_frames_inout=args.batch_frames_inout,
                          iaxis=0, oaxis=0)

    load_tr = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': True}  # Switch the mode of preprocessing
    )
    load_cv = LoadInputsAndTargets(
        mode='asr', load_output=True, preprocess_conf=args.preprocess_conf,
        preprocess_args={'train': False}  # Switch the mode of preprocessing
    )

    # merge paired and unpaired datasets
    train, _ = merge_batchsets(train, train_unp, repeat=False)
    # hack to make batchsize argument as 1
    # actual bathsize is included in a list
    if args.n_iter_processes > 0:
        train_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(train, load_tr),
            batch_size=1, n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20,
            shuffle=not use_sortagrad)
        valid_iter = ToggleableShufflingMultiprocessIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False,
            n_processes=args.n_iter_processes, n_prefetch=8, maxtasksperchild=20)
    else:
        train_iter = ToggleableShufflingSerialIterator(
            TransformDataset(train, load_tr),
            batch_size=1, shuffle=not use_sortagrad)
        valid_iter = ToggleableShufflingSerialIterator(
            TransformDataset(valid, load_cv),
            batch_size=1, repeat=False, shuffle=False)
    # Set up a trainer
    beta = 1 - args.alpha
    updater = CycleUpdater(
                models = (model, tts_model),
        iterator = {'main': train_iter}, # 'main_pair': train_iter},
                optimizer = {'main': asr_optimizer, 'tts_opt': tts_optimizer},
                params = {'alpha': args.alpha,
                  'beta': beta,
                  'tts': args.tts,
                  'device': device,
                  'ngpu': args.ngpu,
                  'char_list': args.char_list,
                  'beam_size': args.beam_size,
                  'nbest': args.nbest,
                  'grad_clip_threshold': args.grad_clip,
                  'grad_noise': args.grad_noise,
                  'accum_grad': args.accum_grad,
                  'use_apex': use_apex,
                  'update_tts': args.update_tts,
                  'converter': converter,
                  'speech_only': args.speech_only,
                  'text_only': args.text_only,
                  })

    trainer = training.Trainer(
        updater, (args.epochs, 'epoch'), out=args.outdir)

    if use_sortagrad:
        trainer.extend(ShufflingEnabler([train_iter]),
                       trigger=(args.sortagrad if args.sortagrad != -1 else args.epochs, 'epoch'))

    # Resume from a snapshot
    if args.resume:
        logging.info('resumed from %s' % args.resume)
        torch_resume(args.resume, trainer)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(CustomEvaluator(model, valid_iter, reporter, converter, device, args.ngpu))

    # Save attention weight each epoch
    if args.num_save_attention > 0 and args.mtlalpha != 1.0:
        data = sorted(list(valid_json.items())[:args.num_save_attention],
                      key=lambda x: int(x[1]['input'][0]['shape'][1]), reverse=True)
        if hasattr(model, "module"):
            att_vis_fn = model.module.calculate_all_attentions
            plot_class = model.module.attention_plot_class
        else:
            att_vis_fn = model.calculate_all_attentions
            plot_class = model.attention_plot_class
        att_reporter = plot_class(
            att_vis_fn, data, args.outdir + "/att_ws",
            converter=converter, transform=load_cv, device=device)
        trainer.extend(att_reporter, trigger=(1, 'epoch'))
    else:
        att_reporter = None

    # Make a plot for training and validation values
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss',
                                          'main/loss_ctc', 'validation/main/loss_ctc',
                                          'main/loss_att', 'validation/main/loss_att'],
                                         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/acc', 'validation/main/acc'],
                                         'epoch', file_name='acc.png'))
    trainer.extend(extensions.PlotReport(['main/cer_ctc', 'validation/main/cer_ctc'],
                                         'epoch', file_name='cer.png'))

    # Save best models
    trainer.extend(snapshot_object(model, 'model.loss.best'),
                   trigger=training.triggers.MinValueTrigger('validation/main/loss'))
    if mtl_mode != 'ctc':
        trainer.extend(snapshot_object(model, 'model.acc.best'),
                       trigger=training.triggers.MaxValueTrigger('validation/main/acc'))

    # save snapshot which contains model and optimizer states
    trainer.extend(torch_snapshot(), trigger=(1, 'epoch'))

    # epsilon decay in the optimizer
    if args.opt == 'adadelta':
        if args.criterion == 'acc' and mtl_mode != 'ctc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(adadelta_eps_decay(args.eps_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
    elif args.opt == 'sgd':
        if args.criterion == 'acc' and mtl_mode != 'ctc':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.acc.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
            trainer.extend(sgd_lr_decay(args.lr_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/acc',
                               lambda best_value, current_value: best_value > current_value))
        elif args.criterion == 'loss':
            trainer.extend(restore_snapshot(model, args.outdir + '/model.loss.best', load_fn=torch_load),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))
            trainer.extend(sgd_lr_decay(args.lr_decay),
                           trigger=CompareValueTrigger(
                               'validation/main/loss',
                               lambda best_value, current_value: best_value < current_value))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=(args.report_interval_iters, 'iteration')))
    report_keys = ['epoch', 'iteration', 'main/loss', 'main/loss_ctc', 'main/loss_att',
                   'validation/main/loss', 'validation/main/loss_ctc', 'validation/main/loss_att',
                   'main/acc', 'validation/main/acc', 'main/cer_ctc', 'validation/main/cer_ctc',
                   'elapsed_time']
    if args.opt == 'adadelta':
        trainer.extend(extensions.observe_value(
            'eps', lambda trainer: trainer.updater.get_optimizer('main').param_groups[0]["eps"]),
            trigger=(args.report_interval_iters, 'iteration'))
        report_keys.append('eps')
    if args.report_cer:
        report_keys.append('validation/main/cer')
    if args.report_wer:
        report_keys.append('validation/main/wer')
    trainer.extend(extensions.PrintReport(
        report_keys), trigger=(args.report_interval_iters, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=args.report_interval_iters))
    set_early_stop(trainer, args)

    if args.tensorboard_dir is not None and args.tensorboard_dir != "":
        trainer.extend(TensorboardLogger(SummaryWriter(args.tensorboard_dir), att_reporter),
                       trigger=(args.report_interval_iters, "iteration"))
    # Run the training
    trainer.run()
    check_early_stop(trainer, args.epochs)
