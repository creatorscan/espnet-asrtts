from distutils.version import LooseVersion
import logging
import random
import six

import numpy as np
import torch
import torch.nn.functional as F

from argparse import Namespace

from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH
from espnet.nets.e2e_asr_common import end_detect

from espnet.nets.pytorch_backend.rnn.attentions import att_to_numpy

from espnet.nets.pytorch_backend.nets_utils import mask_by_length
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.scorer_interface import ScorerInterface
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss

MAX_DECODER_OUTPUT = 5
CTC_SCORING_RATIO = 1.5


class Decoder(torch.nn.Module, ScorerInterface):
    """Decoder module

    :param int eprojs: # encoder projection units
    :param int odim: dimension of outputs
    :param str dtype: gru or lstm
    :param int dlayers: # decoder layers
    :param int dunits: # decoder units
    :param int sos: start of sequence symbol id
    :param int eos: end of sequence symbol id
    :param torch.nn.Module att: attention module
    :param int verbose: verbose level
    :param list char_list: list of character strings
    :param ndarray labeldist: distribution of label smoothing
    :param float lsm_weight: label smoothing weight
    :param float sampling_probability: scheduled sampling probability
    :param float dropout: dropout rate
    :param float context_residual: if True, use context vector for token generation
    :param float replace_sos: use for multilingual (speech/text) translation
    """

    def __init__(self, eprojs, odim, dtype, dlayers, dunits, sos, eos, att, verbose=0,
                 char_list=None, labeldist=None, lsm_weight=0., sampling_probability=0.0,
                 dropout=0.0, context_residual=False, replace_sos=False):

        torch.nn.Module.__init__(self)
        self.dtype = dtype
        self.dunits = dunits
        self.dlayers = dlayers
        self.context_residual = context_residual
        self.embed = torch.nn.Embedding(odim, dunits)
        self.dropout_emb = torch.nn.Dropout(p=dropout)

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()
        self.decoder += [
            torch.nn.LSTMCell(dunits + eprojs, dunits) if self.dtype == "lstm" else torch.nn.GRUCell(dunits + eprojs,
                                                                                                     dunits)]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in six.moves.range(1, self.dlayers):
            self.decoder += [
                torch.nn.LSTMCell(dunits, dunits) if self.dtype == "lstm" else torch.nn.GRUCell(dunits, dunits)]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]
            # NOTE: dropout is applied only for the vertical connections
            # see https://arxiv.org/pdf/1409.2329.pdf
        self.ignore_id = -1

        if context_residual:
            self.output = torch.nn.Linear(dunits + eprojs, odim)
        else:
            self.output = torch.nn.Linear(dunits, odim)

        self.loss = None
        self.att = att
        self.dunits = dunits
        self.sos = sos
        self.eos = eos
        self.odim = odim
        self.verbose = verbose
        self.char_list = char_list
        # for label smoothing
        self.labeldist = labeldist
        self.vlabeldist = None
        self.lsm_weight = lsm_weight
        self.sampling_probability = sampling_probability
        self.dropout = dropout
        # for multilingual translation
        self.replace_sos = replace_sos

        self.logzero = -10000000000.0

    def zero_state(self, hs_pad):
        return hs_pad.new_zeros(hs_pad.size(0), self.dunits)

    def rnn_forward(self, ey, z_list, c_list, z_prev, c_prev):
        if self.dtype == "lstm":
            z_list[0], c_list[0] = self.decoder[0](ey, (z_prev[0], c_prev[0]))
            for l in six.moves.range(1, self.dlayers):
                z_list[l], c_list[l] = self.decoder[l](
                    self.dropout_dec[l - 1](z_list[l - 1]), (z_prev[l], c_prev[l]))
        else:
            z_list[0] = self.decoder[0](ey, z_prev[0])
            for l in six.moves.range(1, self.dlayers):
                z_list[l] = self.decoder[l](self.dropout_dec[l - 1](z_list[l - 1]), z_prev[l])
        return z_list, c_list

    def forward(self, hs_pad, hlens, ys_pad, strm_idx=0, tgt_lang_ids=None):
        """Decoder forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :param int strm_idx: stream index indicates the index of decoding stream.
        :param torch.Tensor tgt_lang_ids: batch of target language id tensor (B, 1)
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        # attention index for the attention module
        # in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.att) - 1)

        # hlen should be list of integer
        hlens = list(map(int, hlens))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        if self.replace_sos:
            ys_in = [torch.cat([idx, y], dim=0) for idx, y in zip(tgt_lang_ids, ys)]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get dim, length info
        batch = ys_out_pad.size(0)
        olength = ys_out_pad.size(1)
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

        # initialization
        c_list = [self.zero_state(hs_pad)]
        z_list = [self.zero_state(hs_pad)]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad))
            z_list.append(self.zero_state(hs_pad))

        att_w = None
        z_all = []
        self.att[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim
        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att[att_idx](hs_pad, hlens, self.dropout_dec[0](z_list[0]), att_w)
            if i > 0 and random.random() < self.sampling_probability:
                logging.info(' scheduled sampling ')
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach().cpu(), axis=1)
                z_out = self.dropout_emb(self.embed(to_device(self, z_out)))
                ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
                ey = torch.cat((eys[:, i, :], att_c), dim=-1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            if self.context_residual:
                z_all.append(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))  # utt x (zdim + hdim)
            else:
                z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)

        z_all = torch.stack(z_all, dim=1).view(batch * olength, -1)
        # compute loss
        y_all = self.output(z_all)
        if LooseVersion(torch.__version__) < LooseVersion('1.0'):
            reduction_str = 'elementwise_mean'
        else:
            reduction_str = 'mean'
        self.loss = F.cross_entropy(y_all, ys_out_pad.view(-1),
                                    ignore_index=self.ignore_id,
                                    reduction=reduction_str)
        # -1: eos, which is removed in the loss computation
        self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        #self.loss = self.kl_loss(y_all.view(batch, olength, -1), ys_out_pad)
        acc = th_accuracy(y_all, ys_out_pad, ignore_label=self.ignore_id)
        logging.info('att loss:' + ''.join(str(self.loss.item()).split('\n')))

        # compute perplexity
        ppl = np.exp(self.loss.item() * np.mean([len(x) for x in ys_in]) / np.sum([len(x) for x in ys_in]))

        # show predicted character sequence for debug
        if self.verbose > 0 and self.char_list is not None:
            ys_hat = y_all.view(batch, olength, -1)
            ys_true = ys_out_pad
            for (i, y_hat), y_true in zip(enumerate(ys_hat.detach().cpu().numpy()),
                                          ys_true.detach().cpu().numpy()):
                if i == MAX_DECODER_OUTPUT:
                    break
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("groundtruth[%d]: " % i + seq_true)
                logging.info("prediction [%d]: " % i + seq_hat)

        if self.labeldist is not None:
            if self.vlabeldist is None:
                self.vlabeldist = to_device(self, torch.from_numpy(self.labeldist))
            loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
            self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg

        return self.loss, acc, ppl

    def recognize_beam(self, h, lpz, recog_args, char_list, rnnlm=None, strm_idx=0):
        """beam search implementation

        :param torch.Tensor h: encoder hidden state (T, eprojs)
        :param torch.Tensor lpz: ctc log softmax output (T, odim)
        :param Namespace recog_args: argument Namespace containing options
        :param char_list: list of character strings
        :param torch.nn.Module rnnlm: language module
        :param int strm_idx: stream index for speaker parallel attention in multi-speaker case
        :return: N-best decoding results
        :rtype: list of dicts
        """
        logging.info('input lengths: ' + str(h.size(0)))
        att_idx = min(strm_idx, len(self.att) - 1)
        # initialization
        c_list = [self.zero_state(h.unsqueeze(0))]
        z_list = [self.zero_state(h.unsqueeze(0))]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(h.unsqueeze(0)))
            z_list.append(self.zero_state(h.unsqueeze(0)))
        a = None
        self.att[att_idx].reset()  # reset pre-computation of h

        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprate sos
        if self.replace_sos and recog_args.tgt_lang:
            y = char_list.index(recog_args.tgt_lang)
        else:
            y = self.sos
        logging.info('<sos> index: ' + str(y))
        logging.info('<sos> mark: ' + char_list[y])
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list,
                   'z_prev': z_list, 'a_prev': a, 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'c_prev': c_list, 'z_prev': z_list, 'a_prev': a}
        if lpz is not None:
            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, np)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i]
                ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
                ey.unsqueeze(0)
                att_c, att_w = self.att[att_idx](h.unsqueeze(0), [h.size(0)],
                                                 self.dropout_dec[0](hyp['z_prev'][0]), hyp['a_prev'])
                ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
                z_list, c_list = self.rnn_forward(ey, z_list, c_list, hyp['z_prev'], hyp['c_prev'])

                # get nbest local scores and their ids
                if self.context_residual:
                    logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
                else:
                    logits = self.output(self.dropout_dec[-1](z_list[-1]))
                local_att_scores = F.log_softmax(logits, dim=1)
                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)
                
                for j in six.moves.range(beam):
                    new_hyp = {}
                    # [:] is needed!
                    new_hyp['z_prev'] = z_list[:]
                    new_hyp['c_prev'] = c_list[:]
                    new_hyp['a_prev'] = att_w[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypotheses: ' + str(len(hyps)))
            logging.debug(
                'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last position in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypotheses to a final list, and removed them from current hypotheses
            # (this will be a problem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remaining hypotheses: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            for hyp in hyps:
                logging.debug(
                    'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypotheses: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheses
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy because Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize_beam(h, lpz, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        # remove sos
        return nbest_hyps

    def recognize_beam_batch(self, h, hlens, lpz, recog_args, char_list, rnnlm=None,
                             normalize_score=True, strm_idx=0, tgt_lang_ids=None):
        logging.info('input lengths: ' + str(h.size(1)))
        att_idx = min(strm_idx, len(self.att) - 1)
        h = mask_by_length(h, hlens, 0.0)

        # search params
        batch = len(hlens)
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight
        ctc_margin = 0 #recog_args.ctc_window_margin

        n_bb = batch * beam
        pad_b = to_device(self, torch.arange(batch) * beam).view(-1, 1)

        max_hlen = int(max(hlens))
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialization
        c_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        c_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        vscores = to_device(self, torch.zeros(batch, beam))

        a_prev = None
        rnnlm_state = None
        ctc_scorer = None
        ctc_state = None

        self.att[att_idx].reset()  # reset pre-computation of h

        if self.replace_sos and recog_args.tgt_lang:
            logging.info('<sos> index: ' + str(char_list.index(recog_args.tgt_lang)))
            logging.info('<sos> mark: ' + recog_args.tgt_lang)
            yseq = [[char_list.index(recog_args.tgt_lang)] for _ in six.moves.range(n_bb)]
        elif tgt_lang_ids is not None:
            # NOTE: used for evaluation during training
            yseq = [[tgt_lang_ids[b // recog_args.beam_size]] for b in six.moves.range(n_bb)]
        else:
            logging.info('<sos> index: ' + str(self.sos))
            logging.info('<sos> mark: ' + char_list[self.sos])
            yseq = [[self.sos] for _ in six.moves.range(n_bb)]

        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = hlens.repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
        exp_hlens = exp_hlens.view(-1).tolist()
        exp_h = h.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        exp_h = exp_h.view(n_bb, h.size()[1], h.size()[2])

        if lpz is not None:
            scoring_ratio = CTC_SCORING_RATIO if att_weight > 0.0 and not lpz.is_cuda else 0
            ctc_scorer = CTCPrefixScoreTH(lpz, hlens, 0, self.eos, beam,
                                          scoring_ratio, margin=ctc_margin)
            #ctc_state = ctc_scorer.initial_state()
        char_score = []
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            vy = to_device(self, torch.LongTensor(self._get_last_yseq(yseq)))
            ey = self.dropout_emb(self.embed(vy))
            att_c, att_w = self.att[att_idx](exp_h, exp_hlens, self.dropout_dec[0](z_prev[0]), a_prev)
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_prev, c_prev)
            if self.context_residual:
                logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
            else:
                logits = self.output(self.dropout_dec[-1](z_list[-1]))
            local_scores = att_weight * F.log_softmax(logits, dim=1)

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_state, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores

            # ctc
            if ctc_scorer:
                att_w_ = att_w if isinstance(att_w, torch.Tensor) else att_w[0]
                ctc_state, local_ctc_scores = ctc_scorer(yseq, ctc_state, local_scores, att_w_)
                local_scores = local_scores + ctc_weight * local_ctc_scores
            
            local_scores = local_scores.view(batch, beam, self.odim)
            if i == 0:
                local_scores[:, 1:, :] = self.logzero

            # accumulate scores
            eos_vscores = local_scores[:, :, self.eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.odim)
            vscores[:, :, self.eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, -1)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)
            accum_odim_ids = torch.fmod(accum_best_ids, self.odim).view(-1).data.cpu().tolist()
            accum_padded_beam_ids = (torch.div(accum_best_ids, self.odim) + pad_b).view(-1).data.cpu().tolist()
            char_score.append(-accum_best_scores)
            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids)
            yseq = self._append_ids(yseq, accum_odim_ids)
            vscores = accum_best_scores
            vidx = to_device(self, torch.LongTensor(accum_padded_beam_ids))

            if isinstance(att_w, torch.Tensor):
                a_prev = torch.index_select(att_w.view(n_bb, *att_w.shape[1:]), 0, vidx)
            elif isinstance(att_w, list):
                # handle the case of multi-head attention
                a_prev = [torch.index_select(att_w_one.view(n_bb, -1), 0, vidx) for att_w_one in att_w]
            else:
                # handle the case of location_recurrent when return is a tuple
                a_prev_ = torch.index_select(att_w[0].view(n_bb, -1), 0, vidx)
                h_prev_ = torch.index_select(att_w[1][0].view(n_bb, -1), 0, vidx)
                c_prev_ = torch.index_select(att_w[1][1].view(n_bb, -1), 0, vidx)
                a_prev = (a_prev_, (h_prev_, c_prev_))
            z_prev = [torch.index_select(z_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]
            c_prev = [torch.index_select(c_list[li].view(n_bb, -1), 0, vidx) for li in range(self.dlayers)]

            if rnnlm:
                rnnlm_state = self._index_select_lm_state(rnnlm_state, 0, vidx)
            if ctc_scorer:
                ctc_state = ctc_scorer.index_select_state(ctc_state, accum_best_ids)

            # pick ended hyps
            if i > minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in six.moves.range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in six.moves.range(beam):
                        #if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                        yk = y_prev[k][:]
                        #yk.append(self.eos)
                        if len(yk) < hlens[samp_i]:
                            _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                            _score = _vscore.data.cpu().numpy()
                            ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score, 'char_score': char_score})
                        k = k + 1

            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i)
                           for samp_i in six.moves.range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

        torch.cuda.empty_cache()

        dummy_hyps = [{'yseq': [self.sos, self.eos], 'score': np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in six.moves.range(batch)]
        if normalize_score:
            for samp_i in six.moves.range(batch):
                for x in ended_hyps[samp_i]:
                    x['score'] /= len(x['yseq'])

        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'],
                             reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)]
                      for samp_i in six.moves.range(batch)]
        return nbest_hyps

    def generate_beam_batch(self, h, hlens, lpz, recog_args, char_list, rnnlm=None,
                             normalize_score=True, strm_idx=0, tgt_lang_ids=None,
                                sampling='ramon', temp=30):
        logging.info('input lengths: ' + str(h.size(1)))
        att_idx = min(strm_idx, len(self.att) - 1)
        h = mask_by_length(h, hlens, 0.0)

        # search params
        batch = len(hlens)
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight
        ctc_margin = 0 #recog_args.ctc_window_margin

        n_bb = batch * beam
        pad_b = to_device(self, torch.arange(batch) * beam).view(-1, 1)

        max_hlen = int(max(hlens))
        if recog_args.maxlenratio == 0:
            maxlen = max_hlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_hlen))
        minlen = int(recog_args.minlenratio * max_hlen)
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialization
        c_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        c_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        vscores = to_device(self, torch.zeros(batch, beam))

        a_prev = None
        rnnlm_state = None
        ctc_scorer = None
        ctc_state = None

        self.att[att_idx].reset()  # reset pre-computation of h

        if self.replace_sos and recog_args.tgt_lang:
            logging.info('<sos> index: ' + str(char_list.index(recog_args.tgt_lang)))
            logging.info('<sos> mark: ' + recog_args.tgt_lang)
            yseq = [[char_list.index(recog_args.tgt_lang)] for _ in six.moves.range(n_bb)]
        elif tgt_lang_ids is not None:
            # NOTE: used for evaluation during training
            yseq = [[tgt_lang_ids[b // recog_args.beam_size]] for b in six.moves.range(n_bb)]
        else:
            logging.info('<sos> index: ' + str(self.sos))
            logging.info('<sos> mark: ' + char_list[self.sos])
            yseq = [[self.sos] for _ in six.moves.range(n_bb)]

        accum_odim_ids = [self.sos for _ in six.moves.range(n_bb)]
        stop_search = [False for _ in six.moves.range(batch)]
        nbest_hyps = [[] for _ in six.moves.range(batch)]
        ended_hyps = [[] for _ in range(batch)]

        exp_hlens = hlens.repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
        exp_hlens = exp_hlens.view(-1).tolist()
        exp_h = h.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        exp_h = exp_h.view(n_bb, h.size()[1], h.size()[2])

        if lpz is not None:
            scoring_ratio = CTC_SCORING_RATIO if att_weight > 0.0 and not lpz.is_cuda else 0
            ctc_scorer = CTCPrefixScoreTH(lpz, hlens, 0, self.eos, beam,
                                          scoring_ratio, margin=ctc_margin)
        y_all = []
        char_scores = []
        char_ids = []
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))
            if i == 0:
                vy = to_device(self, torch.LongTensor(self._get_last_yseq(yseq)))
            else:
                vy = to_device(self, char_ids[-1].long())
            ey = self.dropout_emb(self.embed(vy))
            att_c, att_w = self.att[att_idx](exp_h, exp_hlens, self.dropout_dec[0](z_prev[0]), a_prev)
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_prev, c_prev)
            if self.context_residual:
                logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
            else:
                logits = self.output(self.dropout_dec[-1](z_list[-1]))
            local_scores = att_weight * F.log_softmax(logits, dim=1)
            # prob_scores = att_weight * F.softmax(logits/temp, dim=1)
            # y_all.append(logits)

            # rnnlm
            if rnnlm:
                rnnlm_state, local_lm_scores = rnnlm.buff_predict(rnnlm_state, vy, n_bb)
                local_scores = local_scores + recog_args.lm_weight * local_lm_scores
            # ctc
            #if ctc_scorer:
            #    att_w_ = att_w if isinstance(att_w, torch.Tensor) else att_w[0]
            #    ctc_state, local_ctc_scores = ctc_scorer(yseq, ctc_state, local_scores, att_w_)
            #    local_scores = local_scores + ctc_weight * local_ctc_scores
            local_scores = local_scores.view(batch* beam, self.odim)
            if sampling == 'categorical':
                #logging.info('sampling from categorical dist')
                m = torch.distributions.Categorical(logits=local_scores)
                random_ids = m.sample() # action
                random_scores = m.log_prob(random_ids) # log_prob
            elif sampling == 'multinomial':
                #logging.info('sampling from Ramon')
                #indices = [np.array(range(self.odim), dtype=np.int32)] * int(batch* beam)
                #indices_to_remove = logits < torch.topk(logits, 10)[0][..., -1, None]
                #logits[indices_to_remove] = -float('Inf')
                #probs = F.softmax(logits, dim=-1)
                #probs, topk_ind = torch.topk(local_scores, 20)
                prob_scores = att_weight * F.softmax(logits, dim=1)
                ids = torch.multinomial(prob_scores, num_samples=1)
                random_scores = torch.gather(local_scores, dim=1, index=ids.detach()) # log_prob
                random_ids = ids.squeeze(1)
            elif sampling == 'temperature':
                prob_scores = att_weight * F.softmax(logits/temp, dim=1)
                m = torch.multinomial(prob_scores, num_samples=1)
                random_scores = torch.gather(local_scores, dim=1, index=ids.detach()) # log_prob
                random_ids = ids.squeeze(1)
            elif sampling == 'topk':
                k_scores, k_indices = torch.topk(local_scores, 20)
                indices_to_remove = logits < k_scores
                #logits[indices_to_remove] = filter_value
                #local_scores[local_scores < k_scores] = -float('Inf')
            elif sampling == 'topp':
                sorted_logits, sorted_indices = torch.sort(local_scores, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > 0.5
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = [sorted_indices[i][sorted_indices_to_remove[i]] for i in range(local_scores.size(0))]
                for i in range(local_scores.size(0)):
                    local_scores[i][indices_to_remove[i]] = -float('Inf')
                ids = torch.multinomial(F.softmax(local_scores, dim=-1), num_samples=1)
                random_scores = torch.gather(local_scores, dim=1, index=ids.detach()) # log_prob
                random_ids = ids.squeeze(1)
            #elif sampling == 'ramon':
            #    for i in range(local_scores.size(0)):
            #        ids[i] = np.random.choice(indices[j], 1, p=sy[j])  # or argmax in some cases

            # pick ended hyps
            char_scores.append(random_scores)
            char_ids.append(random_ids)

        char_ids = torch.stack(char_ids).view(batch * beam, -1)
        char_scores = torch.stack(char_scores).view(batch, beam, -1)
        return char_ids, char_scores

    def generate(self, exp_h, exp_hlens, exp_ys, recog_args, strm_idx=0, topk=0, maxlenratio=1.0, minlenratio=0.3, rnnlm=None):
        '''Decoder generate

        :param hs:
        :return:
        '''
        # get dim, length info
        # initialization
        self.loss = None
        logzero = -1.0e+10
        batch = len(exp_hlens)
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight
        ctc_margin = 0 #recog_args.ctc_window_margin
        batch = exp_ys.size(0)
        olength = exp_ys.size(1)
        sos = exp_ys[0].new([self.sos])
        exp_ys_list = [y[y != self.ignore_id] for y in exp_ys]  # parse padded ys
        ys_in = [torch.cat([sos, y], dim=0) for y in exp_ys_list]
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)

        n_bb = batch * beam
        pad_b = to_device(self, torch.arange(batch) * beam).view(-1, 1)
        hlen = exp_hlens.repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
        hlen = exp_hlens.view(-1).tolist()
        hpad = exp_h.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        hpad = exp_h.view(n_bb, exp_h.size(1), exp_h.size(2))
        ys =  ys_in_pad.unsqueeze(1).repeat(1, beam, 1, 1)
        ys = ys.view(n_bb, ys_in_pad.size(1))

        n_samples = len(hlen)
        # initialization
        c_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_prev = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        c_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]
        z_list = [to_device(self, torch.zeros(n_bb, self.dunits)) for _ in range(self.dlayers)]

        #c_list = [self.zero_state(hpad)]
        #z_list = [self.zero_state(hpad)]
        #for l in six.moves.range(1, self.dlayers):
        #    c_list.append(self.zero_state(hpad))
        #    z_list.append(self.zero_state(hpad))
        a_prev = None
        att_idx = min(strm_idx, len(self.att) - 1)
        self.att[att_idx].reset()  # reset pre-computation of h

        # preprate sos
        if maxlenratio == 0:
            maxlen = int(max(hlen))
        else:
            # maxlen >= 1
            maxlen = max(1, int(maxlenratio * int(max(hlen))))
        minlen = int(minlenratio * int(min(hlen)))
        odim = self.eos + 1
        # prepare the first label <sos>
        y = to_device(self, torch.from_numpy(np.full(n_samples, self.sos, dtype=np.int64)))
        indices = [np.array(range(odim), dtype=np.int32)] * n_samples
        y_gen = np.full((maxlen, n_samples), self.ignore_id, dtype=np.int64)
        not_ended = np.array([True] * n_samples, dtype=np.bool_)
        # make a mask to avoid <blank>:0, <unk>:1, and sentence end prediction
        suppress_mask = np.zeros((1, odim), dtype=np.float32)
        suppress_mask[0, (0, 1, self.eos)] = logzero
        suppress_mask = to_device(self, torch.from_numpy(suppress_mask))
        suppress_mask_without_eos = suppress_mask.clone()
        suppress_mask_without_eos[0, self.eos] = 0.
        y_lens = np.zeros(n_samples, dtype=np.int64)
        loss_list = []
        y_all = []
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim
        for i in six.moves.range(olength):
            if i > 0:
                yg = y_gen[i - 1]
                yg[yg == -1] = 0
                y = to_device(self, torch.from_numpy(yg))
            ey = self.embed(y)  # utt x zdim
            att_c, att_w = self.att[att_idx](hpad, hlen, self.dropout_dec[0](z_prev[0]), a_prev)
            if i > 0 and random.random() < 1.0:
                yg = y_gen[i - 1]
                yg[yg == -1] = 0
                y = to_device(self, torch.from_numpy(yg))
                ey = self.embed(y)  # utt x zdim
                ey = torch.cat((ey, att_c), dim=1)  # n_samples x (zdim + hdim)
            else:
                ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_prev, c_prev)
            #if i == minlen:  # exclude <eos> while sequence is short
            #    suppress_mask = suppress_mask_without_eos
            if self.context_residual:
                oy = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))# + suppress_mask
            else:
                oy = self.output(self.dropout_dec[-1](z_list[-1]))# + suppress_mask
            #local_scores = att_weight * F.log_softmax(logits, dim=1)

            #oy = self.output(z_list[-1]) + suppress_mask
            #if nbest is None:
            if 0 < topk < odim:
                topk_logits, topk_indices = torch.topk(oy, topk, dim=1)
                sy = F.softmax(topk_logits, dim=1).data.cpu().numpy()
                indices = topk_indices.data.cpu().numpy()
            else:
                sy = F.softmax(oy, dim=1).data.cpu().numpy()

            if i < maxlen - 1:
                for j in six.moves.range(n_samples):
                    if not_ended[j]:
                        y_gen[i, j] = np.argmax(sy[j])
                        #y_gen[i, j] = np.random.choice(indices[j], 1, p=sy[j])  # or argmax in some cases
                not_ended &= y_gen[i, :] != self.eos
                y_lens[y_gen[i, :] == self.eos] = i + 1
            else:
                y_gen[i, not_ended] = self.eos
                y_lens[not_ended] = i + 1
            del sy
            t = to_device(self, torch.from_numpy(y_gen[i]))
            ce_loss = F.cross_entropy(oy, t, ignore_index=self.ignore_id, reduction='none')
            loss_list.append(ce_loss)

            # y_all for computing KL-divergence loss
            # y_all.append(self.output(z_list[-1]))

            # all ended -> break
            if np.sum(not_ended) == 0:
                break
        # loss array needs to be masked by 0 to exclude indeterminate valuse for igored_id
        #masked_loss = mask_by_length(torch.stack(loss_list).transpose(1, 0),
        #                             y_lens, fill=0.0)
        sample_loss = torch.sum(torch.stack(loss_list).transpose(1, 0), dim=-1)
        # show predicted character sequence for debug
        y_list = []
        y_gen = y_gen.transpose(1, 0)
        y_gen[y_gen == -1] = 0
        for i in six.moves.range(n_samples):
            y_seq = np.array(y_gen[i, :y_lens[i] - 1], dtype=np.int32)
            y_list.append(torch.from_numpy(y_seq))
            if self.verbose > 0 and self.char_list is not None:
                y_str = "".join([self.char_list[int(idx)] for idx in y_seq])
                logging.info("generation[%d]: %.4f " % (i, sample_loss.data[i]) + y_str)
        return -sample_loss, y_list

    def generate_forward(self, hs_pad, hlens, ys_pad, recog_args, strm_idx=0, topk=0, maxlenratio=0.8, minlenratio=0.3, rnnlm=None, tgt_lang_ids=None, oracle_length=False):
        """Decoder forward

        :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor hlens: batch of lengths of hidden state sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :param int strm_idx: stream index indicates the index of decoding stream.
        :param torch.Tensor tgt_lang_ids: batch of target language id tensor (B, 1)
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy
        :rtype: float
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        # attention index for the attention module
        # in SPA (speaker parallel attention), att_idx is used to select attention module. In other cases, it is 0.
        att_idx = min(strm_idx, len(self.att) - 1)

        # hlen should be list of integer
        hlens = list(map(int, hlens))
        if maxlenratio == 0:
            maxlen = hs_pad.shape[1]
        else:
            maxlen = max(1, int(maxlenratio * hs_pad.size(1)))
        minlen = int(minlenratio * hs_pad.size(1))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        if self.replace_sos:
            ys_in = [torch.cat([idx, y], dim=0) for idx, y in zip(tgt_lang_ids, ys)]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        
        batch = len(hlens)
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight
        att_weight = 1.0 - ctc_weight
        ctc_margin = 0 #recog_args.ctc_window_margin
        sampling = recog_args.sampling
        # padding for ys with -1
        # pys: utt x olen
        n_bb = batch * beam
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)
        hlens = torch.tensor(hlens).repeat(beam).view(beam, batch).transpose(0, 1).contiguous()
        hlens = hlens.view(-1).tolist()
        hs_pad = hs_pad.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()
        hs_pad = hs_pad.view(n_bb, hs_pad.size(2), -1)
        ys_in_rep =  ys_in_pad.unsqueeze(1).repeat(1, beam, 1, 1)
        ys_in_pad = ys_in_rep.view(n_bb, ys_in_pad.size(1))
        ys_out_rep =  ys_out_pad.unsqueeze(1).repeat(1, beam, 1, 1)
        ys_out_pad = ys_out_rep.view(n_bb, ys_out_pad.size(1))

        indices = np.array(range(self.eos+1), dtype=np.int32)
        # get dim, length info
        batch = ys_in_pad.size(0) # as we repeat here
        if oracle_length:
            olength = ys_out_pad.size(1)
        else:
            olength = maxlen
        logging.info(self.__class__.__name__ + ' input lengths:  ' + str(hlens))
        logging.info(self.__class__.__name__ + ' output lengths: ' + str([y.size(0) for y in ys_out]))

        # initialization
        c_list = [self.zero_state(hs_pad)]
        z_list = [self.zero_state(hs_pad)]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad))
            z_list.append(self.zero_state(hs_pad))
        att_w = None
        z_all = []
        self.att[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att[att_idx](hs_pad, hlens, self.dropout_dec[0](z_list[0]), att_w)
            if i > 0: #self.sampling_probability:
                z_out = self.output(z_all[-1])
                z_out = np.argmax(z_out.detach().cpu(), axis=1)
                z_out = self.dropout_emb(self.embed(to_device(self, z_out)))
                ey = torch.cat((z_out, att_c), dim=1)  # utt x (zdim + hdim)
            else:
                logging.info(' teacher forching ')
                ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            if self.context_residual:
                z_all.append(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))  # utt x (zdim + hdim)
            else:
                z_all.append(self.dropout_dec[-1](z_list[-1]))  # utt x (zdim)

        z_all = torch.stack(z_all, dim=1).view(batch * olength, -1)
        # compute loss
        y_all = self.output(z_all)
        #self.loss = F.cross_entropy(y_all, ys_out_pad.view(-1),
        #                            ignore_index=self.ignore_id,
        #                            reduction='none')
        # -1: eos, which is removed in the loss computation
        #self.loss *= (np.mean([len(x) for x in ys_in]) - 1)
        #acc = th_accuracy(y_all, ys_out_pad, ignore_label=self.ignore_id)
        # logging.info('unsup att loss:' + ''.join(str(self.loss.mean().item()).split('\n')))
        # compute perplexity
        #ppl = np.exp(self.loss.item() * np.mean([len(x) for x in ys_in]) / np.sum([len(x) for x in ys_in]))

        y_list = []
        # show predicted character sequence for debug
        ys_hat = y_all.view(batch, olength, -1)
        ys_true = ys_out_pad
        for (i, y_hat), y_true in zip(enumerate(ys_hat.detach().cpu().numpy()),
                                      ys_true.detach().cpu().numpy()):
            #if i == MAX_DECODER_OUTPUT:
            #    break
            if sampling == 'argmax':
                idx_hat = np.argmax(y_hat[y_true != self.ignore_id], axis=1)
            elif sampling == 'multinomial':
                idx_hat = np.array([torch.multinomial(F.softmax(torch.from_numpy(y_hat[j]), dim=-1), num_samples=1).item() for j in range(olength)])
            y_list.append(torch.from_numpy(idx_hat))
            if self.verbose > 0 and self.char_list is not None:
                idx_true = y_true[y_true != self.ignore_id]
                seq_hat = [self.char_list[int(idx)] for idx in idx_hat]
                seq_true = [self.char_list[int(idx)] for idx in idx_true]
                seq_hat = "".join(seq_hat)
                seq_true = "".join(seq_true)
                logging.info("gen_gtruth[%d]: " % i + seq_true)
                logging.info("generation [%d]: " % i + seq_hat)
        #if self.labeldist is not None:
        #    if self.vlabeldist is None:
        #        self.vlabeldist = to_device(self, torch.from_numpy(self.labeldist))
        #    loss_reg = - torch.sum((F.log_softmax(y_all, dim=1) * self.vlabeldist).view(-1), dim=0) / len(ys_in)
        #    self.loss = (1. - self.lsm_weight) * self.loss + self.lsm_weight * loss_reg
        #y_list = []
        #y_gen = y_gen.transpose(1, 0)
        #y_gen[y_gen == -1] = 0
        #for i in six.moves.range(n_samples):
        #    y_seq = np.array(y_gen[i, :y_lens[i] - 1], dtype=np.int32)
        #    y_list.append(torch.from_numpy(y_seq))
        #    if self.verbose > 0 and self.char_list is not None:
        #        y_str = "".join([self.char_list[int(idx)] for idx in y_seq])
        #        logging.info("generation[%d]: %.4f " % (i, sample_loss.data[i]) + y_str)
        fake_ys_out_pad = to_device(self, pad_list(y_list, self.ignore_id))
        self.loss = F.cross_entropy(y_all, fake_ys_out_pad.view(-1),
                                    ignore_index=self.ignore_id,
                                    reduction='none')
        if rnnlm is not None:
            with torch.no_grad():
                sos = y_list.data.new([self.sos] * len(y_list))
                rnnlm_state, lmz = self.rnnlm.predictor(None, sos)
                lm_loss = F.cross_entropy(lmz, y_list[:, 0], reduction='none')
                for i in six.moves.range(1, ylens[0]):
                    rnnlm_state, lmz = self.rnnlm.predictor(rnnlm_state, ygen[:, i - 1])
                    if self.rnnloss == 'ce':
                        #logging.info("Using CE loss for RNNLoss")
                        loss_i = F.cross_entropy(lmz, ygen[:, i], reduction='none')
                    elif self.rnnloss == 'kl':
                        #logging.info("Using KL divergence loss for RNNLoss")
                        loss_i = F.kl_div(F.softmax(lmz), F.softmax(y_all[:][i]), reduction='none').mean(1)
                    lm_loss += loss_i
                lm_loss = lm_loss / ylens

        self.loss = self.loss.view(batch, olength).mean(1)
        return self.loss, y_list

    def calculate_all_attentions(self, hs_pad, hlen, ys_pad, strm_idx=0, tgt_lang_ids=None):
        """Calculate all of attentions

            :param torch.Tensor hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            :param torch.Tensor hlen: batch of lengths of hidden state sequences (B)
            :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
            :param int strm_idx: stream index for parallel speaker attention in multi-speaker case
            :param torch.Tensor tgt_lang_ids: batch of target language id tensor (B, 1)
            :return: attention weights with the following shape,
                1) multi-head case => attention weights (B, H, Lmax, Tmax),
                2) other case => attention weights (B, Lmax, Tmax).
            :rtype: float ndarray
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in ys_pad]  # parse padded ys
        att_idx = min(strm_idx, len(self.att) - 1)

        # hlen should be list of integer
        hlen = list(map(int, hlen))

        self.loss = None
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos])
        sos = ys[0].new([self.sos])
        if self.replace_sos:
            ys_in = [torch.cat([idx, y], dim=0) for idx, y in zip(tgt_lang_ids, ys)]
        else:
            ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]

        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos)
        ys_out_pad = pad_list(ys_out, self.ignore_id)

        # get length info
        olength = ys_out_pad.size(1)

        # initialization
        c_list = [self.zero_state(hs_pad)]
        z_list = [self.zero_state(hs_pad)]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(hs_pad))
            z_list.append(self.zero_state(hs_pad))
        att_w = None
        att_ws = []
        self.att[att_idx].reset()  # reset pre-computation of h

        # pre-computation of embedding
        eys = self.dropout_emb(self.embed(ys_in_pad))  # utt x olen x zdim

        # loop for an output sequence
        for i in six.moves.range(olength):
            att_c, att_w = self.att[att_idx](hs_pad, hlen, self.dropout_dec[0](z_list[0]), att_w)
            ey = torch.cat((eys[:, i, :], att_c), dim=1)  # utt x (zdim + hdim)
            z_list, c_list = self.rnn_forward(ey, z_list, c_list, z_list, c_list)
            att_ws.append(att_w)

        # convert to numpy array with the shape (B, Lmax, Tmax)
        att_ws = att_to_numpy(att_ws, self.att[att_idx])
        return att_ws

    @staticmethod
    def _get_last_yseq(exp_yseq):
        last = []
        for y_seq in exp_yseq:
            last.append(y_seq[-1])
        return last

    @staticmethod
    def _append_ids(yseq, ids):
        if isinstance(ids, list):
            for i, j in enumerate(ids):
                yseq[i].append(j)
        else:
            for i in range(len(yseq)):
                yseq[i].append(ids)
        return yseq

    @staticmethod
    def _index_select_list(yseq, lst):
        new_yseq = []
        for l in lst:
            new_yseq.append(yseq[l][:])
        return new_yseq

    @staticmethod
    def _index_select_lm_state(rnnlm_state, dim, vidx):
        if isinstance(rnnlm_state, dict):
            new_state = {}
            for k, v in rnnlm_state.items():
                new_state[k] = [torch.index_select(vi, dim, vidx) for vi in v]
        elif isinstance(rnnlm_state, list):
            new_state = []
            for i in vidx:
                new_state.append(rnnlm_state[int(i)][:])
        return new_state

    # scorer interface methods
    def init_state(self, x):
        c_list = [self.zero_state(x.unsqueeze(0))]
        z_list = [self.zero_state(x.unsqueeze(0))]
        for _ in six.moves.range(1, self.dlayers):
            c_list.append(self.zero_state(x.unsqueeze(0)))
            z_list.append(self.zero_state(x.unsqueeze(0)))
        # TODO(karita): support strm_index for `asr_mix`
        strm_index = 0
        att_idx = min(strm_index, len(self.att) - 1)
        self.att[att_idx].reset()  # reset pre-computation of h
        return dict(c_prev=c_list[:], z_prev=z_list[:], a_prev=None, workspace=(att_idx, z_list, c_list))

    def score(self, yseq, state, x):
        att_idx, z_list, c_list = state["workspace"]
        vy = yseq[-1].unsqueeze(0)
        ey = self.dropout_emb(self.embed(vy))  # utt list (1) x zdim
        att_c, att_w = self.att[att_idx](
            x.unsqueeze(0), [x.size(0)],
            self.dropout_dec[0](state['z_prev'][0]), state['a_prev'])
        ey = torch.cat((ey, att_c), dim=1)  # utt(1) x (zdim + hdim)
        z_list, c_list = self.rnn_forward(ey, z_list, c_list, state['z_prev'], state['c_prev'])
        if self.context_residual:
            logits = self.output(torch.cat((self.dropout_dec[-1](z_list[-1]), att_c), dim=-1))
        else:
            logits = self.output(self.dropout_dec[-1](z_list[-1]))
        logp = F.log_softmax(logits, dim=1).squeeze(0)
        return logp, dict(c_prev=c_list[:], z_prev=z_list[:], a_prev=att_w, workspace=(att_idx, z_list, c_list))


def decoder_for(args, odim, sos, eos, att, labeldist):
    return Decoder(args.eprojs, odim, args.dtype, args.dlayers, args.dunits, sos, eos, att, args.verbose,
                   args.char_list, labeldist,
                   args.lsm_weight, args.sampling_probability, args.dropout_rate_decoder,
                   getattr(args, "context_residual", False),  # use getattr to keep compatibility
                   getattr(args, "replace_sos", False))  # use getattr to keep compatibility
