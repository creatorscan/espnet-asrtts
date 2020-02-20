import logging

import numpy as np
import torch
import torch.nn.functional as F


def hyp2char(hyps, char_list, recog_args):
    for i, y_hat in enumerate(hyps):
        seq_hat = [char_list[int(idx)] for idx in y_hat if int(idx) != -1]
        seq_hat_text = "".join(seq_hat).replace(recog_args.space, ' ')
        seq_hat_text = seq_hat_text.replace(recog_args.blank, '')
        hyp_chars = seq_hat_text.replace(' ', '')
        logging.info("predicted text[%d]: ", i + hyp_chars)
    return None


class Cycle(torch.nn.Module):
    """Cycle consistency module

    :param asr_model: pytorch model
    :param tts_model: pytorch model
    :param asr_optimizer: optimizer
    :param tts_optimizer: optimizer
    :param int alpha: hyperparameter to scale loss
    """

    def __init__(self, asr_model, tts_model, asr_optimizer, tts_optimizer, alpha):
        super().__init__()
        self.asr_model = asr_model
        self.tts_model = tts_model
        self.asr_optimizer = asr_optimizer
        self.tts_optimizer = tts_optimizer
        self.alpha = alpha

    def forward(self, xs, xlens):
        fake_loss, pred_hyps = self.asr_model(xs, xlens)
        hyp2char(pred_hyps, self.char_list, self.recog_args)



