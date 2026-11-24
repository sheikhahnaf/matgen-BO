"""MatInvent loop with the generator held at its pretrained weights.

Used for the frozen-generator control: the GP acquisition gate still selects
which candidates reach the oracle and still refits every cycle, but the
generative model is never fine-tuned. Everything else in the loop (sampling,
SUN filter, reward scoring, long-term memory, replay buffer bookkeeping) is the
unmodified MatInvent code.

Select it with a single Hydra override on top of the gated (ACC) command line:

    pipeline._target_=fme_rev.frozen_pipeline.MatInventFrozen

A SHA-256 fingerprint of the generator parameters is logged before the first
cycle and after the last one, and the run is marked failed if the two differ.
"""

import hashlib
import logging

import torch

from pipeline.mat_invent import MatInvent


def _fingerprint(named_tensors) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in named_tensors:
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _param_fingerprint(model) -> str:
    return _fingerprint(list(model.named_parameters()))


def _buffer_fingerprint(model) -> str:
    if not hasattr(model, 'named_buffers'):
        return 'n/a'
    return _fingerprint(list(model.named_buffers()))


class MatInventFrozen(MatInvent):

    def load_model(self):
        super().load_model()
        for param in self.agent.parameters():
            param.requires_grad = False
        self.agent.eval()
        self._w0 = _param_fingerprint(self.agent)
        logging.info(
            f'Generator frozen at pretrained weights. params={self._w0} '
            f'buffers={_buffer_fingerprint(self.agent)}'
        )

    def ft_step(self, data_list, rewards, baseline):
        logging.info('Generator frozen: fine-tune step skipped.')

    def run_rl(self):
        super().run_rl()
        w1 = _param_fingerprint(self.agent)
        logging.info(
            f'Generator after the last cycle: params={w1} '
            f'buffers={_buffer_fingerprint(self.agent)}'
        )
        if w1 != self._w0:
            raise RuntimeError('Frozen-generator control violated: agent weights changed.')
        logging.info('Frozen-generator check passed: parameters identical to the start.')
