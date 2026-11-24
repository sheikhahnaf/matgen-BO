"""Diversity-only control for the GP gate.

The gated calculators select their oracle picks with a greedy rule,

    first pick   argmax score
    next picks   argmax (score - lambda * max cosine similarity to the picks so far)

where score is the Expected Improvement in property units and the similarity is
computed on the ORB PCA-50 features. The classes below are the same
calculators with the Expected Improvement replaced by zeros. The first pick is
then the first SUN survivor in generation order and every further pick is the
candidate least similar to those already picked, so the surrogate's ranking
plays no part in the selection. Oracle budget, cold-start cycle, GP refits and
logging are unchanged.

Select with one Hydra override on the gated command line:

    reward.prop_cfg.0.calculator._target_=fme_rev.diversity_gate.DiversityOnlyGateCp
    reward.prop_cfg.0.calculator._target_=fme_rev.diversity_gate.DiversityOnlyGateBM
"""

import numpy as np

import src.calculators.local_esen_bm_gp_routed_v4 as bm_mod
import src.calculators.local_esen_gp_routed_v4 as cp_mod


def _zero_scores(mu, sigma, f_best):
    return np.zeros_like(np.asarray(mu, dtype=float))


class _ZeroEI:
    _module = None

    def calc(self, samples, label="tmp"):
        keep = self._module._expected_improvement
        self._module._expected_improvement = _zero_scores
        try:
            return super().calc(samples, label)
        finally:
            self._module._expected_improvement = keep


class DiversityOnlyGateCp(_ZeroEI, cp_mod.LocalESEN_GPRoutedV4):
    _module = cp_mod


class DiversityOnlyGateBM(_ZeroEI, bm_mod.LocalESEN_BM_GPRoutedV4):
    _module = bm_mod
