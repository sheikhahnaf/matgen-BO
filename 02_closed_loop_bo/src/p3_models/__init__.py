"""Phase-3 model adapters: paradigm-specific RL hooks for MatInvent.

Each submodule defines a `<Generator>Module` that subclasses MatInvent's
`BaseModule` (a LightningModule) and exposes the three RL-update primitives
that `pipeline/mat_invent.py:ft_step` calls:

    add_noise(batch, time)    → (noised_input, noises, batch_idx)
    calc_sample_loss(input_all) → (loss, agent_pred)
    calc_kl_reg(agent_pred, prior_pred, batch) → kl_term

and a `<Generator>Suite` (subclass of ModelSuite) that wires it into
MatInvent's pipeline via Hydra config.
"""
