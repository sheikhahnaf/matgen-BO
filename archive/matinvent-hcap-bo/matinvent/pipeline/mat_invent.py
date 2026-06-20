import os
import time
import logging
from typing import Dict
import numpy as np
import torch
from omegaconf import DictConfig

from pipeline.base import ReinL
from pipeline.filters import OptEval, invalid_filter
from pipeline.utils.save import save_structures
from pipeline.utils.logger import Logger
from rewards.reward import Reward
from models.suite.base import ModelSuite


class MatInvent(ReinL):
    def __init__(
        self,
        rl_epoch: int,
        model_suite: ModelSuite,
        reward: Reward,
        sample_cfg: DictConfig,
        finetune_cfg: DictConfig,
        topk_ratio: float,
        save_dir: str,
        save_freq: int = 50,
        device: str = None,
        logger: Logger = None,
        replay: bool = False,
        replay_args: Dict = None,
        div_filter: bool = False,
        df_args: Dict = None,
        gp_config: DictConfig = None,  # NEW: GP routing configuration
        **kwargs,
    ) -> None:
        super().__init__(
            rl_epoch=rl_epoch,
            model_suite=model_suite,
            reward=reward,
            sample_cfg=sample_cfg,
            finetune_cfg=finetune_cfg,
            save_dir=save_dir,
            save_freq=save_freq,
            device=device,
            logger=logger,
            replay=replay,
            replay_args=replay_args,
            **kwargs,
        )
        assert topk_ratio > 0.0 and topk_ratio <= 1.0
        self.topk_ratio = topk_ratio

        # diversity filter
        self.div_filter = div_filter
        self.df_args = df_args

        if 'filter' not in self.sample_cfg:
            self.opt_eval = OptEval()

        # GP routing configuration
        if gp_config is not None:
            self.calibration_steps = gp_config.get('calibration_steps', 10)
            self.gp_trainer = self._init_gp_trainer(gp_config)
            logging.info(f'GP routing enabled: calibration for first {self.calibration_steps} steps')
        else:
            self.calibration_steps = 0
            self.gp_trainer = None

        self.load_model()

    def _init_gp_trainer(self, gp_config: DictConfig):
        """Initialize GP training manager for online learning."""
        from rewards.gp_trainer import GPTrainingManager
        from rewards.gp.noise_estimator import NoiseEstimator

        logging.info('Initializing GP trainer...')

        # CRITICAL FIX: Reuse GP model and featurizer from router
        # Extract from first property that has use_gp_routing enabled
        gp_model = None
        featurizer = None

        for prop_config in self.reward.prop_cfg:
            if hasattr(prop_config, 'use_gp_routing') and prop_config.use_gp_routing:
                if hasattr(prop_config, 'router'):
                    router = prop_config.router
                    gp_model = router.gp_model
                    featurizer = router.featurizer
                    logging.info(f'Extracted GP model and featurizer from router ({prop_config.name})')
                    break

        if gp_model is None or featurizer is None:
            # Fallback: Create new instances (backwards compatible)
            logging.warning('No router found with GP model, creating new instances')
            from rewards.gp.surrogate import GPSurrogate
            from rewards.calculators.orb.featurizer import ORBFeaturizer

            featurizer = ORBFeaturizer(
                n_components=gp_config.get('n_components', 50),
                device=self.device
            )

            gp_model = GPSurrogate(
                input_dim=gp_config.get('n_components', 50),
                task=gp_config.get('task', 'bulk_modulus'),
                device=self.device
            )

        # Initialize noise estimator for heteroscedastic GP
        noise_estimator = None
        if gp_config.get('min_paired_samples'):
            noise_estimator = NoiseEstimator(
                calculator_hierarchy=gp_config.get('calculator_hierarchy', ['vasp', 'orb', 'alignn']),
                min_paired_samples=gp_config.get('min_paired_samples', 20),
                correct_systematic_bias=gp_config.get('correct_systematic_bias', False),
                noise_floor=gp_config.get('noise_floor', 0.1)
            )
            logging.info(f'Noise estimator initialized: hierarchy={noise_estimator.calculator_hierarchy}')

        # Initialize trainer with SHARED model references
        metrics_dir = os.path.join(self.save_dir, 'gp_metrics')
        trainer = GPTrainingManager(
            gp_model=gp_model,  # SHARED reference with router
            featurizer=featurizer,  # SHARED reference with router
            retrain_frequency=gp_config.get('retrain_frequency', 5),
            metrics_dir=metrics_dir,
            min_samples=gp_config.get('min_samples', 10),
            validation_split=gp_config.get('validation_split', 0.2),
            noise_estimator=noise_estimator
        )

        logging.info(f'GP trainer initialized: retrain every {gp_config.get("retrain_frequency", 5)} steps')
        return trainer

    def load_model(self):
        self.agent = self.model_suite.load_model()
        self.prior = self.model_suite.load_model()

        for param in self.agent.parameters():
            param.requires_grad = True
        # Freeze the parameter of prior (pretrained) model
        for param in self.prior.parameters():
            param.requires_grad = False
        self.agent.to(self.device)
        self.prior.to(self.device)

    def sample_step(self):
        sample_data, sample_struc = self.sampler.generate(
            model=self.agent, **self.sample_cfg,
        )
        # Filter invalid samples
        sample_data, sample_struc = invalid_filter(sample_data, sample_struc)

        # save all generated valid structures
        valid_xyz_path = save_structures(
            structures=sample_struc,
            save_dir=self.sample_dir,
            filename=f'step_{self.step:0>4d}_valid.extxyz',
        )

        # MLIP relaxation
        if self.sample_cfg.get('mlip_opt'):
            mlip_opt = self.sample_cfg.mlip_opt
            sample_struc, energies = mlip_opt(sample_struc, valid_xyz_path)
        else:
            energies = None

        # Filter bad samples by selected metrics
        if self.sample_cfg.get('filter'):
            filter = self.sample_cfg.filter
            sample_data, sample_struc, metrics = filter(
                sample_data, sample_struc, energies,
            )
            logging.info(f'Number of filtered samples: {len(sample_struc)}')
        else:
            # metrics, _ = self.opt_eval(sample_struc, energies)
            metrics = {}

        log_str = [f'{k}: {v:.6f}' for k, v in metrics.items()]
        logging.info(', '.join(log_str))

        # max sample size to score/reward
        if self.sample_cfg.get('max_num'):
            max_num = self.sample_cfg.max_num
            if len(sample_struc) > max_num:
                sample_data = sample_data[:max_num]
                sample_struc = sample_struc[:max_num]

        # save structures for evaluation
        eval_xyz_path = save_structures(
            structures=sample_struc,
            save_dir=self.sample_dir,
            filename=f'step_{self.step:0>4d}_eval.extxyz',
        )

        return sample_data, sample_struc, eval_xyz_path, metrics

    def ft_step(self, data_list, rewards, baseline):
        # Tensor Core acceleration for new GPUs (Ampere, Hopper, etc)
        torch.set_float32_matmul_precision("high")
        cfg = self.finetune_cfg
        loader = self.model_suite.get_dataloader(
            samples=data_list,
            rewards=rewards,
            batch_size=len(data_list),
        )

        # model = model.to(args.device)
        optimizer = torch.optim.Adam(self.agent.parameters(), lr=cfg.lr)
        # rewards = torch.tensor(rewards, dtype=torch.float, device=self.device)
        accum_steps = cfg.accum_steps  # accumulation_steps

        for epoch in range(cfg.epochs):
            # logging.info(f"Epoch {epoch} starts:")
            self.agent.train()

            loss_all, loss_diff_all, loss_kl_all = 0., 0., 0.
            for batch in loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                loss, loss_diff, loss_kl = 0., 0., 0.

                for t in range(cfg.timesteps):

                    noised_input = self.agent.add_noise(batch, t)
                    sample_loss, agent_pred = self.agent.calc_sample_loss(noised_input)
                    _, prior_pred = self.prior.calc_sample_loss(noised_input)
                    adv = batch.reward
                    # adv = (batch.score - batch.score.mean()) / batch.score.std()
                    # adv = (batch.reward - baseline) / (batch.reward.max() - baseline)
                    _loss_diff = adv * sample_loss

                    kl_term = self.agent.calc_kl_reg(agent_pred, prior_pred, batch)
                    _loss_kl = kl_term * (1.1 - batch.reward)

                    _loss = (_loss_diff + _loss_kl * cfg.sigma).mean() / accum_steps
                    _loss.backward()
                    if (t + 1) % accum_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    loss += _loss.item() * accum_steps
                    loss_diff += _loss_diff.sum().item()
                    loss_kl += _loss_kl.sum().item()

                loss_diff = loss_diff / cfg.timesteps
                loss_kl = loss_kl / cfg.timesteps
                loss = loss / cfg.timesteps

                if (t + 1) % accum_steps != 0:
                    optimizer.step()

                loss_all += loss * batch.num_graphs
                loss_diff_all += loss_diff
                loss_kl_all += loss_kl

            loss_dict = {
                'loss': loss_all / len(data_list),
                'loss_diff': loss_diff_all / len(data_list),
                'loss_kl': loss_kl_all / len(data_list),
            }
            log_str = [f'{k}: {v:.4f}' for k, v in loss_dict.items()]
            logging.info(f'Epoch {epoch}: ' + ', '.join(log_str))

    def rl_step(self):
        logging.info(f'*****   LOOP {self.step} START   *****')
        start_time = time.time()

        # CALIBRATION PHASE: Set router mode
        if self.gp_trainer is not None:
            if self.step < self.calibration_steps:
                # Enable calibration mode (query all calculators)
                if hasattr(self.reward, 'prop_cfg'):
                    for prop_config in self.reward.prop_cfg:
                        if hasattr(prop_config, 'router'):
                            prop_config.router.calibration_mode = True
                logging.info(f'CALIBRATION MODE: Step {self.step}/{self.calibration_steps}')
            elif self.step == self.calibration_steps:
                # Disable calibration mode (switch to GP routing)
                if hasattr(self.reward, 'prop_cfg'):
                    for prop_config in self.reward.prop_cfg:
                        if hasattr(prop_config, 'router'):
                            prop_config.router.calibration_mode = False
                logging.info(f'CALIBRATION COMPLETE: Switching to GP routing')

        logging.info('SAMPLE:')
        sample_list, sample_struc, xyz_path, sample_metrics = self.sample_step()

        # sample scoring, remove failed samples, ranking and get top k samples
        logging.info('SCORE:')
        sample_list, sample_struc, rewards, prop_dict, failed_mask = self.reward_step(
            sample_list, sample_struc, xyz_path, f'step_{self.step:0>4d}',
        )

        log_dict = {f'{k} mean': v.mean() for k, v in prop_dict.items()}
        log_dict.update({f'{k} std': v.std() for k, v in prop_dict.items()})
        log_dict.update({'reward mean': rewards.mean(), 'reward std': rewards.std()})
        log_dict.update(sample_metrics)

        # Extract routing metadata for GP training (if enabled)
        property_values = None
        features = None
        calculators_used = None
        skip_normal_ltm_extend = False

        if hasattr(self.reward, 'get_routing_metadata'):
            routing_metadata = self.reward.get_routing_metadata()
            if routing_metadata:
                # Extract data for first property (assumes single property for now)
                prop_name = list(routing_metadata.keys())[0]
                metadata = routing_metadata[prop_name]

                # CALIBRATION MODE: Store all calculator results with matching structure_ids
                if metadata.get('mode') == 'calibration':
                    logging.info('Processing calibration data (multi-calculator)...')

                    # Get multi-calculator properties from reward's multi_calc_data attribute
                    if hasattr(self.reward, 'multi_calc_data') and self.reward.multi_calc_data:
                        # In calibration mode, Reward stores multi-calc data separately
                        multi_calc_props = self.reward.multi_calc_data.get(prop_name)
                    else:
                        # Fallback: No multi-calculator data
                        multi_calc_props = {}

                    # Generate globally unique structure IDs (same IDs for all calculators)
                    structure_ids = list(range(
                        self.ltm._structure_id_counter,
                        self.ltm._structure_id_counter + len(sample_struc)
                    ))

                    # Extend LTM for each calculator with SAME structure_ids
                    for calc_name, calc_values in multi_calc_props.items():
                        # Filter calc_values to match successful samples
                        calc_values_filtered = calc_values[~failed_mask] if len(calc_values) == len(failed_mask) else calc_values

                        logging.info(f'  Storing {calc_name} results for {len(calc_values_filtered)} structures')
                        self.ltm.extend(
                            sample_struc,
                            rewards,  # Use same rewards (computed from one calculator)
                            self.step,
                            property_values=calc_values_filtered,  # FIXED: Filtered
                            features=None,  # Features computed once during first store
                            calculators_used=[calc_name] * len(sample_struc),
                            structure_ids=structure_ids  # SAME IDs for all calculators
                        )

                    # Manually increment counter (since we passed structure_ids)
                    self.ltm._structure_id_counter += len(sample_struc)

                    # Log calibration cost
                    if 'cost' in metadata:
                        log_dict['calibration_cost'] = metadata['cost']
                        logging.info(f'Calibration cost: {metadata["cost"]:.4f}')

                    skip_normal_ltm_extend = True  # Don't do normal LTM extend

                else:
                    # NORMAL ROUTING MODE
                    # Log routing statistics
                    if 'routing_counts' in metadata:
                        routing_str = ', '.join([
                            f'{calc}: {count}' for calc, count in metadata['routing_counts'].items()
                        ])
                        logging.info(f'Routing: {routing_str}')
                        log_dict.update({f'routing_{calc}': count for calc, count in metadata['routing_counts'].items()})

                    if 'cost' in metadata:
                        log_dict['routing_cost'] = metadata['cost']
                        logging.info(f'Routing cost: {metadata["cost"]:.4f}')

                    # Extract property values (for GP training) - already filtered by reward_step
                    if prop_name in prop_dict:
                        property_values = prop_dict[prop_name]

                    # Extract features and calculator metadata - NEED FILTERING
                    features_unfiltered = metadata.get('features')
                    routed_to_unfiltered = metadata.get('routed_to')

                    # Filter to match successful samples
                    if features_unfiltered is not None and len(features_unfiltered) == len(failed_mask):
                        features = features_unfiltered[~failed_mask]
                    else:
                        features = features_unfiltered  # Fallback: use as-is if no mismatch

                    if routed_to_unfiltered is not None and len(routed_to_unfiltered) == len(failed_mask):
                        calculators_used = [routed_to_unfiltered[i] for i, failed in enumerate(failed_mask) if not failed]
                    else:
                        calculators_used = routed_to_unfiltered  # Fallback

        # long-term memory (with GP data if available)
        if not skip_normal_ltm_extend:
            self.ltm.extend(
                sample_struc, rewards, self.step,
                property_values=property_values,
                features=features,
                calculators_used=calculators_used
            )
        metrics = self.ltm.calc_metrics(self.reward.threshold)
        self.ltm.save(os.path.join(self.sample_dir, 'long_term_memory.csv'))
        logging.info(
            f'{len(self.ltm)} crystals generated so far, ' +
            f'{len(self.ltm.unique_comps)} unique components.' +
            f'  Burden: {metrics[0]}, Div. Ratio: {metrics[1]}.'
        )
        log_dict.update(
            {
                'crystal_num': len(self.ltm),
                'unique_comps': len(self.ltm.unique_comps),
                'burden': metrics[0],
                'div_ratio': metrics[1],
                'cost': self.cost,
            }
        )

        # GP retraining hook
        if self.gp_trainer is not None and self.gp_trainer.should_retrain(self.step):
            logging.info('GP RETRAIN:')
            gp_metrics = self.gp_trainer.retrain(
                ltm=self.ltm,
                current_step=self.step,
                property_name='property_value'
            )
            if gp_metrics:
                log_dict.update({f'gp_{k}': v for k, v in gp_metrics.items()})
                logging.info(f'GP retraining completed at step {self.step}')

        if self.logger is not None:
            self.logger.log(log_dict, step=self.step)

        # diversity filter
        if self.div_filter:
            rewards, penalty_idx, tol_n, buff_n = self.ltm.div_filter(
                sample_struc, rewards, **self.df_args
            )
            penalty_sample = [sample_list[p] for p in penalty_idx]
            penalty_strucs = [sample_struc[p] for p in penalty_idx]
            logging.info(f'Diversity filter: tol_n={tol_n}, buff_n={buff_n}')

        # topk data points
        sort_idx = np.argsort(rewards)[::-1]
        topk_idx = sort_idx[: int(self.finetune_cfg.batch_size * self.topk_ratio)]
        sample_topk = [sample_list[_i] for _i in topk_idx]
        strucs_topk = [sample_struc[_i] for _i in topk_idx]
        reward_topk = rewards[topk_idx]

        # experience replay
        if self.replay is not None:
            if self.div_filter and len(penalty_strucs) > 0:
                self.replay.memory_purge(penalty_strucs)
            data_replay, reward_replay = self.replay.sample()
            ft_data = sample_topk + data_replay
            ft_reward = np.concatenate((reward_topk, reward_replay))
            self.replay.extend(sample_topk, strucs_topk, reward_topk)
            logging.info(f'replay buffer size={len(self.replay)}')
            # print(f'replay rewards={reward_replay}')
            logging.info(f'buffer reward mean={self.replay.buffer["reward"].values.mean()}')
            # print(f'buffer rewards={replay.buffer["reward"].values}')
        else:
            ft_data = sample_topk
            ft_reward = reward_topk

        # finetuning
        logging.info('FINETUNE:')
        baseline = self.ltm.get_baseline(self.step)
        baseline = min(baseline, ft_reward.min())
        self.ft_step(ft_data, ft_reward, baseline)

        end_time = time.time()
        total_time = (end_time - start_time) / 60
        logging.info(f'*****   LOOP {self.step} FINISH   *****')
        logging.info(f'Total time taken: {total_time:.2f} min.\n\n')

    def run_rl(self):
        logging.info('*****   RL START   *****')
        start_time = time.time()

        for step in range(self.rl_epoch):
            self.step = step
            self.rl_step()
            # Save the agent weights every few iterations
            if (step + 1) % self.save_freq == 0:
                ckpt_dir = os.path.join(self.models_dir, f'loop_{step:0>4d}')
                self.model_suite.save_model(self.agent, ckpt_dir)
        # If the entire training finishes, clean up
        ckpt_dir = os.path.join(self.models_dir, 'final')
        self.model_suite.save_model(self.agent, ckpt_dir)

        logging.info('*****   RL END   *****')
        end_time = time.time()
        logging.info('Total time taken: {} s.'.format(int(end_time - start_time)))
