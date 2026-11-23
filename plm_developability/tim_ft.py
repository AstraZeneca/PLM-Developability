"""
Implement the TIM class.

Provide methods for running tasks and performing TIM-GD inference
with support and query data, including adaptation and optimization of weights.
"""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import wandb

from .balanced_dataloader import balanced_epoch_generator
from .utils_basic import calculate_metrics, get_one_hot, mkdir_if_missing, plot_tsne


class FeatureMLP(nn.Module):
    """Simple MLP used as an optional learnable feature transform."""

    def __init__(self, in_feat, hidden_dim=128, out_feat=None, dropout=0.01):
        super().__init__()
        out_feat = out_feat or in_feat
        self.net = nn.Sequential(
            nn.Linear(in_feat, hidden_dim),
            nn.PReLU(),
        )

    def forward(self, x):  # noqa: D102
        return self.net(x)

class TIM(object):
    """
    TIM (Transductive fine-tuning) class for running tasks and performing TIM-GD inference
    with support and query data, including adaptation and optimization of weights.
    """  # noqa: D205

    def __init__(self, device, config):
        self.config = config
        self.temp = config.get('temp', 15.0)
        self.loss_weights = config.get('loss_weights', [1.0, 0.5, 0.1]).copy()
        self.iter = config.get('iter', 20000)
        self.xfrom_iter = config.get('xfrom_iter', 30)
        self.prior_update = config.get('prior_update', [1, 10, 20, 30, 50, 100, 200, 300, 500])
        self.lr_scale = config.get('lr_scale', 0.1)
        self.lr = float(config.get('lr', 1e-4))
        self.alpha_values = config.get('alphas', [1.0, 1.0, 2.0])
        self.print_prec_iter = config.get('print_prec_iter', False)
        self.device = device
        self.trans = config.get('trans', True)
        self.entropies = config.get('entropies', [ 'Shannon', 'Shannon', 'Alpha' ])
        self.prior_type = config.get('prior_type', 'model')
        self.ft = config.get('ft', 'mlp')
        self.update_feat = config.get('update_feat', True)
        # minibatch adaptation settings
        self.batch_threshold = config.get('batch_threshold', 50000)
        self.batch_size = config.get('batch_size', 4096)
        self.warmup_iter = config.get('warmup_iter', 2000)
        self.epoch_model_freq = config.get('epoch_model_freq', 0)
        if self.warmup_iter > 0:
            self.prior_update = [p + self.warmup_iter for p in self.prior_update]

        self.init_info_lists()
        # Initialize checkpoint / convergence bookkeeping
        self.best_loss = float('inf')
        self.best_state = None
        self.model_path = None
        self.loss_history = []

    def init_info_lists(self):  # noqa: D102
        self.timestamps = []
        self.ce_list = []
        self.q_ent_list = []
        self.q_cond_list = []
        self.test_acc = []
        self.losses = []
        # diagnostics: track gradient norms for prototype weights
        self.weight_grad_norms = []

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim].

        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """  # noqa: D205, D401
        n_tasks = samples.size(0)

        logits = self.temp * (samples.matmul(self.weights.transpose(1, 2)) \
                              - 1 / 2 * (self.weights ** 2).sum(2).view(n_tasks, 1, -1) \
                              - 1 / 2 * (samples ** 2).sum(2).view(n_tasks, -1, 1))  #

        return logits

    def get_probas(self, logits: torch.tensor) -> torch.tensor:
        """
        inputs:
            logits : shape [n_tasks, shot].

        returns :
            probas : shape [n_tasks, shot, num_classes]
        """  # noqa: D205, D401
        return logits.softmax(2)

    def init_weights(self, support, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot].

        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
        """  # noqa: D205, D401
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s)
        self.train_dist = one_hot.sum(1).view(support.size(0), -1) / y_s.shape[1]
        self.train_dist/= self.train_dist.sum()
        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.weights = weights / counts

    def compute_lambda(self, support, query, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot].

        updates :
            self.loss_weights[0] : Scalar
        """  # noqa: D205, D401
        self.N_s, self.N_q = support.size(1), query.size(1)
        self.num_classes = torch.unique(y_s).size(0)

        if self.loss_weights[1] == 'auto':
            self.loss_weights[1] = 1 / self.N_s
        if self.loss_weights[2] == 'auto':
            self.loss_weights[2] = 1 / self.N_s
        if self.loss_weights[0] == 'auto':
            self.loss_weights[0] = (1 + self.loss_weights[2]) * self.N_s / self.N_q

    def compute_prior(self, query, y_q, type = 'model') -> torch.tensor:
        """
        inputs:
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_q : torch.Tensor of shape [n_task, q_shot].

        updates :
            self.prior : torch.Tensor of shape [n_task, num_classes]
        """  # noqa: D205, D401
        self.prior_oracle = self.train_dist.clone()
        if type == 'oracle':
            self.prior = self.prior_oracle.clone()
            self.prior[:, 1] = torch.clamp(self.prior[:, 1], min=0.15)
            self.prior[:, 0] = 1.0 - self.prior[:, 1]
        elif type == 'model':
            logits_q = self.get_logits(query)
            probas_q = self.get_probas(logits_q).detach()
            self.prior = probas_q.sum(1).view(query.size(0), -1) / y_q.shape[1]
        elif isinstance(type, list):
            self.prior = torch.tensor(type).unsqueeze(0).to(self.device)

        # Compute the relative error
        deltas = self.prior[:, 1] / self.prior_oracle[:, 1] - 1
        return deltas


    def run_task(self, support, query, y_s, y_q, embed_name, model_path=None):
        """

        """  # noqa: D419

        self.init_weights(support=support, y_s=y_s)
        self.init_feature_transform(support.shape[-1], ft= self.ft)
        if model_path is not None:
            _ = self.load_best_model(model_path)
            self.resume_mode = True
        else:
            self.resume_mode = False
        self.compute_lambda(support=support, query=query, y_s=y_s)
        self.compute_prior(query=query, y_q=y_q, type=self.prior_type)

        support, query, s_probs, q_probs = self.run_adaptation_ft_tim(support=support,
                                                                        query=query,
                                                                        y_s=y_s, y_q=y_q,
                                                                        update_feat=self.update_feat,
                                                                        embed_name = embed_name,
                                                                        ft=self.ft)

        return support, query, s_probs, q_probs, self.model_path

    def run_prediction(self, support, query, y_s, model_path=None):
        """
        Run prediction using the best model weights and optionally updated features.

        Args:
            support (torch.Tensor): Support set features of shape [n_task, s_shot, feature_dim].
            query (torch.Tensor): Query set features of shape [n_task, q_shot, feature_dim].

        Returns
        -------
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Updated support and query features, support probabilities, and query probabilities.
        """
        y_s_one_hot = get_one_hot(y_s)

        check_fc = self.load_best_model(model_path)
        # Only apply fc transform if it was actually optimized during adaptation
        if check_fc and self.update_feat:
            support = self.get_feat(support.squeeze(0), ft=self.ft)
            query = self.get_feat(query.squeeze(0), ft=self.ft)
            support = F.normalize(support, dim=1).unsqueeze(0)
            query = F.normalize(query, dim=1).unsqueeze(0)

        logits_s = self.get_logits(support)
        logits_q = self.get_logits(query)
        s_probs = self.get_probas(logits_s)
        q_probs = self.get_probas(logits_q)
        # Ensure prior exists: prefer computing it from the model's own predicted query labels.
        if not hasattr(self, "prior") or self.prior is None:
            # use model predictions as pseudo-labels for prior estimation
            pred_y_q = q_probs.argmax(2)  # shape [n_tasks, q_shot], torch tensor
            self.compute_prior(query=query, y_q=pred_y_q, type=self.prior_type)
        loss, ce, q_ent, q_cond_ent = self.calculate_losses(s_probs, q_probs, y_s_one_hot)

        return s_probs, q_probs, loss.item(), ce.item(), q_ent.item(), q_cond_ent.item(), support, query



    def init_feature_transform(self, in_feat, ft):
        """Initialize the feature transformation layer."""
        self.fc = FeatureMLP(in_feat, hidden_dim=in_feat, dropout=0.1).to(self.device)

    def get_feat(self, feat, ft):
        """Apply transformation to features."""
        return self.fc(feat)

    def calculate_losses(self, s_probs, q_probs, y_s_one_hot, only_ce=False):
        """
        Calculate the losses based on the probabilities.

        Args:
            s_probs (torch.Tensor): Support probabilities.
            q_probs (torch.Tensor): Query probabilities.

        Returns
        -------
            Tuple[float, float, float]: Cross-entropy loss, marginal entropy loss, conditional entropy loss.
        """
        class_weights = (1 - self.train_dist).unsqueeze(0)
        ce = - (class_weights * y_s_one_hot * torch.log(s_probs + 1e-12)).sum(2).mean(1).sum(0)

        if not only_ce:
        # Marginal entropy type
            if self.entropies[1] == 'Shannon':
                marginal = q_probs.mean(1)
                q_ent = - (marginal * torch.log(marginal / (self.prior + 1e-12))).sum(1).sum(0)
            elif self.entropies[1] == 'Alpha':
                alpha1 = self.alpha_values[1]
                if abs(alpha1 - 1.0) < 1e-6:  # alpha=1 limit → Shannon entropy
                    marginal = q_probs.mean(1)
                    q_ent = -(marginal * torch.log(marginal + 1e-12)).sum(1).sum(0)
                else:
                    q_ent = ((1 - (torch.pow(q_probs.mean(1), alpha1)).sum(1)) / (alpha1 - 1)).sum(0)
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

            # Conditional entropy type
            if self.entropies[2] == 'Shannon':
                q_cond_ent = - (q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
            elif self.entropies[2] == 'Alpha':
                alpha2 = self.alpha_values[2]
                if abs(alpha2 - 1.0) < 1e-6:  # alpha=1 limit → Shannon entropy
                    q_cond_ent = -(q_probs * torch.log(q_probs + 1e-12)).sum(2).mean(1).sum(0)
                else:
                    q_cond_ent = ((1 - (torch.pow(q_probs + 1e-12, alpha2)).sum(2)) / (alpha2 - 1)).mean(1).sum(0)  # noqa: E501
            else:
                raise ValueError("Entropies must be in ['Shannon', 'Alpha']")

        if only_ce:
            return ce, ce, torch.Tensor([0.0]), torch.Tensor([0.0])

        if self.trans:
            loss = self.loss_weights[0] * ce  - (self.loss_weights[1] * q_ent - self.loss_weights[2] * q_cond_ent)
        else:
            loss = self.loss_weights[0] * ce + self.loss_weights[1] * (s_probs.mean(1) * torch.log(s_probs.mean(1))).sum(1).sum(0)  # noqa: E501

        return loss, ce, -q_ent, q_cond_ent

    def run_adaptation_ft_tim(self, support, query, y_s, y_q, update_feat, embed_name = "", ft = 'mlp'):
        """
        Corresponds to the TIM-GD inference
        inputs:
            support : torch.Tensor of shape [n_task, s_shot, feature_dim]
            query : torch.Tensor of shape [n_task, q_shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, s_shot]
            y_q : torch.Tensor of shape [n_task, q_shot]
            update_feat: if to update the transformation weights.


        updates :
            self.weights : torch.Tensor of shape [n_task, num_class, feature_dim]
            self.fc : torch.Tensor of shape [feature_dim, feature_dim]

        """  # noqa: D205
        y_s_one_hot = get_one_hot(y_s)

        self.weights.requires_grad_()
        weight_decay = 1e-4
        optimizer = torch.optim.AdamW([self.weights], lr=self.lr, weight_decay=weight_decay)
        # Decide whether to use minibatch adaptation
        Ns = support.size(1)
        use_minibatch = (Ns > self.batch_threshold)

        if not use_minibatch:
            # Original full-batch behavior
            pbar_fb = tqdm(range(self.iter), desc="tim-iter")
            for i in pbar_fb:

                if i >= self.xfrom_iter and update_feat:
                    if i == self.xfrom_iter:
                        support_init = support
                        query_init = query
                        param_groups = [
                            {'params': self.fc.parameters(), 'lr': self.lr_scale * self.lr},
                            {'params': [self.weights], 'lr': self.lr},
                        ]
                        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

                    support = self.get_feat(support_init.squeeze(0), ft=ft)
                    query = self.get_feat(query_init.squeeze(0), ft=ft)
                    support = F.normalize(support, dim=1).unsqueeze(0)
                    query = F.normalize(query, dim=1).unsqueeze(0)

                logits_s = self.get_logits(support)
                logits_q = self.get_logits(query)

                s_probs = self.get_probas(logits_s)
                q_probs = self.get_probas(logits_q)

                only_ce_now = (i < self.warmup_iter)
                if i == self.warmup_iter:
                    self.best_loss = float('inf')
                    self.best_state = None
                    self.loss_history = []
                loss, ce, q_ent, q_cond_ent = self.calculate_losses(s_probs, q_probs, y_s_one_hot, only_ce=only_ce_now)

                self.losses.append(loss.item())
                self.ce_list.append(ce.item())
                self.q_cond_list.append(q_cond_ent.item())
                self.q_ent_list.append(q_ent.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    title1 = self.check_feat_str()
                    if (i + 1) in self.prior_update:
                        if self.prior_type == 'model':
                            self.compute_prior(query, y_q, type=self.prior_type)
                        P_q_log = self.get_probas(self.get_logits(query)).detach()
                        self._log_prior_step(embed_name, i + 1, P_q_log.squeeze())

                    P_q_logits = self.get_logits(query).detach()
                    P_q = self.get_probas(P_q_logits).detach()
                    correct_mask = (P_q.argmax(2) == y_q)
                    f1 = calculate_metrics(y_q.detach().squeeze(), P_q.squeeze(), train=True)['F1 Score']
                    if self.print_prec_iter and i % 10 == 0:
                        print('\nF1 at iter {}={}'.format(i, f1))
                        print('\nLoss at iter {}={}'.format(i, loss.item()))

                    # Log metrics to wandb if active
                    if wandb.run is not None:
                        wandb.log({
                            "iter": i,
                            f"{embed_name}_loss": loss.item(),
                            f"{embed_name}_cross_entropy": ce.item(),
                            f"{embed_name}_marginal_entropy": q_ent.item(),
                            f"{embed_name}_conditional_entropy": q_cond_ent.item(),
                        })

                    # Plot TSNE
                    if self.config.get("plot", False) and update_feat:
                        if i % self.config.get("plot_freq", 100) == 0:
                            title = title1 + '\n F1: {:.2f}%'.format(f1 * 100)
                            save_name = '{}/{}/{}/iter-{}-ft-{}.png'.format(
                                f"{self.config['log_dir']}/plots", title1, embed_name, i, ft)
                            plot_tsne(torch.cat((support.squeeze(), query.squeeze()), dim=0),
                                      torch.cat((y_s, y_q), dim=1),
                                      support_set=self.N_s,
                                      save_name=save_name,
                                      correct_mask=correct_mask,
                                      title=title,
                                      pred_probs=P_q.squeeze())

                    pbar_fb.set_postfix({
                        'iter': i,
                        'loss': f'{loss.item():.4f}',
                        'CE': f'{ce.item():.4f}',
                        'q_ent': f'{q_ent.item():.4f}',
                        'q_cond': f'{q_cond_ent.item():.4f}',
                        'F1': f'{f1:.3f}',
                    })

                    if i >= self.xfrom_iter and i >= self.warmup_iter:
                        if self.check_convergence(loss.item()):
                            print(f"Loss converged at iter {i}")
                            break
            if self.best_state is not None:
                with torch.no_grad():
                    self.weights = self.best_state['weights'].to(self.device)
                    if 'fc' in self.best_state:
                        self.fc.load_state_dict(self.best_state['fc'])
                    if 'support_init' in locals():
                        support = self.get_feat(support_init.squeeze(0), ft=ft)
                        support = F.normalize(support, dim=1).unsqueeze(0)
                        query = self.get_feat(query_init.squeeze(0), ft=ft)
                        query = F.normalize(query, dim=1).unsqueeze(0)
                    s_probs = self.get_probas(self.get_logits(support))
                    q_probs = self.get_probas(self.get_logits(query))
        else:
            # Minibatch adaptation for very large support sets (Ns > batch_threshold).
            Ns = support.size(1)
            Nq = query.size(1)
            # Remove batch dim for sampling
            support_full = support.squeeze(0)
            query_full = query.squeeze(0)
            y_s_full = y_s.squeeze(0)
            y_q_full = y_q.squeeze(0)

            optimizer_main = optimizer
            ft_optimizer_set = False  # tracks whether FT phase has been activated

            bs_s = min(self.batch_size, Ns)
            num_batches = (Ns + bs_s - 1) // bs_s
            # determine epochs: prefer explicit config, otherwise match iter (1 epoch = 1 full pass = 1 full-batch iter)
            if 'epochs' in self.config and self.config.get('epochs') is not None:
                epochs = int(self.config.get('epochs'))
            else:
                epochs = self.iter

            loader = balanced_epoch_generator(
                support_full, y_s_full, query_full, y_q_full,
                batch_size=bs_s, epochs=epochs,
                seed=int(self.config.get('seed', 2025)), device=self.device,
            )

            total_iters = epochs * num_batches

            def compute_probs_in_chunks(feats_full, chunk_size):
                with torch.no_grad():
                    n = feats_full.size(0)
                    parts = []
                    for start in range(0, n, chunk_size):
                        end = min(start + chunk_size, n)
                        chunk = feats_full[start:end]
                        if ft_optimizer_set and update_feat:
                            chunk = self.get_feat(chunk, ft=ft)
                            chunk = F.normalize(chunk, dim=1)
                        chunk_t = chunk.unsqueeze(0)
                        logits_chunk = self.get_logits(chunk_t)
                        probs_chunk = self.get_probas(logits_chunk).squeeze(0)
                        parts.append(probs_chunk)
                    return torch.cat(parts, dim=0)

            def get_feats_in_chunks(feats_full, chunk_size):
                with torch.no_grad():
                    n = feats_full.size(0)
                    parts = []
                    for start in range(0, n, chunk_size):
                        end = min(start + chunk_size, n)
                        chunk = feats_full[start:end]
                        if ft_optimizer_set and update_feat:
                            chunk = self.get_feat(chunk, ft=ft)
                            chunk = F.normalize(chunk, dim=1)
                        parts.append(chunk)
                    return torch.cat(parts, dim=0)

            # Derive only_ce from loss_weights: CE/FT-CE have weights[1]==0, weights[2]==0;
            # alpha-TIM has nonzero entropy weights and should use the full TIM loss.
            only_ce_mb_config = (float(self.loss_weights[1]) == 0.0 and float(self.loss_weights[2]) == 0.0)

            # use tqdm to display epoch/inner and global iter information
            pbar = tqdm(loader, total=total_iters, desc='tim-adapt', unit='it')
            losses = []
            ce_losses = []
            q_cond_losses = []
            q_ent_losses = []
            last_metrics: dict = {}  # carries epoch-level metrics across batches for live postfix
            for support_mb, y_s_mb, query_mb, y_q_mb, epoch_idx, inner_idx, yielded_global in pbar:
                # Use the global index provided by the generator for consistency
                global_iter = yielded_global
                # Reset accumulators at the start of each epoch
                if inner_idx == 0:
                    losses, ce_losses, q_cond_losses, q_ent_losses = [], [], [], []
                # Activate feature transform at xfrom_iter (mirrors non-minibatch path)
                if epoch_idx >= self.xfrom_iter and update_feat:
                    if not ft_optimizer_set:
                        param_groups = [
                            {'params': self.fc.parameters(), 'lr': self.lr_scale * self.lr},
                            {'params': [self.weights], 'lr': self.lr},
                        ]
                        optimizer_main = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
                        ft_optimizer_set = True
                    s_feat = self.get_feat(support_mb.squeeze(0), ft=ft)
                    s_feat = F.normalize(s_feat, dim=1).unsqueeze(0)
                    q_feat = self.get_feat(query_mb.squeeze(0), ft=ft)
                    q_feat = F.normalize(q_feat, dim=1).unsqueeze(0)
                else:
                    s_feat = support_mb
                    q_feat = query_mb

                logits_s = self.get_logits(s_feat)
                logits_q = self.get_logits(q_feat)

                s_probs = self.get_probas(logits_s)
                q_probs = self.get_probas(logits_q)

                y_s_one_hot_mb = get_one_hot(y_s_mb, num_classes=self.num_classes)
                only_ce_now = only_ce_mb_config or (epoch_idx < self.warmup_iter)
                loss, ce, q_ent, q_cond_ent = self.calculate_losses(s_probs, q_probs, y_s_one_hot_mb, only_ce=only_ce_now)

                # Update tqdm postfix once per epoch (when inner_idx is the last batch of the epoch)
                losses.append(loss.item())
                ce_losses.append(ce.item())
                q_cond_losses.append(q_cond_ent.item())
                q_ent_losses.append(q_ent.item())
                try:
                    end_of_epoch = (inner_idx == (num_batches - 1))
                except Exception:
                    end_of_epoch = False

                if end_of_epoch:
                    if epoch_idx == self.warmup_iter:
                        self.best_loss = float('inf')
                        self.best_state = None
                        self.loss_history = []
                    with torch.no_grad():
                        avg_loss = float(np.mean(losses)) if len(losses) else 0.0
                        avg_ce = float(np.mean(ce_losses)) if len(ce_losses) else 0.0
                        avg_q_ent = float(np.mean(q_ent_losses)) if len(q_ent_losses) else 0.0
                        avg_q_cond = float(np.mean(q_cond_losses)) if len(q_cond_losses) else 0.0

                        self.losses.append(avg_loss)
                        self.ce_list.append(avg_ce)
                        self.q_cond_list.append(avg_q_cond)
                        self.q_ent_list.append(avg_q_ent)

                        last_metrics = {
                            'avg_loss': f'{avg_loss:.4f}',
                            'avg_CE': f'{avg_ce:.4f}',
                            'avg_q_ent': f'{avg_q_ent:.4f}',
                            'avg_q_cond': f'{avg_q_cond:.4f}',
                        }

                        # Compute full-query F1 (mirrors non-minibatch per-iter check)
                        P_q = compute_probs_in_chunks(query_full, self.batch_threshold)
                        if (epoch_idx + 1) in self.prior_update:
                            if self.prior_type == 'model':
                                self.prior = P_q.mean(0, keepdim=True)
                            self._log_prior_step(embed_name, epoch_idx + 1, P_q)
                        f1 = calculate_metrics(y_q_full.detach(), P_q.detach(), train=True)['F1 Score']
                        if self.print_prec_iter and epoch_idx % 10 == 0:
                            print(f'\nF1 at epoch {epoch_idx+1}={f1}')
                            print(f'\nLoss at epoch {epoch_idx+1}={avg_loss}')

                        if wandb.run is not None:
                            wandb.log({
                                "iter": global_iter,
                                f"{embed_name}_loss": avg_loss,
                                f"{embed_name}_cross_entropy": avg_ce,
                                f"{embed_name}_marginal_entropy": avg_q_ent,
                                f"{embed_name}_conditional_entropy": avg_q_cond,
                                f"{embed_name}_f1": f1,
                            })

                        if self.config.get("plot", False):
                            if (epoch_idx + 1) % self.config.get("plot_freq", 100) == 0:
                                correct_mask_plot = (P_q.argmax(1) == y_q_full)
                                title1_plot = self.check_feat_str()
                                s_feats = get_feats_in_chunks(support_full, self.batch_threshold)
                                q_feats = get_feats_in_chunks(query_full, self.batch_threshold)
                                y_s_plot = y_s_full
                                y_q_plot = y_q_full
                                support_set_plot = self.N_s

                                max_s = self.config.get("plot_tsne_max_samples", None)
                                if max_s is not None and s_feats.size(0) > max_s:
                                    s_idx = torch.randperm(s_feats.size(0), device=s_feats.device)[:max_s]
                                    s_feats = s_feats[s_idx]
                                    y_s_plot = y_s_full[s_idx]
                                    support_set_plot = max_s

                                title_plot = title1_plot + '\n F1: {:.2f}%'.format(f1 * 100)
                                save_name = '{}/{}/{}/epoch-{}-ft-{}.png'.format(
                                    f"{self.config['log_dir']}/plots", title1_plot,
                                    embed_name, epoch_idx + 1, ft)
                                plot_tsne(
                                    torch.cat((s_feats, q_feats), dim=0),
                                    torch.cat((y_s_plot.unsqueeze(0), y_q_plot.unsqueeze(0)), dim=1),
                                    support_set=support_set_plot,
                                    save_name=save_name,
                                    correct_mask=correct_mask_plot,
                                    title=title_plot,
                                    pred_probs=P_q)

                # Update postfix every batch so epoch/batch counter stays live
                pbar.set_postfix({
                    'epoch': f'{epoch_idx+1}/{epochs}',
                    'batch': f'{inner_idx+1}/{num_batches}',
                    'giter': global_iter,
                    **last_metrics,
                })
                optimizer_main.zero_grad()
                loss.backward()
                optimizer_main.step()
                with torch.no_grad():
                    if epoch_idx >= self.xfrom_iter and epoch_idx >= self.warmup_iter and end_of_epoch:
                        if self.check_convergence(avg_loss):
                            print(f"Loss converged at epoch {epoch_idx+1}")
                            break

                    # Save intermediate epoch models after every epoch_model_freq:
                    if self.epoch_model_freq > 0 and (epoch_idx + 1) % self.epoch_model_freq == 0 and end_of_epoch:
                        title1 = self.check_feat_str()
                        dataset = self.config['log_dir'].split("/")[-1]
                        _tr = self.config.get("train_stem", "")
                        _te = self.config.get("test_stem", "")
                        _data_tag = f"_{_tr}_{_te}" if (_tr or _te) else ""
                        minibatch_tag = '-SGD'
                        model_name = f"epoch_{epoch_idx+1}_{title1}_{embed_name}_{dataset}{_data_tag}-{minibatch_tag}"  # noqa: E501
                        epoch_model_path = f"{self.config['log_dir']}/models/inference/{model_name}.pt"
                        mkdir_if_missing(epoch_model_path)
                        epoch_state = {
                            'weights': self.weights.detach().cpu().clone(),
                            'fc': self.fc.state_dict(),
                        }
                        torch.save(epoch_state, epoch_model_path)
                        if wandb.run is not None:
                            artifact = wandb.Artifact(model_name, type='model')
                            artifact.add_file(epoch_model_path)
                            wandb.log_artifact(artifact)
                # end inner loop

            if self.best_state is not None:
                with torch.no_grad():
                    self.weights = self.best_state['weights'].to(self.device)
                    if 'fc' in self.best_state:
                        self.fc.load_state_dict(self.best_state['fc'])
            with torch.no_grad():
                s_probs = compute_probs_in_chunks(support_full, self.batch_threshold).unsqueeze(0)
                q_probs = compute_probs_in_chunks(query_full, self.batch_threshold).unsqueeze(0)
            # end loader loop

        dataset = self.config['log_dir'].split("/")[-1]
        _tr = self.config.get("train_stem", "")
        _te = self.config.get("test_stem", "")
        _data_tag = f"_{_tr}_{_te}" if (_tr or _te) else ""
        _mode = self.config.get("mode", "").replace("_", "-")
        _mode_tag = f"_{_mode}" if _mode else ""
        # Ensure we have a feature string for naming
        try:
            title1
        except NameError:
            title1 = self.check_feat_str()

        minibatch_tag = '-SGD' if (
            ('use_minibatch' in locals() and use_minibatch) or
            getattr(self, 'batch_threshold', 0) < support.size(1)
        ) else ''

        if self.resume_mode:
            model_name = f"best_{title1}_{embed_name}_{dataset}{_mode_tag}{_data_tag}-inference{minibatch_tag}"
            self.model_path = f"{self.config['log_dir']}/models/inference/{model_name}.pt"
        else:
            model_name = f"best_{title1}_{embed_name}_{dataset}{_mode_tag}{_data_tag}{minibatch_tag}"
            self.model_path = f"{self.config['log_dir']}/models/{model_name}.pt"

        mkdir_if_missing(self.model_path)
        if self.best_state is not None:
            torch.save(self.best_state, self.model_path)
            # Log the model to wandb if active
            if wandb.run is not None:
                artifact = wandb.Artifact(model_name, type='model')
                artifact.add_file(self.model_path)
                wandb.log_artifact(artifact)
        else:
            best_state = {'weights': self.weights.detach().cpu().clone()}
            if self.update_feat:
                best_state['fc'] = {k: v.clone() for k, v in self.fc.state_dict().items()}
            torch.save(best_state, self.model_path)
            # Log the model to wandb if active
            if wandb.run is not None:
                artifact = wandb.Artifact(model_name, type='model')
                artifact.add_file(self.model_path)
                wandb.log_artifact(artifact)

        return support, query, s_probs, q_probs

    def load_best_model(self, model_path = None):  # noqa: D102

        if model_path is None:
            model_path = self.model_path

        checkpoint = torch.load(model_path, map_location=self.device)
        self.weights = checkpoint['weights'].to(self.device)
        check_fc = 'fc' in checkpoint
        if check_fc:
            self.fc.load_state_dict(checkpoint['fc'])
            self.fc = self.fc.to(self.device)
        return check_fc

    def check_convergence(self, current_loss):  # noqa: D102
        # Running best state — saved model is always the best checkpoint seen,
        # protects against convergence firing at an oscillation peak.
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_state = {'weights': self.weights.detach().cpu().clone()}
            if self.update_feat:
                self.best_state['fc'] = {k: v.clone() for k, v in self.fc.state_dict().items()}

        self.loss_history.append(current_loss)
        window  = self.config.get('convergence_window', 500)
        smooth  = self.config.get('convergence_smooth', 50)
        rel_tol = float(self.config.get('convergence_rel_tol', 1e-4))

        if len(self.loss_history) < window + smooth:
            return False  # not enough history yet

        # Compare smoothed means — oscillations cancel within each average;
        # single-point comparison would be fooled by the sawtooth pattern.
        recent_mean = np.mean(self.loss_history[-smooth:])
        old_mean    = np.mean(self.loss_history[-(window + smooth):-window])
        rel_change  = (old_mean - recent_mean) / (abs(old_mean) + 1e-12)
        if len(self.loss_history) % window == 0:
            print(
                f"[convergence] iter={len(self.loss_history)}  "
                f"rel_change={rel_change:.4f}  tol={rel_tol:.4f}  "
                f"recent_loss={recent_mean:.4f}  old_loss={old_mean:.4f}"
            )
        use_rel_diverge = self.check_feat_str() not in ('Alpha-TIM', 'FT-Alpha-TIM')
        return abs(rel_change) < rel_tol or (use_rel_diverge and rel_change < -0.05)

    def plot_losses(self, embed_name=None):
        """
        Plot each of the loss curves from the run.

        Args:
            embed_name (str, optional): Name of the embedding method. Defaults to None.
        """
        self.plot_loss(embed_name=embed_name, type='CE')
        self.plot_loss(embed_name=embed_name, type='Marginal')
        self.plot_loss(embed_name=embed_name, type='Conditional')
        self.plot_loss(embed_name=embed_name, type='Tim')


    def plot_loss(self, embed_name=None, type='Tim'):
        """
        Plot the specified loss curve from the latest adaptation run.

        Args:
            embed_name (str, optional): Name of the embedding method. Defaults to None.
            type (str, optional): Type of loss to plot.
                      Options are 'Tim', 'CE', 'Marginal', or 'Conditional'. Defaults to 'Tim'.
        """
        if type == 'Tim':
            loss_list = self.losses
        elif type == 'CE':
            loss_list = self.ce_list
        elif type == 'Marginal':
            loss_list = self.q_ent_list
        elif type == 'Conditional':
            loss_list = self.q_cond_list
        else:
            print("Invalid type. Choose from 'Tim', 'CE', 'Marginal', or 'Conditional'.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(loss_list)
        plt.title('{} Loss During Adaptation'.format(type))
        plt.xlabel('Iteration')
        plt.ylabel('{} Loss'.format(type))
        plt.grid(True)


        feat_str = self.check_feat_str()
        save_path = '{}/{}/{}/{}-Loss-ft-{}.png'.format(
            f"{self.config['log_dir']}/plots", feat_str, embed_name, type, self.ft)
        mkdir_if_missing(save_path)
        plt.savefig(save_path)
        plt.close()

    def _log_prior_step(self, embed_name, step, q_probs_full):
        import csv
        import os
        split = f"{self.config.get('train_stem', '')}_{self.config.get('test_stem', '')}"
        seed = self.config.get('seed', 2025)
        model_type = self.check_feat_str()
        pos_prior = self.prior[0, 1].item() if self.prior.dim() == 2 else self.prior[1].item()
        pos_rate = (q_probs_full.argmax(-1) == 1).float().mean().item()
        log_path = os.path.join(self.config['log_dir'], 'prior_trajectory.csv')
        write_header = not os.path.exists(log_path)
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['split', 'backbone', 'model', 'seed', 'step',
                                 'estimated_positive_prior', 'query_positive_rate'])
            writer.writerow([split, embed_name, model_type, seed, step,
                             round(pos_prior, 6), round(pos_rate, 6)])

    def check_feat_str(self):
        """
        Check the feature string based on the update_feat argument.

        Returns
        -------
            str: Feature string indicating whether features are updated or not.
        """
        feat_type = "FT" if self.update_feat else ""
        alpha_marginal = self.entropies[1] == 'Alpha' and abs(self.alpha_values[1] - 1.0) > 1e-6
        if float(self.loss_weights[1]) == 0.0 and float(self.loss_weights[2]) == 0.0:
            model_str = "CE"
        elif alpha_marginal:
            model_str = "Alpha-TIM"
        elif self.entropies[1] == 'Shannon' and self.entropies[2] == 'Alpha' and self.alpha_values[2] == 2:
            model_str = "GINI-TIM"
        elif all(ent == 'Shannon' for ent in self.entropies[1:]):
            model_str = "TIM"
        else:
            model_str = "TIM"  # alpha=1 limit degrades to Shannon
        if feat_type:
            model_str = f"{feat_type}-{model_str}"
        return model_str
