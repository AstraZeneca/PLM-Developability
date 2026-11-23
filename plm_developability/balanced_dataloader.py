"""Support minibatch generator with full epoch coverage.

Provides an epoch-aware minibatch generator for support embeddings.
Each epoch shuffles all support indices and chunks them into batches of
`batch_size` (the last batch may be smaller). This guarantees every support
sample is seen **exactly once per epoch** — standard SGD epoch semantics.
The full query set is returned with every batch.

Usage (in TIM.run_adaptation_ft_tim minibatch branch):

    from plm_developability.balanced_dataloader import balanced_epoch_generator

    loader = balanced_epoch_generator(
        support_full,      # Tensor [Ns, feat]
        y_s_full,          # Tensor [Ns]
        query_full,        # Tensor [Nq, feat]
        y_q_full,          # Tensor [Nq]
        batch_size=4096,
        epochs=500,
        seed=42,
        device=self.device,
    )
    for support_mb, y_s_mb, query_mb, y_q_mb, epoch_idx, inner_idx, global_iter in loader:
        # support_mb: [1, bs, feat], query_mb: [1, Nq, feat]
        # run forward/backward optimizer step as usual

The generator is deterministic when seed is provided.
"""

import math
import random
from typing import Iterator, Tuple

import torch


def balanced_epoch_generator(
    support_full: torch.Tensor,
    y_s_full: torch.Tensor,
    query_full: torch.Tensor,
    y_q_full: torch.Tensor,
    batch_size: int = 64,
    epochs: int = 1,
    seed: int = 42,
    device: torch.device = None,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]]:
    """Yield support minibatches and the full query set for each epoch.

    Each epoch shuffles all support indices and chunks them into batches of
    ``batch_size`` (last batch may be smaller). Every support sample is seen
    exactly once per epoch. The full query set is returned with every batch.

    Yields tuples:
        (support_mb, y_s_mb, query_mb, y_q_mb, epoch_idx, inner_idx, global_iter)

    where ``support_mb`` is ``[1, bs, feat]`` and ``query_mb`` is ``[1, Nq, feat]``.
    """
    device = device or support_full.device
    Ns = support_full.size(0)
    num_batches = math.ceil(Ns / batch_size)

    global_iter = 0
    for epoch in range(epochs):
        rnd = random.Random(seed + epoch * 131)
        shuffled = list(range(Ns))
        rnd.shuffle(shuffled)

        for b in range(num_batches):
            start = b * batch_size
            end = min(start + batch_size, Ns)
            chunk = shuffled[start:end]

            perm_s = torch.tensor(chunk, dtype=torch.long, device=device)
            support_mb = support_full[perm_s].unsqueeze(0).to(device)
            y_s_mb = y_s_full[perm_s].unsqueeze(0).to(device)

            query_mb = query_full.unsqueeze(0).to(device)
            y_q_mb = y_q_full.unsqueeze(0).to(device)

            yield (support_mb, y_s_mb, query_mb, y_q_mb, epoch, b, global_iter)
            global_iter += 1
