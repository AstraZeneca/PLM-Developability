"""
Enhanced TIM for poly-scale datasets: DeepFeatureMLP, TIMPoly, lora_finetune_plm.

Three modes supported from benchmark_poly.py:
  - deep_mlp : frozen PLM embeddings + TIMPoly (DeepFeatureMLP)
  - lora_ce  : LoRA-fine-tuned PLM + CE classifier head
  - lora_tim : LoRA-fine-tuned PLM embeddings + TIMPoly
"""
import gc
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .tim_ft import TIM
from .utils_basic import calculate_metrics, mkdir_if_missing


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced binary/multi-class classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    gamma=2 is the standard choice; higher values focus more on hard examples.
    alpha is a per-class weight tensor (same as nn.CrossEntropyLoss weight=).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


# ---------------------------------------------------------------------------
# DeepFeatureMLP
# ---------------------------------------------------------------------------

class DeepFeatureMLP(nn.Module):
    """Multi-layer MLP feature transform with optional residual connection.

    Architecture: Linear(in) -> PReLU -> [Linear(h) -> PReLU] * ... -> Linear(out)
    With residual=True a skip Linear(in -> out) is added and the output is L2-normalised.
    """

    def __init__(self, in_feat, hidden_dims=None, out_feat=None, dropout=0.1, residual=True):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        out_feat = out_feat or in_feat

        layers = []
        prev = in_feat
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.PReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_feat))
        self.net = nn.Sequential(*layers)

        self.residual = residual
        self.res_proj = (
            nn.Linear(in_feat, out_feat, bias=False) if (residual and in_feat != out_feat) else None
        )

    def forward(self, x):  # noqa: D102
        out = self.net(x)
        if self.residual:
            res = self.res_proj(x) if self.res_proj is not None else x
            out = out + res
        return out


# ---------------------------------------------------------------------------
# TIMPoly
# ---------------------------------------------------------------------------

class TIMPoly(TIM):
    """TIM subclass that swaps FeatureMLP for DeepFeatureMLP.

    Config keys (in addition to all TIM keys):
        mlp_hidden_dims : list[int]  e.g. [512, 256]  (default [512, 256])
        mlp_residual    : bool       whether to add skip connection (default True)
        mlp_dropout     : float      dropout rate in DeepFeatureMLP (default 0.1)
    """

    def init_feature_transform(self, in_feat, ft):
        """Override: use DeepFeatureMLP instead of FeatureMLP."""
        
        hidden_dims = self.config.get("mlp_hidden_dims", [512, 256])
        residual = self.config.get("mlp_residual", True)
        dropout = self.config.get("mlp_dropout", 0.1)
        self.fc = DeepFeatureMLP(
            in_feat=in_feat,
            hidden_dims=hidden_dims,
            out_feat=in_feat,
            dropout=dropout,
            residual=residual,
        ).to(self.device)


# ---------------------------------------------------------------------------
# LoRA fine-tuning helpers
# ---------------------------------------------------------------------------

class _SaprotTokenizerWrapper:
    """Wraps SaProt EsmTokenizer to auto-convert plain AA sequences to SaProt format.

    SaProt expects 2-char tokens like "A#M#K#..." (amino acid + '#' dummy structure).
    Uses encoding.seq_to_struc_sec — same logic as SaprotEncoding.encode_sequences.
    """

    def __init__(self, tokenizer, encoding):
        self.tokenizer = tokenizer
        self.encoding = encoding
        self.pad_token_id = tokenizer.pad_token_id

    def _to_saprot(self, seq):
        heavy_list, light_list = self.encoding.seq_to_struc_sec([seq])
        return heavy_list[0] + light_list[0]

    def __call__(self, sequences, return_tensors="pt", max_length=512, truncation=False, padding=False, **kwargs):
        if isinstance(sequences, str):
            sequences = [sequences]
        converted = [self._to_saprot(s) for s in sequences]
        return self.tokenizer(
            converted,
            return_tensors=return_tensors,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            **kwargs,
        )


class _FairEsmTokenizerWrapper:
    """Wraps fair-esm alphabet to mimic the HuggingFace tokenizer call API."""

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.batch_converter = alphabet.get_batch_converter()
        self.pad_token_id = alphabet.padding_idx

    def __call__(self, sequences, return_tensors="pt", max_length=512, truncation=False, padding=False, **kwargs):
        if isinstance(sequences, str):
            sequences = [sequences]
        if truncation and max_length:
            sequences = [s[:max_length - 2] for s in sequences]  # reserve 2 for BOS/EOS
        _, _, tokens = self.batch_converter([("", s) for s in sequences])
        attention_mask = (tokens != self.alphabet.padding_idx).long()
        return {"input_ids": tokens, "attention_mask": attention_mask}


class _FairEsmBackbone(nn.Module):
    """Wraps a fair-esm model with a HuggingFace-compatible forward API."""

    def __init__(self, esm_model, num_layers):
        super().__init__()
        self.esm_model = esm_model
        self.num_layers = num_layers

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, **kwargs):
        from types import SimpleNamespace
        results = self.esm_model(input_ids, repr_layers=[self.num_layers], return_contacts=False)
        hs = results["representations"][self.num_layers]
        return SimpleNamespace(hidden_states=(hs,), last_hidden_state=hs)


class _PLMDataset(Dataset):
    """On-the-fly tokenising dataset for HuggingFace tokenizers."""

    def __init__(self, tokenizer, sequences, labels, max_length=512):
        self.tokenizer = tokenizer
        self.sequences = list(sequences)
        self.labels = [int(x) for x in labels]
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        toks = self.tokenizer(
            self.sequences[idx],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        # squeeze the batch dim added by return_tensors="pt"
        toks = {k: v.squeeze(0) for k, v in toks.items()}
        toks["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return toks


def _collate_pad(batch, pad_token_id=1):
    """Collate variable-length sequences with padding."""
    keys = [k for k in batch[0] if k != "labels"]
    out = {}
    id_lengths = [b["input_ids"].size(0) for b in batch if "input_ids" in b]
    max_len = max(id_lengths) if id_lengths else max(b[k].size(0) for b in batch for k in keys)
    for k in keys:
        tensors = [b[k] for b in batch]
        if k == "input_ids":
            padded = torch.full((len(tensors), max_len), pad_token_id, dtype=torch.long)
            for i, t in enumerate(tensors):
                padded[i, : t.size(0)] = t
        elif tensors[0].dtype == torch.long:
            padded = torch.zeros(len(tensors), max_len, dtype=torch.long)
            for i, t in enumerate(tensors):
                padded[i, : t.size(0)] = t
        else:
            padded = torch.zeros(len(tensors), max_len, dtype=tensors[0].dtype)
            for i, t in enumerate(tensors):
                padded[i, : t.size(0)] = t
        out[k] = padded
    out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


def _get_backbone_and_tokenizer(encoding):
    """Extract a HuggingFace-compatible backbone and tokenizer from an encoding object.

    Supports:
      - SaprotEncoding  : encoding.model.model (EsmForMaskedLM) + encoding.tokenizer
      - ESMEncoding     : loads facebook/esm2_* from HuggingFace (requires internet or cached)
    """
    from plm_developability.encodings import SaprotEncoding, ESMEncoding

    if isinstance(encoding, SaprotEncoding):
        # SaprotBaseModel sets lm_head=None on EsmForMaskedLM; use the inner EsmModel instead.
        # Wrap tokenizer to convert plain AA strings to SaProt format ("A#M#K#...") before tokenizing.
        backbone = encoding.model.model.esm
        tokenizer = _SaprotTokenizerWrapper(encoding.tokenizer, encoding)
        return backbone, tokenizer

    if isinstance(encoding, ESMEncoding):
        # Reuse the already-loaded fair-esm model via HuggingFace-compatible wrappers
        import re
        m = re.search(r't(\d+)', encoding.model_version)
        num_layers = int(m.group(1)) if m else 33
        logger.info("Wrapping fair-esm %s backbone for LoRA (num_layers=%d)", encoding.model_version, num_layers)
        return _FairEsmBackbone(encoding.model, num_layers), _FairEsmTokenizerWrapper(encoding.alphabet)

    raise RuntimeError(
        f"lora_finetune_plm does not support encoding type {type(encoding).__name__}. "
        f"Supported: SaprotEncoding, ESMEncoding."
    )


@torch.no_grad()
def _embed_with_wrapper(wrapper, tokenizer, sequences, device, batch_size=32):
    """Re-compute mean-pooled L2-normalised embeddings using a fine-tuned wrapper model."""
    wrapper.eval()
    all_feats = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        enc = tokenizer(
            chunk,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = wrapper.backbone(output_hidden_states=True, **enc)
        if hasattr(out, "hidden_states"):
            hs = out.hidden_states[-1]
        elif isinstance(out, dict) and "hidden_states" in out:
            hs = out["hidden_states"][-1]
        else:
            hs = out.last_hidden_state
        feats = hs.mean(dim=1)
        feats = F.normalize(feats, dim=-1)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def lora_finetune_plm(encoding, df_train, df_test, df_val, config, device):
    """Fine-tune a PLM with LoRA + CE on poly training data and return LoRA embeddings.

    Stage 1: Wrap PLM backbone with LoRA, train with CrossEntropyLoss.
    Stage 2: Re-compute embeddings with the fine-tuned backbone (no_grad).

    Returns
    -------
    dict with keys:
        embeddings_train : float32 Tensor [N_train, D]
        embeddings_test  : float32 Tensor [N_test,  D]
        embeddings_val   : float32 Tensor [N_val,   D]  (empty if val is empty)
        ce_probs_train   : float32 Tensor [N_train, 2]
        ce_probs_test    : float32 Tensor [N_test,  2]
        model_path       : str — path to saved LoRA weights
    """
    import wandb

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    target_col = config["target_column"]
    data_dir = config["data_dir"]
    log_dir = config["log_dir"]
    enc_name = str(encoding)

    train_stem = Path(config["train_file"]).stem
    test_stem = Path(config["test_file"]).stem
    val_stem = Path(config.get("val_file", "val_poly.csv")).stem
    dataset_tag = Path(data_dir).name.lower()

    lora_emb_dir = f"{data_dir}/encodings"
    os.makedirs(lora_emb_dir, exist_ok=True)

    train_cache = f"{lora_emb_dir}/{enc_name}_lora_{train_stem}_{train_stem}_{dataset_tag}.pt"
    test_cache = f"{lora_emb_dir}/{enc_name}_lora_{train_stem}_{test_stem}_{dataset_tag}.pt"
    val_cache = f"{lora_emb_dir}/{enc_name}_lora_{train_stem}_{val_stem}_{dataset_tag}.pt"
    weights_dir = f"{log_dir}/lora_weights/{enc_name}"
    os.makedirs(weights_dir, exist_ok=True)

    seq_col = config.get("seq_col", "VH-VL")
    seqs_train = list(df_train[seq_col].values)
    y_train = list(df_train[target_col].values)
    seqs_test = list(df_test[seq_col].values)
    seqs_val = list(df_val[seq_col].values) if len(df_val) > 0 else []

    # --- Load or run LoRA training ---
    best_path_check = os.path.join(weights_dir, f"best_model_{train_stem}.pt")
    embeddings_cached = os.path.exists(train_cache) and os.path.exists(test_cache)
    model_cached = os.path.exists(best_path_check)
    lora_tim_test_only = (
        config.get("mode") == "lora_tim"
        and os.path.exists(train_cache)
        and not os.path.exists(test_cache)
        and model_cached
    )

    if embeddings_cached and model_cached and config.get("mode") != "lora_tim":
        # lora_ce: only ce_probs are needed downstream — skip loading large embedding tensors
        logger.info("Loading cached LoRA embeddings + model for %s", enc_name)
        emb_train = torch.empty(0)
        emb_test = torch.empty(0)
        emb_val = torch.empty(0)
        # re-build wrapper with LoRA applied (state dict has peft keys) then load weights
        backbone, tokenizer = _get_backbone_and_tokenizer(encoding)
        backbone = backbone.to(device)
        saved_sd_cached = torch.load(best_path_check, map_location=device)
        in_dim = saved_sd_cached["head.weight"].shape[1]
        wrapper = _LoRAWrapper(backbone=backbone, in_dim=in_dim, temp=config.get("temp", 15.0)).to(device)
        has_lora_keys_cached = any("lora_" in k or "base_model" in k for k in saved_sd_cached.keys())
        if has_lora_keys_cached:
            try:
                from peft import LoraConfig, get_peft_model
                if isinstance(backbone, _FairEsmBackbone):
                    default_target_cached = ["q_proj", "k_proj", "v_proj", "out_proj"]
                else:
                    default_target_cached = ["query", "key", "value", "intermediate.dense", "output.dense"]
                lora_cfg_cached = LoraConfig(
                    r=config.get("lora_r", 16),
                    lora_alpha=config.get("lora_alpha", 32),
                    lora_dropout=config.get("lora_dropout", 0.1),
                    bias="none",
                    target_modules=config.get("lora_target_modules", default_target_cached),
                )
                wrapper.backbone = get_peft_model(wrapper.backbone, lora_cfg_cached)
            except Exception as exc:
                logger.warning("PEFT unavailable (%s); loading head weights only", exc)
        wrapper.load_state_dict(saved_sd_cached)
        infer_bs_cached = config.get("lora_infer_batch_size", 64)
        ce_probs_train, ce_probs_test = _get_ce_probs(
            wrapper, tokenizer, seqs_train, seqs_test, device, batch_size=infer_bs_cached
        )
    elif embeddings_cached:
        # lora_tim: only embeddings needed — no CE head, no model weights
        logger.info("Loading cached LoRA embeddings for %s (no model weights needed)", enc_name)
        emb_train = torch.load(train_cache, map_location="cpu")
        emb_test = torch.load(test_cache, map_location="cpu")
        emb_val = torch.load(val_cache, map_location="cpu") if os.path.exists(val_cache) else torch.empty(0)
        ce_probs_train = torch.zeros(len(emb_train), 2)
        ce_probs_test = torch.zeros(len(emb_test), 2)
    elif lora_tim_test_only:
        # lora_tim: train embeddings cached, new test file — re-embed test only using saved model
        logger.info("Loading cached train embeddings; re-embedding new test split for %s", enc_name)
        emb_train = torch.load(train_cache, map_location="cpu")
        backbone, tokenizer = _get_backbone_and_tokenizer(encoding)
        backbone = backbone.to(device)
        saved_sd_tim = torch.load(best_path_check, map_location=device)
        in_dim = saved_sd_tim["head.weight"].shape[1]
        wrapper = _LoRAWrapper(backbone=backbone, in_dim=in_dim, temp=config.get("temp", 15.0)).to(device)
        has_lora_keys_tim = any("lora_" in k or "base_model" in k for k in saved_sd_tim.keys())
        if not config.get("full_finetune", False) and has_lora_keys_tim:
            try:
                from peft import LoraConfig, get_peft_model
                if isinstance(backbone, _FairEsmBackbone):
                    default_target_tim = ["q_proj", "k_proj", "v_proj", "out_proj"]
                else:
                    default_target_tim = ["query", "key", "value", "intermediate.dense", "output.dense"]
                lora_cfg_tim = LoraConfig(
                    r=next((v.shape[0] for k, v in saved_sd_tim.items() if "lora_A" in k), config.get("lora_r", 16)),
                    lora_alpha=config.get("lora_alpha", 32),
                    lora_dropout=config.get("lora_dropout", 0.1),
                    bias="none",
                    target_modules=config.get("lora_target_modules", default_target_tim),
                )
                wrapper.backbone = get_peft_model(wrapper.backbone, lora_cfg_tim)
            except Exception as exc:
                logger.warning("PEFT unavailable (%s); loading head weights only", exc)
        wrapper.load_state_dict(saved_sd_tim)
        infer_bs_tim = config.get("lora_infer_batch_size", 64)
        emb_test = _embed_with_wrapper(wrapper, tokenizer, seqs_test, device, batch_size=infer_bs_tim)
        torch.save(emb_test, test_cache)
        emb_val = torch.empty(0)
        if seqs_val:
            emb_val = _embed_with_wrapper(wrapper, tokenizer, seqs_val, device, batch_size=infer_bs_tim)
            torch.save(emb_val, val_cache)
        ce_probs_train = torch.zeros(len(emb_train), 2)
        ce_probs_test = torch.zeros(len(emb_test), 2)
    elif model_cached:
        logger.info("Model weights found but no embeddings — re-computing embeddings for %s", enc_name)
        backbone, tokenizer = _get_backbone_and_tokenizer(encoding)
        backbone = backbone.to(device)
        sample_enc = tokenizer(seqs_train[0], return_tensors="pt", max_length=64, truncation=True)
        sample_enc = {k: v.to(device) for k, v in sample_enc.items()}
        with torch.no_grad():
            out = backbone(output_hidden_states=True, **sample_enc)
        if hasattr(out, "hidden_states"):
            in_dim = out.hidden_states[-1].shape[-1]
        elif isinstance(out, dict) and "hidden_states" in out:
            in_dim = out["hidden_states"][-1].shape[-1]
        else:
            in_dim = out.last_hidden_state.shape[-1]
        wrapper = _LoRAWrapper(backbone=backbone, in_dim=in_dim, temp=config.get("temp", 15.0)).to(device)
        saved_sd = torch.load(best_path_check, map_location=device)
        has_lora_keys = any("lora_" in k or "base_model" in k for k in saved_sd.keys())
        if not config.get("full_finetune", False) and has_lora_keys:
            try:
                from peft import LoraConfig, get_peft_model
                if isinstance(backbone, _FairEsmBackbone):
                    default_target = ["q_proj", "k_proj", "v_proj", "out_proj"]
                else:
                    default_target = ["query", "key", "value", "intermediate.dense", "output.dense"]
                lora_cfg = LoraConfig(
                    r=next((v.shape[0] for k, v in saved_sd.items() if "lora_A" in k), config.get("lora_r", 16)),
                    lora_alpha=config.get("lora_alpha", 32),
                    lora_dropout=config.get("lora_dropout", 0.1),
                    bias="none",
                    target_modules=config.get("lora_target_modules", default_target),
                )
                wrapper.backbone = get_peft_model(wrapper.backbone, lora_cfg)
            except Exception as exc:
                logger.warning("PEFT unavailable (%s); loading head weights only", exc)
        wrapper.load_state_dict(saved_sd)
        infer_bs = config.get("lora_infer_batch_size", 64)
        emb_train = _embed_with_wrapper(wrapper, tokenizer, seqs_train, device, batch_size=infer_bs)
        emb_test  = _embed_with_wrapper(wrapper, tokenizer, seqs_test,  device, batch_size=infer_bs)
        torch.save(emb_train, train_cache)
        torch.save(emb_test,  test_cache)
        emb_val = torch.empty(0)
        if seqs_val:
            emb_val = _embed_with_wrapper(wrapper, tokenizer, seqs_val, device, batch_size=infer_bs)
            torch.save(emb_val, val_cache)
        ce_probs_train, ce_probs_test = _get_ce_probs(
            wrapper, tokenizer, seqs_train, seqs_test, device, batch_size=infer_bs
        )
    else:
        backbone, tokenizer = _get_backbone_and_tokenizer(encoding)
        backbone = backbone.to(device)

        # infer embedding dim
        sample_enc = tokenizer(seqs_train[0], return_tensors="pt", max_length=64, truncation=True)
        sample_enc = {k: v.to(device) for k, v in sample_enc.items()}
        with torch.no_grad():
            out = backbone(output_hidden_states=True, **sample_enc)
        if hasattr(out, "hidden_states"):
            in_dim = out.hidden_states[-1].shape[-1]
        elif isinstance(out, dict) and "hidden_states" in out:
            in_dim = out["hidden_states"][-1].shape[-1]
        else:
            in_dim = out.last_hidden_state.shape[-1]

        wrapper = _LoRAWrapper(backbone=backbone, in_dim=in_dim, temp=config.get("temp", 15.0)).to(device)

        # apply LoRA (skip if full_finetune: true in config)
        if config.get("full_finetune", False):
            logger.info("Full fine-tuning %s backbone (no LoRA)", enc_name)
            if hasattr(wrapper.backbone, "gradient_checkpointing_enable"):
                wrapper.backbone.gradient_checkpointing_enable()
        else:
            try:
                from peft import LoraConfig, get_peft_model
                # fair-esm uses q_proj/k_proj/v_proj/out_proj; SaProt (HF EsmModel) uses query/key/value/dense
                if isinstance(backbone, _FairEsmBackbone):
                    default_target = ["q_proj", "k_proj", "v_proj", "out_proj"]
                else:
                    default_target = ["query", "key", "value", "intermediate.dense", "output.dense"]
                lora_cfg = LoraConfig(
                    r=config.get("lora_r", 16),
                    lora_alpha=config.get("lora_alpha", 32),
                    lora_dropout=config.get("lora_dropout", 0.1),
                    bias="none",
                    target_modules=config.get("lora_target_modules", default_target),
                )
                wrapper.backbone = get_peft_model(wrapper.backbone, lora_cfg)
                if hasattr(wrapper.backbone, "gradient_checkpointing_enable"):
                    wrapper.backbone.gradient_checkpointing_enable()
                logger.info("Applied LoRA to %s backbone", enc_name)
            except Exception as exc:
                logger.warning("PEFT unavailable (%s); continuing with full fine-tuning", exc)

        # build dataset and loader
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1
        train_ds = _PLMDataset(tokenizer, seqs_train, y_train)
        train_loader = DataLoader(
            train_ds,
            batch_size=config.get("lora_batch_size", 8),
            shuffle=True,
            collate_fn=lambda b: _collate_pad(b, pad_token_id=pad_id),
        )

        opt = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, wrapper.parameters()),
            lr=float(config.get("lora_lr", 5e-5)),
        )
        loss_type = config.get("loss_type", "ce")
        if loss_type == "focal" or config.get("use_class_weights", False):
            y_arr = np.array(y_train, dtype=int)
            n_cls = int(y_arr.max()) + 1
            counts = np.bincount(y_arr, minlength=n_cls).astype(float)
            raw = len(y_arr) / (n_cls * counts)
            alpha = torch.tensor(np.sqrt(raw), dtype=torch.float, device=device)
        else:
            alpha = None
        if loss_type == "focal":
            crit = FocalLoss(gamma=config.get("focal_gamma", 2.0), alpha=alpha)
        else:
            crit = nn.CrossEntropyLoss(weight=alpha)
        epochs = config.get("lora_epochs", 5)
        best_loss = float("inf")
        best_path = os.path.join(weights_dir, f"best_model_{train_stem}.pt")

        for ep in range(epochs):
            wrapper.train()
            total, n = 0.0, 0
            pbar = tqdm(train_loader, desc=f"LoRA [{enc_name}] epoch {ep+1}/{epochs}")
            for batch in pbar:
                labels = batch.pop("labels").to(device)
                batch = {k: v.to(device) for k, v in batch.items()}
                opt.zero_grad()
                logits = wrapper(**batch)
                loss = crit(logits, labels)
                loss.backward()
                opt.step()
                total += loss.item() * labels.size(0)
                n += labels.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            ep_loss = total / max(n, 1)
            logger.info("LoRA %s epoch %d loss %.4f", enc_name, ep + 1, ep_loss)
            if ep_loss < best_loss:
                best_loss = ep_loss
                torch.save(wrapper.state_dict(), best_path)
            if wandb.run is not None:
                wandb.log({f"lora/{enc_name}/train_loss": ep_loss, "lora/epoch": ep + 1})

        wrapper.load_state_dict(torch.load(best_path, map_location=device))

        # re-embed with fine-tuned backbone
        logger.info("Re-computing LoRA embeddings for %s", enc_name)
        infer_bs = config.get("lora_infer_batch_size", 64)
        emb_train = _embed_with_wrapper(wrapper, tokenizer, seqs_train, device, batch_size=infer_bs)
        emb_test = _embed_with_wrapper(wrapper, tokenizer, seqs_test, device, batch_size=infer_bs)
        torch.save(emb_train, train_cache)
        torch.save(emb_test, test_cache)

        emb_val = torch.empty(0)
        if seqs_val:
            emb_val = _embed_with_wrapper(wrapper, tokenizer, seqs_val, device, batch_size=infer_bs)
            torch.save(emb_val, val_cache)

        ce_probs_train, ce_probs_test = _get_ce_probs(wrapper, tokenizer, seqs_train, seqs_test, device,
                                                       batch_size=infer_bs)

    if 'wrapper' in locals():
        del wrapper
    gc.collect()

    return {
        "embeddings_train": emb_train,
        "embeddings_test": emb_test,
        "embeddings_val": emb_val,
        "ce_probs_train": ce_probs_train,
        "ce_probs_test": ce_probs_test,
        "model_path": weights_dir,
        "enc_name": enc_name,
    }


class _LoRAWrapper(nn.Module):
    """Thin wrapper: backbone -> mean-pool -> L2-norm -> TIM-style prototype logits."""

    def __init__(self, backbone, in_dim, num_classes=2, temp=15.0):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(in_dim, num_classes)
        self.temp = float(temp)

    def _encode(self, **batch):
        out = self.backbone(output_hidden_states=True, **batch)
        if hasattr(out, "hidden_states"):
            hs = out.hidden_states[-1]
        elif isinstance(out, dict) and "hidden_states" in out:
            hs = out["hidden_states"][-1]
        else:
            hs = out.last_hidden_state
        return hs.mean(dim=1)

    def forward(self, **batch):  # noqa: D102
        return self.head(self._encode(**batch))


@torch.no_grad()
def _get_ce_probs(wrapper, tokenizer, seqs_train, seqs_test, device, batch_size=64):
    """Return softmax probabilities from the CE wrapper for train and test sets."""
    wrapper.eval()

    def _probs(seqs, desc):
        all_p = []
        n_batches = (len(seqs) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(seqs), batch_size), total=n_batches, desc=desc)
        for start in pbar:
            chunk = seqs[start : start + batch_size]
            enc = tokenizer(chunk, max_length=512, truncation=True, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = wrapper(**enc)
            all_p.append(logits.softmax(-1).cpu())
        return torch.cat(all_p, dim=0)

    return _probs(seqs_train, "CE-probs train"), _probs(seqs_test, "CE-probs test")
