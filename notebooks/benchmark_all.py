"""benchmark_all.py — unified TIM / LoRA-CE / LoRA-TIM benchmark.

Usage:
    poetry run python notebooks/benchmark_all.py --config notebooks/configs/minibatch/config-ce.yaml
    poetry run python notebooks/benchmark_all.py --config notebooks/configs/minibatch/config-poly-lora-ce.yaml
    poetry run python notebooks/benchmark_all.py --config notebooks/configs/minibatch/config-poly-lora-alpha-tim.yaml

Mode is selected by the 'mode' key in the config (default: tim):
    tim      — frozen PLM embeddings + TIM; multi-seed aggregation if 'seeds' list given
    lora_ce  — LoRA-fine-tuned PLM + CE classifier head; single seed
    lora_tim — LoRA-fine-tuned PLM re-embeddings + TIM; multi-seed on TIM stage if 'seeds' given
"""
import gc
import logging
import os
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import wandb

warnings.filterwarnings("ignore")

from plm_developability.encodings import (
    ESMEncoding,
    OneHotEncoding,
    SaprotEncoding,
)
from plm_developability.tim_ft import TIM
from plm_developability.tim_ft_poly import lora_finetune_plm
from plm_developability.utils_basic import (
    calculate_metrics,
    load_yaml_config,
    mkdir_if_missing,
    plot_pr_curves,
    plot_tsne,
    set_basic_logging_config,
)

RUN_STARTED_TIMESTAMP = "RUN_STARTED_TIMESTAMP"


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _clear_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def setup_working_directory(target_folder="PLM-Developability"):
    current_dir = os.getcwd()
    while target_folder not in os.listdir(current_dir):
        current_dir = os.path.dirname(current_dir)
    os.chdir(os.path.join(current_dir, target_folder))
    logging.info(f"Working directory set to: {os.getcwd()}")


def load_data(file_path):
    return pd.read_csv(file_path)


def initialize_encodings(config):
    device = _get_device()
    dd_dict = {"device": device, "dtype": torch.float32}
    encodings, encoding_names = [], []
    esm_model_map = {
        "t6-8m":    "esm2_t6_8M_UR50D",
        "t12-35m":  "esm2_t12_35M_UR50D",
        "t30-150m": "esm2_t30_150M_UR50D",
        "t33-650m": "esm2_t33_650M_UR50D",
        "t36-3b":   "esm2_t36_3B_UR50D",
        "t48-15b":  "esm2_t48_15B_UR50D",
    }
    tim_model_paths_raw = config.get("tim_model_paths", [])
    for idx, plm in enumerate(config.get("plms", [])):
        if plm.lower().startswith("esm"):
            model_version = config.get("model_version", "esm2_t30_150M_UR50D")
            if "." in plm:
                tag = plm.split(".", 1)[1].replace("_", "-").lower()
                model_version = esm_model_map.get(tag, model_version)
            enc = ESMEncoding(dd_dict=dd_dict, model_version=model_version,
                              chunk_size=config.get("chunk_size", 10))
        elif plm.lower() == "onehot":
            enc = OneHotEncoding()
        elif plm.lower() == "saprot-seq-pdb":
            enc = SaprotEncoding(use_structure="seq", model_dir=config["local_saprot_dir_pdb"],
                                 dd_dict=dd_dict, chain_mapper={"H": "A", "L": "B"},
                                 chunk_size=config.get("chunk_size", 100), saprot_version="PDB")
        elif plm.lower() == "saprot-struct-pdb":
            enc = SaprotEncoding(use_structure="full", model_dir=config["local_saprot_dir_pdb"],
                                 dd_dict=dd_dict, chain_mapper={"H": "A", "L": "B"},
                                 chunk_size=config.get("chunk_size", 100), saprot_version="PDB")
        elif plm.lower() == "saprot-seq-af2":
            enc = SaprotEncoding(use_structure="seq", model_dir=config["local_saprot_dir_af2"],
                                 dd_dict=dd_dict, chain_mapper={"H": "A", "L": "B"},
                                 chunk_size=config.get("chunk_size", 100), saprot_version="AF2")
        elif plm.lower() == "saprot-struct-af2":
            enc = SaprotEncoding(use_structure="full", model_dir=config["local_saprot_dir_af2"],
                                 dd_dict=dd_dict, chain_mapper={"H": "A", "L": "B"},
                                 chunk_size=config.get("chunk_size", 100), saprot_version="AF2")
        else:
            logging.warning(f"Unknown encoding '{plm}' — skipping.")
            continue
        encodings.append(enc)
        encoding_names.append(str(enc))
    tim_model_paths = [
        tim_model_paths_raw[i] if i < len(tim_model_paths_raw) else None
        for i in range(len(encodings))
    ]
    return encodings, encoding_names, tim_model_paths


def get_pdb_list(data_dir, df):
    dataset = data_dir.split("/")[-1]
    if dataset == "bet_data":
        pdb_list = df["filename"]
        pdb_list = pdb_list.apply(lambda x: f"{data_dir}/pdb_files" + x.split("pdb_files")[-1])
    elif dataset == "csi_bli":
        pdb_list = df["Name"]
        pdb_list = pdb_list.apply(lambda x: f"{data_dir}/structures/pdb_symlinks/" + f"{x}.pdb")
    else:
        pdb_list = None
    return pdb_list


def generate_or_load_embeddings(encodings, encoding_names, df_train, df_test, df_val,
                                data_dir, train_stem=None, test_stem=None, val_stem=None,
                                seq_col="VH-VL"):
    enc_results_train, enc_results_test, enc_results_val = {}, {}, {}
    dataset_tag = Path(data_dir).name.lower()

    for i, encod in enumerate(encodings):
        logging.info(f"Processing encoding: {encoding_names[i]}")
        train_file_path = f"{data_dir}/encodings/{encoding_names[i]}_{train_stem}_{dataset_tag}.pt"
        test_file_path  = f"{data_dir}/encodings/{encoding_names[i]}_{test_stem}_{dataset_tag}.pt"
        val_file_path   = f"{data_dir}/encodings/{encoding_names[i]}_{val_stem}_{dataset_tag}.pt"
        mkdir_if_missing(train_file_path)

        if os.path.exists(train_file_path):
            logging.info(f"Loading train embeddings from {train_file_path}")
            embeddings = torch.load(train_file_path, map_location=torch.device("cpu"))
        else:
            logging.info(f"Generating train embeddings for {encoding_names[i]}")
            if isinstance(encod, SaprotEncoding) and encoding_names[i] in ["Saprot-struct-AF2", "Saprot-struct-PDB"]:
                pdb_list_train = get_pdb_list(data_dir, df_train)
                embeddings = encod.encode_sequences(
                    sequences=list(df_train[seq_col].values),
                    pdb_files=list(pdb_list_train),
                    data_dir=data_dir, split="train",
                )
            else:
                embeddings = encod.encode_sequences(list(df_train[seq_col].values))
            torch.save(embeddings, train_file_path)
        enc_results_train[encoding_names[i]] = embeddings
        del embeddings

        if os.path.exists(test_file_path):
            logging.info(f"Loading test embeddings from {test_file_path}")
            embeddings = torch.load(test_file_path, map_location=torch.device("cpu"))
        else:
            logging.info(f"Generating test embeddings for {encoding_names[i]}")
            if isinstance(encod, SaprotEncoding) and encoding_names[i] in ["Saprot-struct-AF2", "Saprot-struct-PDB"]:
                pdb_list_test = get_pdb_list(data_dir, df_test)
                embeddings = encod.encode_sequences(
                    sequences=list(df_test[seq_col].values),
                    pdb_files=list(pdb_list_test),
                    data_dir=data_dir, split="test",
                )
            else:
                embeddings = encod.encode_sequences(list(df_test[seq_col].values))
            torch.save(embeddings, test_file_path)
        enc_results_test[encoding_names[i]] = embeddings
        del embeddings

        if len(df_val) > 0:
            if os.path.exists(val_file_path):
                logging.info(f"Loading val embeddings from {val_file_path}")
                embeddings = torch.load(val_file_path, map_location=torch.device("cpu"))
            else:
                logging.info(f"Generating val embeddings for {encoding_names[i]}")
                if isinstance(encod, SaprotEncoding) and encoding_names[i] in ["Saprot-struct-AF2", "Saprot-struct-PDB"]:
                    pdb_list_val = get_pdb_list(data_dir, df_val)
                    embeddings = encod.encode_sequences(
                        sequences=list(df_val[seq_col].values),
                        pdb_files=list(pdb_list_val),
                        data_dir=data_dir, split="val",
                    )
                else:
                    embeddings = encod.encode_sequences(list(df_val[seq_col].values))
                torch.save(embeddings, val_file_path)
            enc_results_val[encoding_names[i]] = embeddings
            del embeddings

        gc.collect()
        _clear_cache(_get_device())
        del encod

    return enc_results_train, enc_results_test, enc_results_val


# ---------------------------------------------------------------------------
# TIM model helpers
# ---------------------------------------------------------------------------

def tim_model_ft(config, enc_results_train, enc_results_test, embed_name, y_s, y_q, device,
                 finetune=True, model_path=None):
    feature_s = enc_results_train[embed_name].unsqueeze(0).to(device)
    feature_q = enc_results_test[embed_name].unsqueeze(0).to(device)
    feature_s = F.normalize(feature_s, dim=2)
    feature_q = F.normalize(feature_q, dim=2)
    y_s = torch.tensor(y_s, dtype=torch.long).unsqueeze(0).to(device)
    y_q = torch.tensor(y_q, dtype=torch.long).unsqueeze(0).to(device)

    tim_model = TIM(config=config, device=device)

    time_ft = 0
    if finetune:
        start = time.time()
        _, _, s_probs, q_probs, model_path = tim_model.run_task(
            support=feature_s, query=feature_q, y_s=y_s, y_q=y_q,
            embed_name=embed_name, model_path=model_path,
        )
        time_ft = time.time() - start
        logging.info(f"Finetuning takes {time_ft:.3f} seconds")
        tim_model.plot_losses(embed_name=embed_name)

    start = time.time()
    logging.info(f"Running prediction for encoding: {embed_name}")
    tim_model_check(tim_model, feature_s, y_s)
    s_probs, q_probs, loss, ce, q_ent, q_cond_ent, support, query = tim_model.run_prediction(
        support=feature_s, query=feature_q, y_s=y_s, model_path=model_path,
    )
    time_infer = time.time() - start
    logging.info(f"Inference takes {time_infer:.3f} seconds")

    correct_mask = q_probs.argmax(2) == y_q
    return (
        s_probs.detach().squeeze(0),
        q_probs.detach().squeeze(0),
        model_path,
        loss, ce, q_ent, q_cond_ent,
        support, query,
        correct_mask,
        time_ft, time_infer,
    )


def tim_model_check(tim_model, feature_s, y_s):
    if not hasattr(tim_model, "weights"):
        tim_model.init_weights(support=feature_s, y_s=y_s)
    if not hasattr(tim_model, "fc"):
        tim_model.init_feature_transform(feature_s.shape[-1], ft=tim_model.ft)


def check_feat_str(update_feat, entropies, loss_weights, alphas=None):
    if alphas is None:
        alphas = [2.0, 2.0, 2.0]
    feat_type = "FT" if update_feat else ""
    alpha_marginal = entropies[1] == "Alpha" and abs(alphas[1] - 1.0) > 1e-6
    if loss_weights[1] == 0 and loss_weights[2] == 0:
        model_str = "CE"
    elif alpha_marginal:
        model_str = "Alpha-TIM"
    elif entropies[1] == "Shannon" and entropies[2] == "Alpha" and alphas[2] == 2:
        model_str = "GINI-TIM"
    elif all(e == "Shannon" for e in entropies[1:]):
        model_str = "TIM"
    else:
        model_str = "TIM"  # alpha=1 limit degrades to Shannon
    if feat_type:
        model_str = f"{feat_type}-{model_str}"
    return model_str


def _tim_str(config):
    return check_feat_str(
        config.get("update_feat", True),
        config.get("entropies", ["Shannon", "Shannon", "Shannon"]),
        config.get("loss_weights", [1.0, 0.5, 0.1]),
        config.get("alphas", [2.0, 2.0, 2.0]),
    )


# ---------------------------------------------------------------------------
# Multi-seed aggregation (shared by tim and lora_tim)
# ---------------------------------------------------------------------------

def _aggregate_seed_results(per_seed_results):
    agg = {}
    for split in ("train", "test", "val"):
        frames = [r[split] for r in per_seed_results if r and r.get(split) is not None]
        if not frames:
            agg[split] = None
            continue
        indexed = [df.set_index("Model") for df in frames]
        stacked  = pd.concat(indexed, keys=range(len(frames)), names=["seed_idx", "Model"])
        stacked  = stacked.select_dtypes(include="number")
        mean_df  = stacked.groupby("Model").mean().dropna(axis=1, how="all")
        std_df   = stacked.groupby("Model").std(ddof=1).fillna(0.0)[mean_df.columns]
        agg_df   = pd.DataFrame(index=mean_df.index, columns=mean_df.columns, dtype=object)
        for col in mean_df.columns:
            agg_df[col] = mean_df[col].map(lambda v: f"{v:.4f}") + " ± " + std_df[col].map(lambda v: f"{v:.4f}")
        agg[split] = agg_df.reset_index()
    return agg


def _print_agg_results(agg, seeds):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Aggregated results across {len(seeds)} seeds: {seeds}")
    print(sep)
    for split in ("train", "test", "val"):
        df = agg.get(split)
        if df is None:
            continue
        print(f"\n{split.capitalize()} (mean ± std across {len(seeds)} seeds):")
        print(df.to_string(index=False))
    print(sep + "\n")


def _save_agg_csv(agg, config, seeds, title=None):
    if title is None:
        title = check_feat_str(config["update_feat"], config["entropies"], config["loss_weights"], config.get("alphas", [2.0, 2.0, 2.0]))
    train_stem = config.get("train_stem", Path(config["train_file"]).stem)
    test_stem  = config.get("test_stem",  Path(config["test_file"]).stem)
    val_stem   = Path(config["val_file"]).stem if "val_file" in config and config["val_file"] else "noval"
    seeds_tag  = "-".join(str(s) for s in seeds)
    csv_path   = (
        f"{config['log_dir']}/results_table/"
        f"results_{train_stem}_{test_stem}_{val_stem}_{title}_seeds-{seeds_tag}_agg.csv"
    )
    mkdir_if_missing(csv_path)
    frames = []
    for split in ("train", "test", "val"):
        df = agg.get(split)
        if df is not None:
            frames.append(df.assign(split=split))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        front = ["Model", "split"]
        combined = combined[front + [c for c in combined.columns if c not in front]]
        combined.to_csv(csv_path, index=False)
        logging.info(f"Aggregated results saved to {csv_path}")


def _log_agg_to_wandb(agg):
    for split, df in agg.items():
        if df is None:
            continue
        for _, row in df.iterrows():
            model_name = row["Model"]
            for col in df.columns:
                if col == "Model":
                    continue
                parts = str(row[col]).split(" ± ")
                if len(parts) != 2:
                    continue
                try:
                    wandb.run.summary[f"agg/{split}/{model_name}/{col}_mean"] = float(parts[0])
                    wandb.run.summary[f"agg/{split}/{model_name}/{col}_std"]  = float(parts[1])
                except ValueError:
                    pass


# ---------------------------------------------------------------------------
# Shared results helper (lora modes)
# ---------------------------------------------------------------------------

def _log_and_save_results(train_results, test_results, val_results, config,
                          train_stem, test_stem, val_stem, title,
                          test_probs=None, val_probs=None, y_q=None, y_q_val=None):
    """Print, wandb-log, optionally plot PR curves, save CSV. Returns numeric DataFrames."""
    has_val = len(val_results) > 0

    def _fmt(rows):
        df = pd.DataFrame(rows)
        return df.applymap(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

    train_df = _fmt(train_results)
    test_df  = _fmt(test_results)
    print(f"\nTrain:\n{train_df}\n\nTest:\n{test_df}")
    if has_val:
        print(f"\nVal:\n{_fmt(val_results)}")

    if wandb.run is not None:
        for split, rows in [("train", train_results), ("test", test_results), ("val", val_results)]:
            for m in rows:
                name = m.get("Model", "")
                wandb.log({f"{split}/{name}/{k}": v for k, v in m.items() if k != "Model"})

    if config.get("plot_pr_curves", False) and test_probs is not None and y_q is not None:
        pr_test_path = (
            f"{config['log_dir']}/plots/{title}/"
            f"pr_curves_{train_stem}_{test_stem}_{title}.png"
        )
        plot_pr_curves(test_probs, y_q, pr_test_path, title=f"PR Curves — {title} (test)")
        if has_val and val_probs and y_q_val is not None:
            pr_val_path = (
                f"{config['log_dir']}/plots/{title}/"
                f"pr_curves_{train_stem}_{val_stem}_{title}_val.png"
            )
            plot_pr_curves(val_probs, y_q_val, pr_val_path, title=f"PR Curves — {title} (val)")

    seed_suffix = f"_seed-{config.get('seed', 2025)}" if config.get("_multi_seed") else ""
    csv_path = (
        f"{config['log_dir']}/results_table/"
        f"results_{train_stem}_{test_stem}_{val_stem}_{title}{seed_suffix}.csv"
    )
    mkdir_if_missing(csv_path)
    frames = [train_df.assign(split="train"), test_df.assign(split="test")]
    if has_val:
        frames.append(_fmt(val_results).assign(split="val"))
    combined = pd.concat(frames, ignore_index=True)
    front = ["Model", "split"]
    combined = combined[front + [c for c in combined.columns if c not in front]]
    combined.to_csv(csv_path, index=False)
    logging.info(f"Results saved to {csv_path}")

    return {
        "train": pd.DataFrame(train_results),
        "test":  pd.DataFrame(test_results),
        "val":   pd.DataFrame(val_results) if has_val else None,
    }


# ---------------------------------------------------------------------------
# Mode: tim
# ---------------------------------------------------------------------------

def run_tim(config):
    """Single-seed TIM run on frozen PLM embeddings."""
    seed = config.get("seed", 2025)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)

    logging.info(f"Running with config: {config}")
    df_train = load_data(f"{config['data_dir']}/{config['train_file']}")
    if "val_file" in config and config["val_file"]:
        df_val = load_data(f"{config['data_dir']}/{config['val_file']}")
    else:
        df_val = pd.DataFrame()
    df_test  = load_data(f"{config['data_dir']}/{config['test_file']}")
    has_val  = len(df_val) > 0

    encodings, encoding_names, tim_model_paths = initialize_encodings(config)
    train_stem = Path(config["train_file"]).stem
    val_stem   = Path(config["val_file"]).stem if "val_file" in config and config["val_file"] else "noval"
    test_stem  = Path(config["test_file"]).stem
    config["train_stem"] = train_stem
    config["test_stem"]  = test_stem

    enc_results_train, enc_results_test, enc_results_val = generate_or_load_embeddings(
        encodings=encodings, encoding_names=encoding_names,
        df_train=df_train, df_test=df_test, df_val=df_val,
        data_dir=config["data_dir"],
        train_stem=train_stem, val_stem=val_stem, test_stem=test_stem,
        seq_col=config.get("seq_col", "VH-VL"),
    )
    target_column = config["target_column"]
    y_s     = df_train[target_column].values
    y_q     = df_test[target_column].values
    y_q_val = df_val[target_column].values if has_val else np.array([])

    dataset = config["data_dir"].split("/")[-1]
    num_pos_train = int((y_s == 1).sum())
    num_neg_train = int((y_s == 0).sum())
    num_pos_test  = int((y_q == 1).sum())
    num_neg_test  = int((y_q == 0).sum())
    log_val_str = ""
    if has_val:
        num_pos_val = int((y_q_val == 1).sum())
        num_neg_val = int((y_q_val == 0).sum())
        log_val_str = (
            f"Num pos val: {num_pos_val}, Num neg val: {num_neg_val},\n"
            f"Neg % val: {num_neg_val/len(y_q_val): .3f}, Pos % val: {num_pos_val/len(y_q_val): .3f}\n"
        )
    logging.info(
        f"Dataset: {dataset}:\n"
        f"Num train: {len(y_s)}, Num test: {len(y_q)}\n"
        f"Num pos train: {num_pos_train}, Num neg train: {num_neg_train},\n"
        f"Num pos test: {num_pos_test}, Num neg test: {num_neg_test},\n"
        + log_val_str +
        f"Pos % train: {num_pos_train/len(y_s): .3f}, Pos % test: {num_pos_test/len(y_q): .3f},\n"
        f"Neg % train: {num_neg_train/len(y_s): .3f}, Neg % test: {num_neg_test/len(y_q): .3f},\n"
    )

    device = _get_device()
    test_results, train_results, val_results = [], [], []
    test_probs, val_probs = {}, {}
    title1 = check_feat_str(config["update_feat"], config["entropies"], config["loss_weights"], config.get("alphas", [2.0, 2.0, 2.0]))

    for i, embed_name in enumerate(encoding_names):
        _saved_path  = tim_model_paths[i] if i < len(tim_model_paths) else None
        _do_finetune = _saved_path is None
        logging.info(
            f"{'Fine-tuning' if _do_finetune else 'Loading saved TIM model for'} encoding: {embed_name}"
        )
        (
            y_s_probs, y_q_probs, model_path,
            loss, ce, q_ent, q_cond_ent,
            support, query, correct_mask,
            time_ft, time_infer,
        ) = tim_model_ft(
            config=config,
            enc_results_train=enc_results_train,
            enc_results_test=enc_results_test,
            embed_name=embed_name,
            y_s=y_s, y_q=y_q, device=device,
            finetune=_do_finetune, model_path=_saved_path,
        )

        if config.get("save_probs", False):
            probs_save_path = (
                f"{config['log_dir']}/results_table/"
                f"probs_{train_stem}_{test_stem}_{embed_name}_{title1}.npz"
            )
            mkdir_if_missing(probs_save_path)
            np.savez(
                probs_save_path,
                y_true=y_q,
                y_pred_probs=y_q_probs.cpu().numpy() if hasattr(y_q_probs, "cpu") else np.array(y_q_probs),
            )
            logging.info(f"Saved test probabilities to {probs_save_path}")

        if has_val:
            _val_ret = tim_model_ft(
                config=config,
                enc_results_train=enc_results_train,
                enc_results_test=enc_results_val,
                embed_name=embed_name,
                y_s=y_s, y_q=y_q_val, device=device,
                finetune=False, model_path=model_path,
            )
            y_val_probs, time_infer_val = _val_ret[1], _val_ret[11]
            metrics = calculate_metrics(y_q_val, y_val_probs)
            metrics["Model"] = embed_name
            metrics["Time infer"] = time_infer_val
            val_results.append(metrics)
            val_probs[embed_name] = y_val_probs

        metrics = calculate_metrics(y_q, y_q_probs)
        metrics["Model"] = embed_name
        metrics["Time infer"] = time_infer
        f1 = metrics["F1 Score"]
        test_results.append(metrics)
        test_probs[embed_name] = y_q_probs

        metrics = calculate_metrics(y_s, y_s_probs)
        metrics["Model"] = embed_name
        metrics["Time ft"] = time_ft
        train_results.append(metrics)

        title_plot = title1 + "\n F1: {:.2f}%".format(f1 * 100)
        savename = "{}/{}/{}/final-ft-{}.png".format(
            f"{config['log_dir']}/plots", title1, embed_name, config.get("ft", "mlp")
        )
        y_s_t = torch.tensor(y_s, dtype=torch.long).unsqueeze(0).to(device)
        y_q_t = torch.tensor(y_q, dtype=torch.long).unsqueeze(0).to(device)
        plot_tsne(
            torch.cat((support.squeeze(), query.squeeze()), dim=0),
            torch.cat((y_s_t, y_q_t), dim=1),
            support_set=support.size(1),
            save_name=savename,
            correct_mask=correct_mask,
            title=title_plot,
            pred_probs=y_q_probs,
        )

        train_m = train_results[-1]
        test_m  = test_results[-1]
        val_str = (
            f"Val   F1={val_results[-1]['F1 Score']:.3f}  AUC-PR={val_results[-1]['AUC-PR']:.3f}"
            if has_val else "Val   (skipped)"
        )
        print(
            f"\n[{embed_name}]  "
            f"Train F1={train_m['F1 Score']:.3f}  AUC-PR={train_m['AUC-PR']:.3f}  |  "
            f"Test  F1={test_m['F1 Score']:.3f}  AUC-PR={test_m['AUC-PR']:.3f}  |  "
            f"{val_str}"
        )

        enc_results_train.pop(embed_name, None)
        enc_results_test.pop(embed_name, None)
        enc_results_val.pop(embed_name, None)
        gc.collect()
        _clear_cache(device)

    if config.get("plot_pr_curves", False):
        pr_test_path = (
            f"{config['log_dir']}/plots/{title1}/"
            f"pr_curves_{train_stem}_{test_stem}_{title1}.png"
        )
        plot_pr_curves(test_probs, y_q, pr_test_path, title=f"PR Curves — {title1} (test)")
        if has_val and val_probs:
            pr_val_path = (
                f"{config['log_dir']}/plots/{title1}/"
                f"pr_curves_{train_stem}_{val_stem}_{title1}_val.png"
            )
            plot_pr_curves(val_probs, y_q_val, pr_val_path, title=f"PR Curves — {title1} (val)")

    train_results_df_num = pd.DataFrame(train_results)
    test_results_df_num  = pd.DataFrame(test_results)
    val_results_df_num   = pd.DataFrame(val_results) if has_val else None

    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else x  # noqa: E731
    train_results_df = train_results_df_num.copy().applymap(fmt)
    test_results_df  = test_results_df_num.copy().applymap(fmt)
    print("Train results:\n{}\n".format(train_results_df))
    print("Test results:\n{}".format(test_results_df))
    if has_val:
        val_results_df = val_results_df_num.copy().applymap(fmt)
        print("Validation results:\n{}".format(val_results_df))

    if wandb.run is not None:
        for split, results in zip(["train", "test", "val"], [train_results, test_results, val_results]):
            for m in results:
                model_name = m.get("Model", "")
                wandb.log({f"{split}/{model_name}/{k}": v for k, v in m.items() if k != "Model"})

    seed_suffix = f"_seed-{config.get('seed', 2025)}" if config.get("_multi_seed") else ""
    results_csv_path = (
        f"{config['log_dir']}/results_table/"
        f"results_{train_stem}_{test_stem}_{val_stem}_{title1}{seed_suffix}.csv"
    )
    mkdir_if_missing(results_csv_path)
    _frames = [train_results_df.assign(split="train"), test_results_df.assign(split="test")]
    if has_val:
        _frames.append(val_results_df.assign(split="val"))
    combined_df = pd.concat(_frames, ignore_index=True)
    front_cols  = ["Model", "split"]
    combined_df = combined_df[front_cols + [c for c in combined_df.columns if c not in front_cols]]
    combined_df.to_csv(results_csv_path, index=False)
    logging.info(f"Results saved to {results_csv_path}")

    return {"train": train_results_df_num, "test": test_results_df_num, "val": val_results_df_num}


def run_tim_multi_seed(config):
    """Run run_tim() once per seed; aggregate mean ± std if multiple seeds given."""
    seeds = config.get("seeds") or [config.get("seed", 2025)]
    if len(seeds) == 1:
        cfg = {k: v for k, v in config.items() if k != "_multi_seed"}
        cfg["seed"] = seeds[0]
        run_tim(cfg)
        return
    per_seed = []
    for idx, seed in enumerate(seeds):
        logging.info(f"=== TIM Seed run {idx + 1}/{len(seeds)}: seed={seed} ===")
        seed_cfg = dict(config)
        seed_cfg["seed"] = seed
        seed_cfg["_multi_seed"] = True
        per_seed.append(run_tim(seed_cfg))
    agg = _aggregate_seed_results(per_seed)
    _print_agg_results(agg, seeds)
    _save_agg_csv(agg, config, seeds)
    if wandb.run is not None:
        _log_agg_to_wandb(agg)


# ---------------------------------------------------------------------------
# Mode: lora_tim
# ---------------------------------------------------------------------------

def _run_lora_tim_single(config, df_train, df_test, df_val, encodings, encoding_names, device):
    """Single-seed lora_tim: Phase 1 uses/caches LoRA embeddings; Phase 2 runs TIM."""
    seed = config.get("seed", 2025)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)

    train_stem = Path(config["train_file"]).stem
    test_stem  = Path(config["test_file"]).stem
    val_stem   = Path(config["val_file"]).stem if "val_file" in config and config["val_file"] else "noval"
    config["train_stem"] = train_stem
    config["test_stem"]  = test_stem
    target_col = config["target_column"]
    has_val    = len(df_val) > 0

    y_s     = df_train[target_col].values
    y_q     = df_test[target_col].values
    y_q_val = df_val[target_col].values if has_val else np.array([])

    title = "LoRA-TIM-" + _tim_str(config)
    train_results, test_results, val_results = [], [], []
    test_probs, val_probs = {}, {}

    for encod in encodings:
        if hasattr(encod, 'model'):
            encod.model.cpu()
    gc.collect()
    _clear_cache(device)

    for encod, enc_name in zip(encodings, encoding_names):
        if hasattr(encod, 'model'):
            encod.model.to(device)
        # Phase 1: lora_finetune_plm handles caching internally —
        # trains once, then loads from cache on subsequent seed calls.
        lora_result = lora_finetune_plm(
            encoding=encod, df_train=df_train, df_test=df_test, df_val=df_val,
            config=config, device=device,
        )
        if hasattr(encod, 'model'):
            encod.model.cpu()
        gc.collect()
        _clear_cache(device)
        emb_train = lora_result["embeddings_train"]
        emb_test  = lora_result["embeddings_test"]
        emb_val   = lora_result["embeddings_val"]

        enc_train_lora = {enc_name: emb_train}
        enc_test_lora  = {enc_name: emb_test}
        enc_val_lora   = {enc_name: emb_val} if (has_val and emb_val.numel() > 0) else {}

        # Phase 2: TIM on LoRA embeddings
        ret = tim_model_ft(
            config=config,
            enc_results_train=enc_train_lora,
            enc_results_test=enc_test_lora,
            embed_name=enc_name,
            y_s=y_s, y_q=y_q, device=device,
            finetune=True, model_path=None,
        )
        (y_s_probs, y_q_probs, model_path, loss, ce, q_ent, q_cond_ent,
         support, query, correct_mask, time_ft, time_infer) = ret

        if has_val and enc_val_lora:
            val_ret = tim_model_ft(
                config=config,
                enc_results_train=enc_train_lora,
                enc_results_test=enc_val_lora,
                embed_name=enc_name,
                y_s=y_s, y_q=y_q_val, device=device,
                finetune=False, model_path=model_path,
            )
            m_val = calculate_metrics(y_q_val, val_ret[1])
            m_val["Model"] = enc_name
            m_val["Time infer"] = val_ret[11]
            val_results.append(m_val)
            val_probs[enc_name] = val_ret[1]

        m_test = calculate_metrics(y_q, y_q_probs)
        m_test["Model"] = enc_name
        m_test["Time infer"] = time_infer
        test_results.append(m_test)
        test_probs[enc_name] = y_q_probs

        m_train = calculate_metrics(y_s, y_s_probs)
        m_train["Model"] = enc_name
        m_train["Time ft"] = time_ft
        train_results.append(m_train)

        print(f"\n[{enc_name}] LoRA-TIM Test F1={m_test['F1 Score']:.3f} AUC-PR={m_test['AUC-PR']:.3f}")

    return _log_and_save_results(
        train_results, test_results, val_results, config,
        train_stem, test_stem, val_stem, title,
        test_probs=test_probs,
        val_probs=val_probs if has_val else None,
        y_q=y_q,
        y_q_val=y_q_val if has_val else None,
    )


def run_lora_tim(config, df_train, df_test, df_val, encodings, encoding_names, device):
    """LoRA fine-tune PLM → cache embeddings → TIM; multi-seed on TIM stage if seeds given."""
    seeds = config.get("seeds") or [config.get("seed", 2025)]
    if len(seeds) == 1:
        seed_cfg = dict(config)
        seed_cfg["seed"] = seeds[0]
        _run_lora_tim_single(seed_cfg, df_train, df_test, df_val, encodings, encoding_names, device)
        return
    per_seed = []
    for idx, seed in enumerate(seeds):
        logging.info(f"=== LoRA-TIM Seed run {idx + 1}/{len(seeds)}: seed={seed} ===")
        seed_cfg = dict(config)
        seed_cfg["seed"] = seed
        seed_cfg["_multi_seed"] = True
        per_seed.append(
            _run_lora_tim_single(seed_cfg, df_train, df_test, df_val, encodings, encoding_names, device)
        )
    agg   = _aggregate_seed_results(per_seed)
    title = "LoRA-TIM-" + _tim_str(config)
    _print_agg_results(agg, seeds)
    _save_agg_csv(agg, config, seeds, title=title)
    if wandb.run is not None:
        _log_agg_to_wandb(agg)


# ---------------------------------------------------------------------------
# Mode: lora_ce
# ---------------------------------------------------------------------------

def run_lora_ce(config, df_train, df_test, df_val, encodings, encoding_names, device):
    """LoRA-fine-tuned PLM + direct CE classifier; single seed."""
    train_stem = Path(config["train_file"]).stem
    test_stem  = Path(config["test_file"]).stem
    val_stem   = Path(config["val_file"]).stem if "val_file" in config and config["val_file"] else "noval"
    config["train_stem"] = train_stem
    config["test_stem"]  = test_stem
    target_col = config["target_column"]

    y_s = df_train[target_col].values
    y_q = df_test[target_col].values

    title = "LoRA-CE"
    train_results, test_results, val_results = [], [], []
    test_probs: dict = {}

    for encod in encodings:
        if hasattr(encod, 'model'):
            encod.model.cpu()
    gc.collect()
    _clear_cache(device)

    for encod, enc_name in zip(encodings, encoding_names):
        if hasattr(encod, 'model'):
            encod.model.to(device)
        t0 = time.time()
        result = lora_finetune_plm(
            encoding=encod, df_train=df_train, df_test=df_test, df_val=df_val,
            config=config, device=device,
        )
        time_ft = time.time() - t0
        if hasattr(encod, 'model'):
            encod.model.cpu()
        gc.collect()
        _clear_cache(device)

        t1 = time.time()
        ce_probs_test  = result["ce_probs_test"]
        ce_probs_train = result["ce_probs_train"]
        time_infer = time.time() - t1

        m_test = calculate_metrics(y_q, ce_probs_test)
        m_test["Model"] = enc_name
        m_test["Time infer"] = time_infer
        test_results.append(m_test)
        test_probs[enc_name] = ce_probs_test

        m_train = calculate_metrics(y_s, ce_probs_train)
        m_train["Model"] = enc_name
        m_train["Time ft"] = time_ft
        train_results.append(m_train)

        print(f"\n[{enc_name}] LoRA-CE Test F1={m_test['F1 Score']:.3f} AUC-PR={m_test['AUC-PR']:.3f}")

    _log_and_save_results(
        train_results, test_results, val_results, config,
        train_stem, test_stem, val_stem, title,
        test_probs=test_probs, val_probs=None,
        y_q=y_q, y_q_val=None,
    )


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

def main(config):
    setup_working_directory()
    seed = config.get("seed", 2025)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    np.random.seed(seed)

    mode = config.get("mode", "tim")
    logging.info(f"benchmark_all — mode={mode}")
    logging.info(f"Running with config: {config}")

    if mode == "tim":
        # run_tim handles data loading internally per seed call
        run_tim_multi_seed(config)
        return

    df_train = load_data(f"{config['data_dir']}/{config['train_file']}")
    if "val_file" in config and config["val_file"]:
        df_val = load_data(f"{config['data_dir']}/{config['val_file']}")
    else:
        df_val = pd.DataFrame()
    df_test  = load_data(f"{config['data_dir']}/{config['test_file']}")
    encodings, encoding_names, _ = initialize_encodings(config)
    device = _get_device()
    logging.info(f"Using device: {device}")

    if mode == "lora_ce":
        run_lora_ce(config, df_train, df_test, df_val, encodings, encoding_names, device)
    elif mode == "lora_tim":
        run_lora_tim(config, df_train, df_test, df_val, encodings, encoding_names, device)
    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: tim, lora_ce, lora_tim")


def _construct_run_id(config):
    model_str = check_feat_str(
        update_feat=config["update_feat"],
        entropies=config["entropies"],
        loss_weights=config["loss_weights"],
        alphas=config.get("alphas", [2.0, 2.0, 2.0]),
    )
    timestamp = os.environ.get(RUN_STARTED_TIMESTAMP)
    return f"{model_str}_{timestamp}"


def run(config):
    if "wandb" in config:
        _run_with_wandb(config)
    else:
        main(config)


def _run_with_wandb(config):
    wandb_config = config["wandb"]
    with wandb.init(config=config, **wandb_config):
        config = wandb.config
        main(config)


@click.command()
@click.option("-c", "--config", help="Path to a YAML config file.")
@click.option("-i", "--run-id", default=None, help="Run ID for logging.")
def _main(config: str, run_id: str):
    config = load_yaml_config(config)
    mode   = config.get("mode", "tim")
    if run_id is None:
        timestamp = os.environ.get(RUN_STARTED_TIMESTAMP, "0")
        run_id = _construct_run_id(config) if mode == "tim" else f"{mode}_{timestamp}"
    set_basic_logging_config(run_id, config["log_dir"], verbose=True)
    logging.info(f"Starting benchmark_all run: {run_id}")
    config.update(dict(run_id=run_id))
    if "wandb" in config:
        config["wandb"]["group"] = run_id.replace("_", "-")
    run(config)


if __name__ == "__main__":
    if os.environ.get(RUN_STARTED_TIMESTAMP) is None:
        os.environ[RUN_STARTED_TIMESTAMP] = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    _main()
