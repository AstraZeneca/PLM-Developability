"""Zero-shot WT-relative masked pseudo-log-likelihood benchmark.

Scores every sequence in each split with:
  LLR(v) = sum_{i in mutated(v)} [log P(v_i | v_{-i}) - log P(WT_i | WT_{-i})]

No training is performed.  The WT sequence must be provided via the config key 'wt_seq'.
Metrics: AUC-ROC, AUC-PR, Spearman correlation (all computed on raw LLR scores).
"""
import gc
import logging
import os
import random
import time
import warnings
from pathlib import Path

import click
import wandb

warnings.filterwarnings("ignore")

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

from plm_developability.encodings import (
    ESMEncoding,
    OneHotEncoding,
    SaprotEncoding,
)
from plm_developability.utils_basic import (
    load_yaml_config,
    mkdir_if_missing,
    set_basic_logging_config,
)

RUN_STARTED_TIMESTAMP = "RUN_STARTED_TIMESTAMP"


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


def setup_working_directory(target_folder="PLM-Developability"):
    current_dir = os.getcwd()
    while target_folder not in os.listdir(current_dir):
        current_dir = os.path.dirname(current_dir)
    os.chdir(os.path.join(current_dir, target_folder))
    logging.info(f"Working directory set to: {os.getcwd()}")


def load_data(file_path):
    return pd.read_csv(file_path)


def initialize_encodings(config):
    """Initialize only PLM encodings that support masked LM scoring."""
    device = _get_device()
    dd_dict = {"device": device, "dtype": torch.float32}
    encodings = []
    encoding_names = []

    esm_model_map = {
        "t6-8m": "esm2_t6_8M_UR50D",
        "t12-35m": "esm2_t12_35M_UR50D",
        "t30-150m": "esm2_t30_150M_UR50D",
        "t33-650m": "esm2_t33_650M_UR50D",
        "t36-3b": "esm2_t36_3B_UR50D",
        "t48-15b": "esm2_t48_15B_UR50D",
    }
    plm_list = config.get("plms", [])
    for plm in plm_list:
        if plm.lower() == "onehot":
            logging.info(f"Skipping '{plm}': no masked LM scoring available.")
            continue
        if plm.lower().startswith("esm"):
            model_version = config.get("model_version", "esm2_t30_150M_UR50D")
            if "." in plm:
                tag = plm.split(".", 1)[1].replace("_", "-").lower()
                model_version = esm_model_map.get(tag, model_version)
            enc = ESMEncoding(dd_dict=dd_dict, model_version=model_version, chunk_size=config.get("chunk_size", 10))
        elif plm.lower() == "saprot-struct-af2":
            enc = SaprotEncoding(
                use_structure="full", model_dir=config["local_saprot_dir_af2"],
                dd_dict=dd_dict, chain_mapper={"H": "A", "L": "B"},
                chunk_size=config.get("chunk_size", 100), saprot_version="AF2",
            )
        elif plm.lower() == "saprot-seq-af2":
            enc = SaprotEncoding(
                use_structure="seq", model_dir=config["local_saprot_dir_af2"],
                dd_dict=dd_dict, chain_mapper={"H": "A", "L": "B"},
                chunk_size=config.get("chunk_size", 100), saprot_version="AF2",
            )
        elif plm.lower() == "saprot-struct-pdb":
            enc = SaprotEncoding(
                use_structure="full", model_dir=config["local_saprot_dir_pdb"],
                dd_dict=dd_dict, chain_mapper={"H": "A", "L": "B"},
                chunk_size=config.get("chunk_size", 100), saprot_version="PDB",
            )
        elif plm.lower() == "saprot-seq-pdb":
            enc = SaprotEncoding(
                use_structure="seq", model_dir=config["local_saprot_dir_pdb"],
                dd_dict=dd_dict, chain_mapper={"H": "A", "L": "B"},
                chunk_size=config.get("chunk_size", 100), saprot_version="PDB",
            )
        else:
            logging.warning(f"Unknown encoding '{plm}' — skipping.")
            continue
        encodings.append(enc)
        encoding_names.append(str(enc))
    return encodings, encoding_names


def score_or_load_scores(encodings, encoding_names, df_train, df_test, df_val,
                         wt_seq, data_dir, train_stem, test_stem, val_stem):
    """Score all splits for each encoding, caching results to disk.

    Returns score dicts and timing dicts (seconds; 0.0 when loaded from cache).
    """
    results_train, results_test, results_val = {}, {}, {}
    times_train,   times_test,   times_val   = {}, {}, {}

    for i, encod in enumerate(encodings):
        name = encoding_names[i]
        logging.info(f"Scoring with encoding: {name}")

        train_path = f"{data_dir}/scores/{name}_{train_stem}_zeroshot.pt"
        test_path  = f"{data_dir}/scores/{name}_{test_stem}_zeroshot.pt"
        val_path   = f"{data_dir}/scores/{name}_{val_stem}_zeroshot.pt"
        mkdir_if_missing(train_path)

        for split_name, path, df in [
            ("train", train_path, df_train),
            ("test",  test_path,  df_test),
            ("val",   val_path,   df_val),
        ]:
            if os.path.exists(path):
                logging.info(f"  Loading {split_name} scores from {path}")
                scores = torch.load(path, map_location="cpu")
                elapsed = 0.0
            else:
                logging.info(f"  Computing {split_name} scores for {name}")
                t0 = time.perf_counter()
                scores = encod.score_sequences(list(df["VH-VL"].values), wt_seq)
                elapsed = time.perf_counter() - t0
                logging.info(f"  Done in {elapsed:.1f}s")
                torch.save(scores, path)

            if split_name == "train":
                results_train[name] = scores
                times_train[name]   = elapsed
            elif split_name == "test":
                results_test[name] = scores
                times_test[name]   = elapsed
            else:
                results_val[name] = scores
                times_val[name]   = elapsed

        gc.collect()
        _clear_cache(_get_device())
        del encod

    return results_train, results_test, results_val, times_train, times_test, times_val


def find_optimal_threshold(y_binary_train, llr_train):
    """Find the LLR threshold that maximises F1 on the training set."""
    from sklearn.metrics import precision_recall_curve

    scores_np = llr_train.cpu().numpy() if hasattr(llr_train, "cpu") else np.array(llr_train)
    precisions, recalls, thresholds = precision_recall_curve(y_binary_train, scores_np)
    f1_values = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = int(np.argmax(f1_values[:-1]))
    return float(thresholds[best_idx])


def compute_zeroshot_metrics(y_binary, y_scores, llr_scores, threshold):
    """Compute zero-shot metrics from continuous LLR scores.

    Args:
        y_binary: binary 0/1 ground-truth labels (for AUC and classification metrics).
        y_scores: continuous ground-truth scores (for Spearman); pass None to skip.
        llr_scores: LLR tensor/array from the model.
        threshold: classification threshold learned from the training set.

    Returns dict with AUC-ROC, AUC-PR, Spearman, F1, Precision, Recall, Accuracy, MCC, FPR, Threshold.
    """
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from scipy.stats import spearmanr

    scores_np = llr_scores.cpu().numpy() if hasattr(llr_scores, "cpu") else np.array(llr_scores)
    y_bin = np.array(y_binary)

    auc_roc      = roc_auc_score(y_bin, scores_np)
    auc_roc_flip = roc_auc_score(y_bin, -scores_np)
    auc_pr       = average_precision_score(y_bin, scores_np)
    auc_pr_flip  = average_precision_score(y_bin, -scores_np)

    flipped = auc_roc_flip > auc_roc
    best_auc_roc = auc_roc_flip if flipped else auc_roc
    best_auc_pr  = auc_pr_flip  if flipped else auc_pr

    # Apply threshold (flip direction if needed)
    if flipped:
        y_pred = (scores_np < threshold).astype(int)
    else:
        y_pred = (scores_np >= threshold).astype(int)

    f1        = f1_score(y_bin, y_pred, zero_division=0)
    precision = precision_score(y_bin, y_pred, zero_division=0)
    recall    = recall_score(y_bin, y_pred, zero_division=0)
    accuracy  = float(np.mean(y_bin == y_pred))
    mcc       = matthews_corrcoef(y_bin, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_bin, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    result = {
        "AUC-ROC":      auc_roc,
        "AUC-ROC-flip": auc_roc_flip,
        "AUC-ROC-best": best_auc_roc,
        "AUC-PR":       auc_pr,
        "AUC-PR-best":  best_auc_pr,
        "F1":           f1,
        "Precision":    precision,
        "Recall":       recall,
        "Accuracy":     accuracy,
        "MCC":          mcc,
        "FPR":          fpr,
        "Threshold":    threshold,
    }

    if y_scores is not None:
        spearman, _ = spearmanr(np.array(y_scores), scores_np)
        result["Spearman"] = spearman

    return result


def main(config):
    setup_working_directory()
    seed = 2025
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        cudnn.deterministic = True
    np.random.seed(seed)

    logging.info(f"Running zero-shot benchmark with config: {config}")

    df_train = load_data(f"{config['data_dir']}/{config['train_file']}")
    df_val   = load_data(f"{config['data_dir']}/{config['val_file']}")
    df_test  = load_data(f"{config['data_dir']}/{config['test_file']}")

    wt_seq = config["wt_seq"]  # required key: full VH-VL WT sequence

    encodings, encoding_names = initialize_encodings(config)
    train_stem = Path(config["train_file"]).stem
    val_stem   = Path(config["val_file"]).stem
    test_stem  = Path(config["test_file"]).stem

    results_train, results_test, results_val, times_train, times_test, times_val = score_or_load_scores(
        encodings=encodings,
        encoding_names=encoding_names,
        df_train=df_train,
        df_test=df_test,
        df_val=df_val,
        wt_seq=wt_seq,
        data_dir=config["data_dir"],
        train_stem=train_stem,
        test_stem=test_stem,
        val_stem=val_stem,
    )

    binary_col = config["target_column_binary"]
    scores_col = config.get("target_column_scores")

    y_train_binary = df_train[binary_col].values
    y_test_binary  = df_test[binary_col].values
    y_val_binary   = df_val[binary_col].values

    y_train_scores = df_train[scores_col].values if scores_col else None
    y_test_scores  = df_test[scores_col].values  if scores_col else None
    y_val_scores   = df_val[scores_col].values   if scores_col else None

    train_rows, test_rows, val_rows = [], [], []
    for name in encoding_names:
        threshold = find_optimal_threshold(y_train_binary, results_train[name])
        logging.info(f"[{name}] optimal threshold (train): {threshold:.4f}")

        for split, scores_dict, times_dict, y_binary, y_scores, rows in [
            ("train", results_train, times_train, y_train_binary, y_train_scores, train_rows),
            ("test",  results_test,  times_test,  y_test_binary,  y_test_scores,  test_rows),
            ("val",   results_val,   times_val,   y_val_binary,   y_val_scores,   val_rows),
        ]:
            metrics = compute_zeroshot_metrics(y_binary, y_scores, scores_dict[name], threshold)
            metrics["time_llr"] = times_dict.get(name, 0.0)
            metrics["Model"] = name
            rows.append(metrics)

            auc = metrics["AUC-ROC-best"]
            f1  = metrics["F1"]
            spe = metrics.get("Spearman", float("nan"))
            logging.info(f"[{name}] {split}: AUC-ROC-best={auc:.3f}  F1={f1:.3f}  Spearman={spe:.3f}")

    # Pretty-print
    for label, rows in [("Train", train_rows), ("Test", test_rows), ("Val", val_rows)]:
        df_r = pd.DataFrame(rows)
        df_r = df_r.applymap(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
        print(f"\n{label} results:\n{df_r}")

    # Log to W&B
    if wandb.run is not None:
        for split, rows in [("train", train_rows), ("test", test_rows), ("val", val_rows)]:
            for m in rows:
                model_name = m.pop("Model", "")
                log_dict = {f"{split}/{model_name}/{k}": v for k, v in m.items()}
                wandb.log(log_dict)

    # Save combined CSV
    results_csv = (
        f"{config['log_dir']}/results_table/"
        f"results_zeroshot_{train_stem}_{test_stem}_{val_stem}.csv"
    )
    mkdir_if_missing(results_csv)
    combined_df = pd.concat(
        [
            pd.DataFrame(train_rows).assign(split="train"),
            pd.DataFrame(test_rows).assign(split="test"),
            pd.DataFrame(val_rows).assign(split="val"),
        ],
        ignore_index=True,
    )
    front_cols = ["Model", "split"]
    rest_cols  = [c for c in combined_df.columns if c not in front_cols]
    combined_df = combined_df[front_cols + rest_cols]
    combined_df.to_csv(results_csv, index=False)
    logging.info(f"Results saved to {results_csv}")


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
@click.option("-c", "--config", help="Path to a config yaml file.")
@click.option("-i", "--run-id", default=None, help="Run id for logging.")
def _main(config: str, run_id: str):
    config = load_yaml_config(config)
    if run_id is None:
        timestamp = os.environ.get(RUN_STARTED_TIMESTAMP, "0")
        run_id = f"zero-shot_{timestamp}"
    set_basic_logging_config(run_id, config["log_dir"], verbose=True)
    logging.info(f"Starting zero-shot run with id: {run_id}")
    config.update(dict(run_id=run_id))
    if "wandb" in config:
        config["wandb"]["group"] = run_id.replace("_", "-")
    run(config)


if __name__ == "__main__":
    if os.environ.get(RUN_STARTED_TIMESTAMP) is None:
        os.environ[RUN_STARTED_TIMESTAMP] = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    _main()
