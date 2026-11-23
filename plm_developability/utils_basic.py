"""Import packages."""
# Utility
import errno
import json
import logging
import math
import os
import sys
import time
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import torch
import torch.nn.functional as F
import yaml

# joblib parallelism removed: using single-process batched loop instead
from Levenshtein import distance
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    pairwise_distances,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm


def levenshtein_distance(df, column='VH-VL', column_heavy=None, column_light=None):
    """calculate levenshtein distance squared matrix for antibody sequences.

    Args:
        df (pd.DataFrame): DataFrame containing sequences
        column_heavy (str, optional): column name of VH sequences. Defaults to "VH".
        column_light (str, optional): column name of VL sequences. Defaults to "VL".

    Returns
    -------
        np.array: squared distance matrix
    """  # noqa: D403
      # noqa: W293
    if column_light:
        sequence_df = df[[column_heavy, column_light]].copy()

        sequence_df[column] = sequence_df[column_heavy] + sequence_df[column_light]
    else:
        sequence_df = df[[column]].copy()
        sequence_df[column] = sequence_df[column]

    # pairwise Levenshtein distance
    distance_seq = np.zeros([len(sequence_df), len(sequence_df)])
    for i, r1 in sequence_df.iterrows():
        for j, r2 in sequence_df.iterrows():
            if i > j:
                distance_seq[i, j] = distance(r1[column], r2[column])
                distance_seq[j, i] = distance_seq[i, j]
    return distance_seq


def get_one_hot(y_s, num_classes=None):  # noqa: D103
    if num_classes is None:
        num_classes = torch.unique(y_s).size(0)
    eye = torch.eye(num_classes).to(y_s.device)
    one_hot = []
    for y_task in y_s:
        one_hot.append(eye[y_task].unsqueeze(0))
    one_hot = torch.cat(one_hot, 0)
    return one_hot

def get_mi(probs):  # noqa: D103
    q_cond_ent = get_cond_entropy(probs)
    q_ent = get_entropy(probs)
    return q_ent - q_cond_ent


def get_entropy(probs):  # noqa: D103
    q_ent = - (probs.mean(1) * torch.log(probs.mean(1) + 1e-12)).sum(1, keepdim=True)
    return q_ent


def get_cond_entropy(probs):  # noqa: D103
    q_cond_ent = - (probs * torch.log(probs + 1e-12)).sum(2).mean(1, keepdim=True)
    return q_cond_ent


def get_metric(metric_type):  # noqa: D103
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]

def plot_tsne(
    all_features, all_labels, support_set, save_name,
    correct_mask=None, title="",
    df_train=None, df_test=None, n_neighbors=5, annotate_column=None,
    pred_probs=None
):
    """
    Plots t-SNE of features. If df_train/df_test are provided, saves separate annotated
    plots for each incorrect sample and its neighbors, overlaying them on the main plot.
    All plots use the same axis limits, point sizes, and legend placement for consistency.
    Also saves a JSON file with the annotate_column, index, and Levenshtein distance for each plot.
    If correct_mask is provided: shows test-only view — points colored by true label, red X on
    incorrect predictions. Train points are omitted to reduce clutter.
    pred_probs is accepted but unused (reserved for future use).
    """  # noqa: D205, D401
    # --- Main t-SNE plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    feats_np = all_features.cpu().detach().numpy()
    n_pca = min(50, feats_np.shape[0] - 1, feats_np.shape[1])
    feats_reduced = PCA(n_components=n_pca, random_state=0).fit_transform(feats_np)
    X_emb = TSNE(n_components=2, perplexity=30, metric="cosine",
                 max_iter=2000, random_state=0).fit_transform(feats_reduced)

    if correct_mask is not None:
        correct_mask = correct_mask.cpu().numpy() if hasattr(correct_mask, "cpu") else np.array(correct_mask)
        if len(correct_mask.shape) == 2:
            correct_mask = correct_mask.squeeze(0)

    X_emb_support = X_emb[:support_set]
    X_emb_query = X_emb[support_set:]
    all_labels = all_labels.cpu().detach().numpy() if hasattr(all_labels, "cpu") else np.array(all_labels)
    support_labels = all_labels[0, :support_set]
    query_labels = all_labels[0, support_set:]
    # colors = ['blue', 'yellow']
    colors = ['#56B4E9', '#E69F00']
    markers = ['o', 's']
    labels = ['negative', 'positive']
    legend_elements = []
    point_size = 12  # Consistent point size for all plots

    if correct_mask is not None:
        # Test-only: color by true label, red X on incorrect predictions
        for i in range(len(colors)):
            tmp_q = (query_labels == i)
            sc = ax.scatter(
                X_emb_query[tmp_q, 0], X_emb_query[tmp_q, 1],
                s=point_size, c=colors[i], alpha=0.85,
                marker=markers[i], edgecolors='none', zorder=2,
                label=f'Test {labels[i]}'
            )
            legend_elements.append(sc)
        incorrect_mask = np.logical_not(correct_mask)
        if incorrect_mask.any():
            import matplotlib.lines as mlines
            ax.scatter(
                X_emb_query[incorrect_mask, 0], X_emb_query[incorrect_mask, 1],
                s=point_size * 3, c='red', linewidths=1.5,
                marker='x', zorder=3
            )
            legend_elements.append(
                mlines.Line2D([], [], marker='x', color='red', linestyle='None',
                              markersize=7, markeredgewidth=1.5, label='Incorrect')
            )
    else:
        for i in range(len(colors)):
            tmp_s = (support_labels == i)
            support_scatter = ax.scatter(
                X_emb_support[tmp_s, 0], X_emb_support[tmp_s, 1],
                s=point_size, edgecolors='k', c=colors[i],
                marker=markers[i], label=f'Train {labels[i]}'
            )
            legend_elements.append(support_scatter)

            tmp_q = (query_labels == i)
            query_scatter = ax.scatter(
                X_emb_query[tmp_q, 0], X_emb_query[tmp_q, 1],
                s=point_size * 0.6, c=colors[i], alpha=0.4,
                marker=markers[i], label=f'Test {labels[i]}'
            )
            legend_elements.append(query_scatter)

    ax.set_title(title)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    mkdir_if_missing(save_name)
    plt.savefig(save_name, bbox_inches='tight', dpi=300)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plt.close(fig)

    # --- Overlayed Neighborhood plots ---
    if correct_mask is not None and df_train is not None and df_test is not None:
        for i in range(len(colors)):
            tmp_q = (query_labels == i)
            tmp_q_incorrect = np.logical_and(tmp_q, np.logical_not(correct_mask))
            incorrect_indices = np.where(tmp_q_incorrect)[0]
            for idx in incorrect_indices:
                tsne_idx = support_set + idx  # index in X_emb
                dists = pairwise_distances([X_emb[tsne_idx]], X_emb)
                nearest_indices = np.argsort(dists[0])[1:n_neighbors+1]  # skip self

                # Prepare annotation text and info for JSON
                if annotate_column is not None and annotate_column in df_test.columns:
                    txt = str(df_test.iloc[idx][annotate_column])
                else:
                    txt = str(idx)

                # vh_incorrect = df_test.iloc[idx]["VH"]
                # vl_incorrect = df_test.iloc[idx]["VL"]
                seq_incorrect = "".join(df_test.iloc[idx]["VH-VL"].split('-'))

                indices_to_plot = [tsne_idx] + list(nearest_indices)
                fig2, ax2 = plt.subplots(figsize=(8, 8))

                # Plot all points with low alpha for context
                for k in range(len(colors)):
                    tmp_s = (support_labels == k)
                    ax2.scatter(X_emb_support[tmp_s, 0], X_emb_support[tmp_s, 1],
                                s=point_size, edgecolors='k', c=colors[k], marker=markers[k],
                                label=f'Train {labels[k]}', alpha=0.2)
                    tmp_qk = (query_labels == k)
                    ax2.scatter(X_emb_query[tmp_qk, 0], X_emb_query[tmp_qk, 1],
                                s=point_size, c=colors[k], marker=markers[k],
                                label=f'Test {labels[k]}', alpha=0.2)

                json_data = {
                    "incorrect": {
                        "index": int(idx),
                        "tsne_index": int(tsne_idx),
                        "source": "test",
                        "value": txt,
                        "sequence": seq_incorrect
                    },
                    "neighbors": []
                }
                xrange = xlim[1] - xlim[0]
                yrange = ylim[1] - ylim[0]
                offset_radius = 0.05 * max(xrange, yrange)  # 5% of plot width
                for j, ni in enumerate(indices_to_plot):
                    angle = 45 + j * (360 // max(2, len(indices_to_plot)))
                    dx = offset_radius * math.cos(math.radians(angle))
                    dy = offset_radius * math.sin(math.radians(angle))
                    if ni == tsne_idx:
                        color = colors[i]
                        marker = markers[i]
                        label = 'Incorrect'
                        size = point_size
                        edgecolor = 'r'
                    else:
                        label = '1-NN' if j == 1 else None
                        size = point_size
                        edgecolor = 'none'  # No edge color for neighbors
                        which_label = all_labels[0, ni]
                        color = colors[which_label]
                        marker = markers[which_label]
                        if ni < support_set:
                            n_row = df_train.iloc[ni]
                            n_txt = str(n_row[annotate_column]) if (annotate_column and annotate_column in df_train.columns) else str(ni)
                            n_source = "train"
                        else:
                            test_idx = ni - support_set
                            n_row = df_test.iloc[test_idx]
                            n_txt = str(n_row[annotate_column]) if (annotate_column and annotate_column in df_test.columns) else str(test_idx)
                            n_source = "test"
                        seq_neighbor = "".join(n_row["VH-VL"].split('-'))
                        lev_dist = distance(seq_incorrect, seq_neighbor)
                        ax2.annotate(
                            f"{lev_dist}",
                            xy=(X_emb[ni, 0], X_emb[ni, 1]),
                            xytext=(X_emb[ni, 0] + dx, X_emb[ni, 1] + dy),
                            color='black',
                            fontsize=8,
                            arrowprops=dict(arrowstyle="->", color='black', lw=1.0),
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
                        )
                        json_data["neighbors"].append({
                            "index": int(ni if ni < support_set else ni - support_set),
                            "tsne_index": int(ni),
                            "source": n_source,
                            "value": n_txt,
                            "sequence": seq_neighbor,
                            "levenshtein_distance": lev_dist
                        })
                    ax2.scatter(X_emb[ni, 0], X_emb[ni, 1], c=color, marker=marker, s=size, label=label, edgecolors=edgecolor, zorder=5)
                ax2.set_xlim(xlim)
                ax2.set_ylim(ylim)
                ax2.axes.xaxis.set_ticklabels([])
                ax2.axes.yaxis.set_ticklabels([])
                handles, labels_ = ax2.get_legend_handles_labels()
                # Remove duplicate legend entries
                seen = set()
                new_handles, new_labels = [], []
                for h, l in zip(handles, labels_):
                    if l not in seen and l != "":
                        new_handles.append(h)
                        new_labels.append(l)
                        seen.add(l)
                ax2.legend(new_handles, new_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                plt.tight_layout()
                plt.savefig(f"{save_name}_neighborhood_{idx}_labels_{labels[i]}.png", bbox_inches='tight', dpi=300)
                plt.close(fig2)
                with open(f"{save_name}_neighborhood_{idx}_labels_{labels[i]}.json", "w") as f:
                    json.dump(json_data, f, indent=2)

    return X_emb

def mkdir_if_missing(path):
    """Create directory if it doesn't exist.

    Args:
        path (str): Path to directory or file
    """
    # Get the directory path whether input is a file or directory
    directory = os.path.dirname(path) if os.path.splitext(path)[1] else path

    # Create directory if it doesn't exist and not empty
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# Helper to compute CI for binary metrics
def ci_binomial(metric_fn, y_true, y_pred):
    scores = []
    n = len(y_true)
    for i in range(n):
        idx = np.arange(n) != i
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    mean = metric_fn(y_true, y_pred)
    se = np.std(scores, ddof=1) / np.sqrt(n)
    ci = st.t.interval(0.95, n-1, loc=mean, scale=se)
    return mean, ci
# For AUC-ROC and AUC-PR, use bootstrapping
def bootstrap_auc(y_true, y_score, metric_fn, n_boot=1000):
    rng = np.random.default_rng(42)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(np.arange(n), n, replace=True)
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    mean = metric_fn(y_true, y_score)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)
    return mean, (lower, upper)

def calculate_metrics(y_true, y_probs, train=False):
    """Calculate classification metrics.
    
    Args:
        y_true (np.array): Ground truth labels in numpy if not numpy
        y_probs (np.array): Predicted probabilities
        

    Returns
    -------
        dict: Dictionary containing metrics (accuracy, precision, recall, f1, auc, fpr)
    """  # noqa: W293
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    y_probs = y_probs.cpu().numpy()
    y_pred = y_probs.argmax(1)

    # AUC-PR
    precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
    auc_pr = auc(recall, precision)

    # False Positive Rate
    # For binary classification, positive label is assumed to be 1

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fpp = fp / y_probs.shape[0] # False Positive Proportion

    # Matthews Correlation Coefficient (MCC)
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = float("nan")

    # Brier score (binary): mean squared error between predicted probability for class 1 and true label
    try:
        prob_pos = y_probs[:, 1]
        brier_score = float(np.mean((prob_pos - y_true) ** 2))
    except Exception:
        brier_score = float('nan')

    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1 Score': f1_score(y_true, y_pred),
        'Macro F1': macro_f1,
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_probs[:, 1]),
        'MCC': mcc,
        'FPR': fpr,
        'Brier': brier_score,
        'AUC-PR': auc_pr,
        'FPP': fpp,
        }

def set_basic_logging_config(
    name: str,
    path: str = "./",
    level: str = "INFO",
    format: str = "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    verbose: bool = False,
    level_file: str = None,
    level_stdout: str = None,
) -> logging.Logger:
    """Set up the default python logger.

    Parameters
    ----------
    name : str
        Name of the log file.
    path : str
        Path to a storage location for the log file.
    format : str, optional
        Log print format, by default "%(asctime)s %(levelname)-8s %(message)s".
    datefmt : str, optional
        Log datetime fomar, by default "%Y-%m-%d %H:%M:%S".
    level : str, optional
        Log level. See https://docs.python.org/3/library/logging.html, for more details, by default "DEBUG".
    verbose : bool, optional
        If True, will also print all logs to stdout, be default False.
    level_file : str, optional
        Log level for the file log. If not given, uses general level.
    level_stdout : str, optional
        Log level for the stdout log. If not given, uses general level.

    Returns
    -------
    logger: logging.Logger
        a logger with the required handles and format.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logger = logging.getLogger(name)
    handlers = []

    path = mkdir_if_missing(os.path.join(path, 'logs/'))
    if path:
        log_file = os.path.join(path, name + ".log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level.upper() if not level_file else level_file.upper())
        handlers.append(file_handler)

    if verbose:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level.upper() if not level_stdout else level_stdout.upper())
        handlers.append(stdout_handler)

    if not len(handlers):
        raise ValueError(f"Either a path (given {path}) or verbose mode (set {verbose}) must be set.")

    logging.basicConfig(
        level=level.upper(),
        format=format,
        datefmt=datefmt,
        handlers=handlers,
    )

    return logger

class DotDict(dict):
    """Dictionary with dot notation access to attributes."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        data = dict(*args, **kwargs)
        for k, v in data.items():
            self[k] = v  # Use __setitem__

    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")
    def __setattr__(self, attr, value):
        self[attr] = value
    def __setitem__(self, key, value):
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
        elif isinstance(value, list):
            value = [DotDict(i) if isinstance(i, dict) and not isinstance(i, DotDict) else i for i in value]
        super().__setitem__(key, value)
    def __delattr__(self, attr):
        try:
            del self[attr]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{attr}'")

def dict_to_namespace(d):
    """Recursively convert a dict to DotDict for dot access."""
    if isinstance(d, dict):
        return DotDict({k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def load_yaml_config(file_path: str):
    """Load a yaml file and return as SimpleNamespace for dot access."""
    with open(file_path, "r") as fp:
        data = yaml.safe_load(fp)
    return data


def plot_pr_curves(probs_dict, y_true, save_path, title="PR Curves"):
    """Plot overlaid precision-recall curves for multiple models.

    Args:
        probs_dict: dict mapping model name -> predicted probabilities (N x 2 array or tensor)
        y_true: ground truth labels
        save_path: file path to save the figure
        title: plot title
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))

    pos_rate = float(np.mean(y_true))
    ax.axhline(y=pos_rate, color='gray', linestyle='--', linewidth=1.0,
               label=f'Random (prevalence={pos_rate:.2f})')

    for model_name, y_probs in probs_dict.items():
        if hasattr(y_probs, 'cpu'):
            y_probs = y_probs.cpu().numpy()
        prob_pos = y_probs[:, 1]
        precision, recall, _ = precision_recall_curve(y_true, prob_pos)
        auc_pr = auc(recall, precision)
        ax.plot(recall, precision, linewidth=1.5, label=f'{model_name} (AUC-PR={auc_pr:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.6)
    plt.tight_layout()
    mkdir_if_missing(save_path)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    logging.info(f"PR curves saved to {save_path}")


if __name__ == "__main__":

    # Test case 1: Perfect prediction
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_probs = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.7, 0.3]
    ])
    print("Test case 1: Perfect prediction")
    print(calculate_metrics(y_true, torch.tensor(y_probs)))

    # Test case 2: Random prediction
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=100)
    y_probs = np.random.rand(100, 2)
    y_probs = y_probs / y_probs.sum(axis=1, keepdims=True)
    print("\nTest case 2: Random prediction")
    print(calculate_metrics(y_true, torch.tensor(y_probs)))

    # Test case 3: All one class
    y_true = np.zeros(10, dtype=int)
    y_probs = np.ones((10, 2)) * [0.95, 0.05]
    print("\nTest case 3: All one class")
    print(calculate_metrics(y_true, torch.tensor(y_probs)))

    # Test case 4: All wrong prediction
    y_true = np.array([0, 1, 0, 1])
    y_probs = np.array([
        [0.1, 0.9],
        [0.9, 0.1],
        [0.2, 0.8],
        [0.8, 0.2]
    ])
    print("\nTest case 4: All wrong prediction")
    print(calculate_metrics(y_true, torch.tensor(y_probs)))