"""
src/common.py - Shared Reusable Module for IBM Z Datathon Cyber Threat Detection
Provides file discovery, data loading, preprocessing contracts, metric calculation,
plotting utilities, and artifact persistence.
"""

import os
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)

# Project Directory Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Setup Matplotlib Cyber Theme
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#3b82f6"
plt.rcParams["axes.linewidth"] = 1.0


def check_ibm_z_environment():
    """
    Inspects system platform and architecture to determine IBM Z / s390x compatibility.
    """
    sys_arch = platform.machine()
    sys_os = platform.system()
    py_ver = platform.python_version()

    is_s390x = ("s390" in sys_arch.lower()) or ("z" in sys_arch.lower() and sys_os == "Linux")

    info = {
        "machine": sys_arch,
        "system": sys_os,
        "python_version": py_ver,
        "is_ibm_z": is_s390x,
        "runtime_notes": "Native s390x Mainframe Architecture" if is_s390x else f"Workstation/Cloud Simulation ({sys_arch})"
    }
    return info


def resolve_dataset_files():
    """
    Locates the normal and attack dataset CSV files, handling exact names and aliases.
    Handles 'CTIJI ormaI_Traffic.csv' / 'CTU13_Normal_Traffic.csv' and 'CTU13_Attack_Traffic.csv'.
    """
    possible_normal = [
        os.path.join(DATA_DIR, "CTIJI ormaI_Traffic.csv"),
        os.path.join(DATA_DIR, "CTU13_Normal_Traffic.csv"),
        os.path.join(BASE_DIR, "CTIJI ormaI_Traffic.csv"),
        os.path.join(BASE_DIR, "CTU13_Normal_Traffic.csv")
    ]

    possible_attack = [
        os.path.join(DATA_DIR, "CTU13_Attack_Traffic.csv"),
        os.path.join(BASE_DIR, "CTU13_Attack_Traffic.csv")
    ]

    normal_path = None
    for path in possible_normal:
        if os.path.exists(path):
            normal_path = path
            break

    attack_path = None
    for path in possible_attack:
        if os.path.exists(path):
            attack_path = path
            break

    if not normal_path or not attack_path:
        # Scan directory to display existing CSV files
        available_csvs = []
        for root in [BASE_DIR, DATA_DIR]:
            if os.path.exists(root):
                for f in os.listdir(root):
                    if f.endswith(".csv"):
                        available_csvs.append(os.path.join(root, f))
        raise FileNotFoundError(
            f"Dataset files missing!\n"
            f"Looking for Normal: {possible_normal}\n"
            f"Looking for Attack: {possible_attack}\n"
            f"Available CSV files in workspace: {available_csvs}"
        )

    return normal_path, attack_path


def load_raw_datasets(normal_path: str, attack_path: str):
    """
    Loads raw normal and attack CSV datasets and assigns explicit ground truth labels:
    Normal = 0, Attack = 1.
    """
    print(f"[DataLoader] Loading Normal Traffic from: {os.path.basename(normal_path)}")
    df_normal = pd.read_csv(normal_path)
    df_normal["Label"] = 0  # Force ground-truth class 0

    print(f"[DataLoader] Loading Attack Traffic from: {os.path.basename(attack_path)}")
    df_attack = pd.read_csv(attack_path)
    df_attack["Label"] = 1  # Force ground-truth class 1

    return df_normal, df_attack


def clean_and_prepare_features(df: pd.DataFrame, drop_cols=None):
    """
    Cleans infinite and NaN values and drops non-predictive identifiers.
    """
    if drop_cols is None:
        drop_cols = ["Unnamed: 0"]

    df = df.copy()
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Replace Inf with NaN and then impute
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c != "Label"]

    # Check inf values
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    # Impute any NaN with median
    nan_counts = df[feature_cols].isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0].index.tolist()
    if len(cols_with_nan) > 0:
        print(f"[Cleaning] Imputing NaNs with feature medians in columns: {cols_with_nan}")
        for col in cols_with_nan:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0.0)

    return df, feature_cols


def calculate_comprehensive_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates standardized scientific evaluation metrics:
    Accuracy, Precision, Recall (Attack Recall), F1-Score, Balanced Accuracy,
    ROC-AUC, and Confusion Matrix components (TN, FP, FN, TP).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # ROC AUC
    roc_auc = None
    if y_prob is not None:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.5
    else:
        try:
            roc_auc = float(roc_auc_score(y_true, y_pred))
        except Exception:
            roc_auc = 0.5

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn = int(cm[0, 0]) if cm.shape == (2, 2) else 0
    fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
    fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
    tp = int(cm[1, 1]) if cm.shape == (2, 2) else 0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "balanced_accuracy": bal_acc,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    }


def find_best_threshold_by_f1(y_val, y_val_scores):
    """
    Selects the optimal decision threshold based on validation data only,
    maximizing the F1-score on the attack class.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_thresh = 0.5
    best_f1 = -1.0

    for t in thresholds:
        preds = (y_val_scores >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(t)

    return best_thresh, float(best_f1)


def plot_confusion_matrix(cm_dict, title, save_path):
    """
    Plots a publication-quality confusion matrix heatmap.
    """
    cm = np.array([
        [cm_dict["tn"], cm_dict["fp"]],
        [cm_dict["fn"], cm_dict["tp"]]
    ])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Pred: Normal", "Pred: Attack"],
        yticklabels=["True: Normal", "True: Attack"],
        ax=ax, annot_kws={"size": 14, "weight": "bold"}
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_roc_and_pr_curves(y_true, y_prob, model_name, save_path):
    """
    Plots dual ROC Curve and Precision-Recall Curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # ROC Curve
    ax1.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {auc_val:.4f})")
    ax1.plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax1.set_title(f"{model_name} — ROC Curve", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right")

    # Precision-Recall Curve
    ax2.plot(rec, prec, color="#10b981", lw=2, label=f"PR (AP = {avg_prec:.4f})")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision", fontsize=11, fontweight="bold")
    ax2.set_title(f"{model_name} — Precision-Recall Curve", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def save_json_artifact(data: dict, filepath: str):
    """Saves dictionary to clean indented JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[Artifact] Saved: {filepath}")


def load_json_artifact(filepath: str) -> dict:
    """Loads dictionary from JSON artifact file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
