"""
program_6_hybrid_model.py - Program 6: Hybrid XGBoost + Autoencoder
Combines XGBoost supervised attack probabilities with normalized Autoencoder
anomaly scores. Tunes the combination weight and threshold strictly on validation data,
evaluates on held-out test data, compares against individual models, and saves artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

import torch

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common import (
    calculate_comprehensive_metrics, plot_confusion_matrix, plot_roc_and_pr_curves,
    save_json_artifact, load_json_artifact, check_ibm_z_environment, OUTPUTS_DIR
)
from program_3_autoencoder import NetworkFlowAutoencoder, compute_reconstruction_errors

OUT_DIR = os.path.join(OUTPUTS_DIR, "hybrid")
os.makedirs(OUT_DIR, exist_ok=True)
DATA_ANALYSIS_DIR = os.path.join(OUTPUTS_DIR, "dataset_analysis")
XGB_DIR = os.path.join(OUTPUTS_DIR, "xgboost")
AE_DIR = os.path.join(OUTPUTS_DIR, "autoencoder")


def tune_hybrid_parameters(y_val, xgb_val_probs, ae_val_errors):
    """
    Grid searches combination weight (alpha) and decision threshold
    strictly on validation set to maximize F1-score.
    Score = alpha * P_xgb + (1 - alpha) * Norm_AE_error
    """
    # Normalize AE error to [0, 1] using validation 1st and 99th percentiles
    ae_min = np.percentile(ae_val_errors, 1)
    ae_max = np.percentile(ae_val_errors, 99)
    ae_val_norm = np.clip((ae_val_errors - ae_min) / (ae_max - ae_min + 1e-8), 0.0, 1.0)

    alphas = np.linspace(0.0, 1.0, 21)
    thresholds = np.linspace(0.1, 0.9, 81)

    best_alpha = 0.5
    best_thresh = 0.5
    best_f1 = -1.0

    tuning_records = []

    for a in alphas:
        combined_val = a * xgb_val_probs + (1.0 - a) * ae_val_norm
        for t in thresholds:
            preds = (combined_val >= t).astype(int)
            score = f1_score(y_val, preds, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_alpha = a
                best_thresh = t

        tuning_records.append({"alpha": float(a), "val_f1": float(best_f1)})

    return best_alpha, best_thresh, best_f1, (ae_min, ae_max), tuning_records


def plot_weight_tuning(tuning_records, best_alpha, best_f1, save_path):
    alphas = [r["alpha"] for r in tuning_records]
    f1s = [r["val_f1"] for r in tuning_records]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(alphas, f1s, color="#2563eb", lw=2.5, marker="o", markersize=5, label="Validation F1 across Alphas")
    ax.scatter([best_alpha], [best_f1], color="#ef4444", s=100, zorder=5,
               label=f"Selected Alpha: {best_alpha:.2f} (F1: {best_f1:.4f})")
    ax.set_xlabel("XGBoost Weight (Alpha) vs. Autoencoder (1 - Alpha)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("Hybrid Weight Grid Search Optimization (Validation Partition)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_model_comparison_trio(xgb_m, ae_m, hybrid_m, save_path):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics))
    width = 0.25

    xgb_vals = [xgb_m["accuracy"], xgb_m["precision"], xgb_m["recall"], xgb_m["f1_score"], xgb_m["roc_auc"]]
    ae_vals = [ae_m["accuracy"], ae_m["precision"], ae_m["recall"], ae_m["f1_score"], ae_m["roc_auc"]]
    hyb_vals = [hybrid_m["accuracy"], hybrid_m["precision"], hybrid_m["recall"], hybrid_m["f1_score"], hybrid_m["roc_auc"]]

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    ax.bar(x - width, xgb_vals, width, label="XGBoost (Supervised)", color="#3b82f6", edgecolor="black")
    ax.bar(x, ae_vals, width, label="Autoencoder (Unsupervised)", color="#8b5cf6", edgecolor="black")
    ax.bar(x + width, hyb_vals, width, label="Hybrid Pipeline", color="#10b981", edgecolor="black")

    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title("Performance Comparison: XGBoost vs. Autoencoder vs. Hybrid Model", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_ylim(0.7, 1.05)
    ax.legend(loc="lower left", fontsize=10)

    # Annotate hybrid bars
    for i, v in enumerate(hyb_vals):
        ax.annotate(f"{v:.4f}",
                    xy=(x[i] + width, v),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def main():
    print("=" * 75)
    print("PROGRAM 6: HYBRID XGBOOST + AUTOENCODER MODEL INTEGRATION")
    print("IBM LinuxONE Community Cloud • Botnet Detection Pipeline")
    print("=" * 75)

    env_info = check_ibm_z_environment()
    print(f"[System] Platform: {env_info['runtime_notes']}")

    # 1. Verify antecedent model artifacts
    xgb_metrics_path = os.path.join(XGB_DIR, "xgboost_metrics.json")
    ae_metrics_path = os.path.join(AE_DIR, "autoencoder_metrics.json")

    if not os.path.exists(xgb_metrics_path) or not os.path.exists(ae_metrics_path):
        raise FileNotFoundError(
            "Antecedent model artifacts missing! Please run Program 2 and Program 3 first."
        )

    xgb_metrics = load_json_artifact(xgb_metrics_path)
    ae_metrics = load_json_artifact(ae_metrics_path)

    # 2. Load validation and test predictions from both models
    print("[DataLoader] Ingesting validation and test prediction arrays from Programs 2 & 3...")
    xgb_val_probs = np.load(os.path.join(XGB_DIR, "val_probabilities.npy"))
    xgb_test_probs = np.load(os.path.join(XGB_DIR, "test_probabilities.npy"))

    ae_val_errors = np.load(os.path.join(AE_DIR, "val_errors.npy"))
    ae_test_errors = np.load(os.path.join(AE_DIR, "test_errors.npy"))

    # Load ground truth labels from test and val CSVs
    df_val = pd.read_csv(os.path.join(DATA_ANALYSIS_DIR, "val_data.csv"))
    df_test = pd.read_csv(os.path.join(DATA_ANALYSIS_DIR, "test_data.csv"))
    y_val = df_val["Label"].values.astype(int)
    y_test = df_test["Label"].values.astype(int)

    # 3. Grid-search optimal hybrid parameters on VALIDATION DATA ONLY
    print("[Optimization] Tuning combination weight (Alpha) and threshold on validation set...")
    best_alpha, best_thresh, val_f1, (ae_min, ae_max), tuning_records = tune_hybrid_parameters(
        y_val, xgb_val_probs, ae_val_errors
    )
    print(f"[Optimization] Optimal Parameters -> Alpha: {best_alpha:.2f} (XGB) + {1.0-best_alpha:.2f} (AE), "
          f"Threshold: {best_thresh:.4f} (Validation F1: {val_f1:.4f})")

    # 4. Evaluate on held-out test partition
    print("[Evaluation] Computing combined hybrid scores on held-out test partition...")
    ae_test_norm = np.clip((ae_test_errors - ae_min) / (ae_max - ae_min + 1e-8), 0.0, 1.0)
    hybrid_test_probs = best_alpha * xgb_test_probs + (1.0 - best_alpha) * ae_test_norm
    hybrid_test_preds = (hybrid_test_probs >= best_thresh).astype(int)

    metrics = calculate_comprehensive_metrics(y_test, hybrid_test_preds, hybrid_test_probs)
    metrics["model_name"] = "Hybrid XGBoost + Autoencoder"
    metrics["combination_weight_xgb"] = float(best_alpha)
    metrics["combination_weight_ae"] = float(1.0 - best_alpha)
    metrics["tuned_threshold"] = float(best_thresh)
    metrics["validation_f1"] = float(val_f1)

    print("\n--- HYBRID MODEL TEST BENCHMARK RESULTS ---")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  Attack Recall      : {metrics['recall']:.4f}")
    print(f"  F1-Score           : {metrics['f1_score']:.4f}")
    print(f"  Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix   : TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, "
          f"FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")

    # 5. Scientific Comparison Analysis
    delta_f1 = metrics["f1_score"] - xgb_metrics["f1_score"]
    delta_rec = metrics["recall"] - xgb_metrics["recall"]
    delta_fp = metrics["confusion_matrix"]["fp"] - xgb_metrics["confusion_matrix"]["fp"]

    print("\n--- SCIENTIFIC INTEGRITY COMPARISON (vs. Standalone XGBoost) ---")
    print(f"  Delta F1-Score     : {delta_f1:+.4f}")
    print(f"  Delta Attack Recall: {delta_rec:+.4f}")
    print(f"  Delta False Pos    : {delta_fp:+d} flows")
    print("  Conclusion: The hybrid pipeline incorporates Autoencoder unsupervised anomaly signal,")
    print("              balancing deep representation discovery with gradient-boosted decision bounds.")

    # 6. Save artifacts
    print("[Storage] Saving hybrid model artifacts and configuration...")
    hybrid_config = {
        "alpha_xgboost": float(best_alpha),
        "alpha_autoencoder": float(1.0 - best_alpha),
        "tuned_threshold": float(best_thresh),
        "ae_normalization": {"min": float(ae_min), "max": float(ae_max)},
        "validation_f1": float(val_f1)
    }
    save_json_artifact(hybrid_config, os.path.join(OUT_DIR, "hybrid_config.json"))

    np.save(os.path.join(OUT_DIR, "test_probabilities.npy"), hybrid_test_probs)

    pred_df = pd.DataFrame({
        "True_Label": y_test,
        "Hybrid_Probability": hybrid_test_probs,
        "Predicted_Label": hybrid_test_preds
    })
    pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    save_json_artifact(metrics, os.path.join(OUT_DIR, "hybrid_metrics.json"))

    # 7. Generate Visualizations
    print("[Visualization] Generating Hybrid evaluation and comparison charts...")
    plot_weight_tuning(tuning_records, best_alpha, val_f1, os.path.join(OUT_DIR, "hybrid_weight_tuning.png"))
    plot_confusion_matrix(metrics["confusion_matrix"], "Hybrid Pipeline — Test Confusion Matrix", os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_roc_and_pr_curves(y_test, hybrid_test_probs, "Hybrid Model", os.path.join(OUT_DIR, "roc_pr_curves.png"))
    plot_model_comparison_trio(xgb_metrics, ae_metrics, metrics, os.path.join(OUT_DIR, "hybrid_comparison.png"))

    print("\n" + "=" * 75)
    print("PROGRAM 6 COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {OUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
