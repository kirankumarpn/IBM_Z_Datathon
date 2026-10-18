"""
program_5_isolationforest.py - Program 5: Isolation Forest Anomaly Detection
Trains an unsupervised Isolation Forest strictly on normal training flows.
Tunes anomaly score threshold on validation data only, evaluates on test data,
plots score distributions and confusion matrix, and saves artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common import (
    calculate_comprehensive_metrics, plot_confusion_matrix, plot_roc_and_pr_curves,
    save_json_artifact, check_ibm_z_environment, OUTPUTS_DIR
)

OUT_DIR = os.path.join(OUTPUTS_DIR, "isolationforest")
os.makedirs(OUT_DIR, exist_ok=True)
DATA_ANALYSIS_DIR = os.path.join(OUTPUTS_DIR, "dataset_analysis")


def plot_anomaly_score_distribution(scores, y_true, threshold, save_path):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    sns.kdeplot(scores[y_true == 0], ax=ax, label="Normal Flows (0)", color="#10b981", fill=True, alpha=0.3, lw=2)
    sns.kdeplot(scores[y_true == 1], ax=ax, label="Attack Flows (1)", color="#ef4444", fill=True, alpha=0.3, lw=2)

    ax.axvline(threshold, color="#f59e0b", lw=2, linestyle="--", label=f"Tuned Threshold ({threshold:.4f})")
    ax.set_title("Isolation Forest Anomaly Score Distribution (Test Partition)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Anomaly Score (-Decision Function, Higher = More Malicious)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def tune_isolation_threshold(y_val, val_scores, save_path):
    percentiles = np.linspace(50, 99.9, 100)
    f1_scores = []
    threshold_vals = []

    for p in percentiles:
        t = np.percentile(val_scores[y_val == 0], p)
        preds = (val_scores >= t).astype(int)
        f1_scores.append(f1_score(y_val, preds, zero_division=0))
        threshold_vals.append(t)

    best_idx = np.argmax(f1_scores)
    best_p = percentiles[best_idx]
    best_f1 = f1_scores[best_idx]
    best_t = threshold_vals[best_idx]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(percentiles, f1_scores, color="#06b6d4", lw=2.5, label="Validation F1-Score")
    ax.scatter([best_p], [best_f1], color="#ef4444", s=80, zorder=5, label=f"Optimal: {best_p:.1f}th pct (F1={best_f1:.4f})")
    ax.set_xlabel("Normal Score Percentile Cutoff", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("Isolation Forest Anomaly Threshold Optimization", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")
    return best_t, best_f1


def plot_metrics_summary(metrics, model_name, save_path):
    labels = ["Accuracy", "Precision", "Attack Recall", "F1-Score", "Balanced Acc", "ROC-AUC"]
    values = [
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["balanced_accuracy"],
        metrics["roc_auc"]
    ]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(labels, values, color=["#06b6d4", "#0891b2", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899"], edgecolor="black")
    ax.set_ylim(0.0, 1.1)
    ax.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax.set_title(f"{model_name} — Benchmark Performance Metrics", fontsize=13, fontweight="bold", pad=12)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.xticks(fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def main():
    print("=" * 75)
    print("PROGRAM 5: ISOLATION FOREST ANOMALY DETECTION")
    print("IBM LinuxONE Community Cloud • Botnet Detection Pipeline")
    print("=" * 75)

    env_info = check_ibm_z_environment()
    print(f"[System] Platform: {env_info['runtime_notes']}")

    # 1. Load prepared data
    train_path = os.path.join(DATA_ANALYSIS_DIR, "train_data.csv")
    val_path = os.path.join(DATA_ANALYSIS_DIR, "val_data.csv")
    test_path = os.path.join(DATA_ANALYSIS_DIR, "test_data.csv")

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        raise FileNotFoundError("Preprocessed partition files missing! Run Program 1 first.")

    print("[DataLoader] Reading preprocessed dataset splits...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)

    feature_cols = [c for c in df_train.columns if c != "Label"]
    print(f"[Features] {len(feature_cols)} flow features loaded.")

    # Isolation Forest is fitted strictly on normal training flows
    normal_train = df_train[df_train["Label"] == 0]
    X_train_normal = normal_train[feature_cols].values
    print(f"[Training Data] Fitting Isolation Forest on {len(X_train_normal):,} normal flows only.")

    X_val = df_val[feature_cols].values
    y_val = df_val["Label"].values.astype(int)

    X_test = df_test[feature_cols].values
    y_test = df_test["Label"].values.astype(int)

    # 2. Fit StandardScaler on normal training data only
    print("[Preprocessing] Fitting StandardScaler on normal training flows only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 3. Fit Isolation Forest
    print("[Training] Fitting Isolation Forest (n_estimators=100, max_samples=512, seed=42)...")
    model = IsolationForest(
        n_estimators=100,
        max_samples=512,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled)
    print("[Training] Isolation Forest training completed.")

    # 4. Compute anomaly scores (negative decision function: higher = more anomalous)
    print("[Inference] Computing anomaly scores on validation set...")
    val_scores = -model.decision_function(X_val_scaled)

    # 5. Tune threshold on validation set only
    print("[Threshold] Tuning anomaly threshold on validation set...")
    best_threshold, val_f1 = tune_isolation_threshold(
        y_val, val_scores, os.path.join(OUT_DIR, "threshold_tuning_curve.png")
    )
    print(f"[Threshold] Selected Anomaly Threshold: {best_threshold:.6f} (Validation F1: {val_f1:.4f})")

    # 6. Evaluate on held-out test partition
    print("[Evaluation] Evaluating Isolation Forest on held-out test partition...")
    test_scores = -model.decision_function(X_test_scaled)
    test_preds = (test_scores >= best_threshold).astype(int)

    # MinMax normalize score to [0, 1] range for ROC-AUC
    min_s = np.percentile(val_scores, 1)
    max_s = np.percentile(val_scores, 99)
    test_probs = np.clip((test_scores - min_s) / (max_s - min_s + 1e-8), 0.0, 1.0)
    val_probs = np.clip((val_scores - min_s) / (max_s - min_s + 1e-8), 0.0, 1.0)

    metrics = calculate_comprehensive_metrics(y_test, test_preds, test_probs)
    metrics["anomaly_threshold"] = float(best_threshold)
    metrics["model_name"] = "Isolation Forest Anomaly Detection"
    metrics["validation_f1"] = float(val_f1)

    print("\n--- ISOLATION FOREST TEST BENCHMARK RESULTS ---")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  Attack Recall      : {metrics['recall']:.4f}")
    print(f"  F1-Score           : {metrics['f1_score']:.4f}")
    print(f"  Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix   : TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, "
          f"FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")

    # 7. Save model and predictions
    print("[Storage] Saving Isolation Forest model, scaler, and predictions...")
    joblib.dump(model, os.path.join(OUT_DIR, "isolationforest_model.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

    np.save(os.path.join(OUT_DIR, "test_scores.npy"), test_scores)
    np.save(os.path.join(OUT_DIR, "val_scores.npy"), val_scores)
    np.save(os.path.join(OUT_DIR, "test_probabilities.npy"), test_probs)
    np.save(os.path.join(OUT_DIR, "val_probabilities.npy"), val_probs)

    pred_df = pd.DataFrame({
        "True_Label": y_test,
        "Anomaly_Score": test_scores,
        "Predicted_Label": test_preds
    })
    pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    save_json_artifact(metrics, os.path.join(OUT_DIR, "isolationforest_metrics.json"))

    # 8. Generate Visualizations
    print("[Visualization] Generating Isolation Forest evaluation charts...")
    plot_anomaly_score_distribution(test_scores, y_test, best_threshold, os.path.join(OUT_DIR, "anomaly_score_dist.png"))
    plot_confusion_matrix(metrics["confusion_matrix"], "Isolation Forest — Test Confusion Matrix", os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_roc_and_pr_curves(y_test, test_probs, "Isolation Forest", os.path.join(OUT_DIR, "roc_pr_curves.png"))
    plot_metrics_summary(metrics, "Isolation Forest", os.path.join(OUT_DIR, "metrics_summary.png"))

    print("\n" + "=" * 75)
    print("PROGRAM 5 COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {OUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
