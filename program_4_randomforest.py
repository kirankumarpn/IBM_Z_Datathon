"""
program_4_randomforest.py - Program 4: Random Forest Classifier
Trains an ensemble Random Forest classifier on the shared train split,
tunes decision threshold on validation data only, evaluates on held-out test data,
plots confusion matrix, ROC/PR curves, and feature importance, and saves artifacts.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common import (
    calculate_comprehensive_metrics, find_best_threshold_by_f1,
    plot_confusion_matrix, plot_roc_and_pr_curves,
    save_json_artifact, check_ibm_z_environment, OUTPUTS_DIR
)

OUT_DIR = os.path.join(OUTPUTS_DIR, "randomforest")
os.makedirs(OUT_DIR, exist_ok=True)
DATA_ANALYSIS_DIR = os.path.join(OUTPUTS_DIR, "dataset_analysis")


def plot_feature_importance(model, feature_names, save_path, top_n=15):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_names = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    bars = ax.barh(range(top_n), top_scores[::-1], color="#10b981", edgecolor="black", height=0.65)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=10, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=11, fontweight="bold")
    ax.set_title(f"Random Forest Top {top_n} Most Discriminative Features", fontsize=12, fontweight="bold", pad=12)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.4f}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=9)

    ax.set_xlim(0, max(top_scores) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


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
    bars = ax.bar(labels, values, color=["#10b981", "#059669", "#3b82f6", "#8b5cf6", "#f59e0b", "#ec4899"], edgecolor="black")
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
    print("PROGRAM 4: RANDOM FOREST CLASSIFIER TRAINING & EVALUATION")
    print("IBM LinuxONE Community Cloud • Botnet Detection Pipeline")
    print("=" * 75)

    env_info = check_ibm_z_environment()
    print(f"[System] Platform: {env_info['runtime_notes']}")

    # 1. Load prepared splits
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

    X_train = df_train[feature_cols].values
    y_train = df_train["Label"].values.astype(int)

    X_val = df_val[feature_cols].values
    y_val = df_val["Label"].values.astype(int)

    X_test = df_test[feature_cols].values
    y_test = df_test["Label"].values.astype(int)

    # 2. Fit StandardScaler on training data only
    print("[Preprocessing] Fitting StandardScaler on training partition only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 3. Train Random Forest Classifier
    print("[Training] Fitting Random Forest Classifier (trees=100, max_depth=16, seed=42)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    print("[Training] Random Forest training completed.")

    # 4. Tune decision threshold on validation set only
    print("[Threshold] Tuning optimal classification threshold on validation partition...")
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    best_threshold, val_f1 = find_best_threshold_by_f1(y_val, val_probs)
    print(f"[Threshold] Optimal validation threshold: {best_threshold:.4f} (Validation F1: {val_f1:.4f})")

    # 5. Evaluate on held-out test partition
    print("[Evaluation] Evaluating tuned model on held-out test partition...")
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    metrics = calculate_comprehensive_metrics(y_test, test_preds, test_probs)
    metrics["tuned_threshold"] = float(best_threshold)
    metrics["model_name"] = "Random Forest Classifier"
    metrics["validation_f1"] = float(val_f1)

    print("\n--- RANDOM FOREST TEST BENCHMARK RESULTS ---")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  Attack Recall      : {metrics['recall']:.4f}")
    print(f"  F1-Score           : {metrics['f1_score']:.4f}")
    print(f"  Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix   : TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, "
          f"FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")

    # 6. Save model and predictions
    print("[Storage] Saving trained model, scaler, and predictions...")
    joblib.dump(model, os.path.join(OUT_DIR, "randomforest_model.joblib"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.joblib"))

    np.save(os.path.join(OUT_DIR, "test_probabilities.npy"), test_probs)
    np.save(os.path.join(OUT_DIR, "val_probabilities.npy"), val_probs)

    pred_df = pd.DataFrame({
        "True_Label": y_test,
        "Attack_Probability": test_probs,
        "Predicted_Label": test_preds
    })
    pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    save_json_artifact(metrics, os.path.join(OUT_DIR, "randomforest_metrics.json"))

    # 7. Generate Visualizations
    print("[Visualization] Generating Random Forest evaluation charts...")
    plot_confusion_matrix(metrics["confusion_matrix"], "Random Forest — Test Confusion Matrix", os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_roc_and_pr_curves(y_test, test_probs, "Random Forest", os.path.join(OUT_DIR, "roc_pr_curves.png"))
    plot_feature_importance(model, feature_cols, os.path.join(OUT_DIR, "feature_importance.png"))
    plot_metrics_summary(metrics, "Random Forest", os.path.join(OUT_DIR, "metrics_summary.png"))

    print("\n" + "=" * 75)
    print("PROGRAM 4 COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {OUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
