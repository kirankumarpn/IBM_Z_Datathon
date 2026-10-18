"""
program_7_comparison.py - Program 7: Final Model Comparison & Results Dashboard
Ingests JSON benchmark metrics from Programs 2–6, prints clean comparative tables,
generates grouped metric charts, ranking plots, confusion matrix grids, and
saves standardized comparison CSV and JSON artifacts.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common import (
    load_json_artifact, save_json_artifact, check_ibm_z_environment, OUTPUTS_DIR, BASE_DIR
)

OUT_DIR = os.path.join(OUTPUTS_DIR, "comparison")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_grouped_metrics_bar(df_comp, save_path):
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score", "Balanced_Acc", "ROC_AUC"]
    models = df_comp["Model"].tolist()

    x = np.arange(len(metrics))
    total_models = len(models)
    width = 0.8 / total_models
    colors = ["#3b82f6", "#8b5cf6", "#10b981", "#06b6d4", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(13, 6), dpi=150)

    for i, model in enumerate(models):
        row = df_comp[df_comp["Model"] == model].iloc[0]
        vals = [row[m] for m in metrics]
        offset = (i - total_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model, color=colors[i % len(colors)], edgecolor="black", alpha=0.9)

    ax.set_ylabel("Metric Score", fontsize=11, fontweight="bold")
    ax.set_title("Comprehensive Multi-Model Performance Benchmark Comparison", fontsize=14, fontweight="bold", pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score", "Balanced Acc", "ROC-AUC"], fontsize=11, fontweight="bold")
    ax.set_ylim(0.65, 1.05)
    ax.legend(loc="lower left", fontsize=10, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_accuracy_ranking(df_comp, save_path):
    df_sorted = df_comp.sort_values(by="Accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    bars = ax.barh(df_sorted["Model"], df_sorted["Accuracy"], color="#3b82f6", edgecolor="black", height=0.55)
    ax.set_xlabel("Overall Test Accuracy", fontsize=11, fontweight="bold")
    ax.set_title("Test Accuracy Leaderboard Ranking", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0.70, 1.02)

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.4%}",
                    xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_precision_recall_tradeoff(df_comp, save_path):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = ["#3b82f6", "#8b5cf6", "#10b981", "#06b6d4", "#f59e0b"]

    for i, row in df_comp.iterrows():
        ax.scatter(row["Recall"], row["Precision"], s=180, color=colors[i % len(colors)],
                   edgecolor="black", lw=1.5, label=row["Model"], zorder=4)
        ax.annotate(row["Model"], (row["Recall"], row["Precision"]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=9, fontweight="bold")

    ax.set_xlabel("Attack Recall (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax.set_title("Precision vs. Recall Trade-Off Analysis Across Architectures", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0.75, 1.02)
    ax.set_ylim(0.65, 1.02)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_confusion_matrices_grid(models_data, save_path):
    fig, axes = plt.subplots(1, len(models_data), figsize=(4 * len(models_data), 3.8), dpi=150)
    if len(models_data) == 1:
        axes = [axes]

    for i, (name, data) in enumerate(models_data.items()):
        ax = axes[i]
        cm_dict = data["confusion_matrix"]
        cm = np.array([
            [cm_dict["tn"], cm_dict["fp"]],
            [cm_dict["fn"], cm_dict["tp"]]
        ])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["Pred: 0", "Pred: 1"],
                    yticklabels=["True: 0", "True: 1"],
                    ax=ax, annot_kws={"size": 11, "weight": "bold"})
        ax.set_title(f"{name}", fontsize=11, fontweight="bold")

    plt.suptitle("Confusion Matrices Across All Tested Architectures", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def sync_with_pitch_deck(best_model_name, best_metrics):
    """
    Automatically propagates measured datathon benchmarks to presentation_state.json
    so the interactive pitch deck and PPTX/PDF exports reflect real experimental findings!
    """
    pres_path = os.path.join(BASE_DIR, "presentation_state.json")
    if os.path.exists(pres_path):
        try:
            with open(pres_path, "r", encoding="utf-8") as f:
                pres = json.load(f)

            # Slide 8 is Evaluation & Results
            for slide in pres.get("slides", []):
                if slide.get("layout") == "evaluation_results" or slide.get("number") == 8:
                    content = slide.get("content", {})
                    content["metrics"] = [
                        {"label": "Accuracy", "value": f"{best_metrics['accuracy']:.2%}", "desc": f"Overall test accuracy ({best_model_name})"},
                        {"label": "Precision", "value": f"{best_metrics['precision']:.2%}", "desc": "Botnet flow precision"},
                        {"label": "Recall", "value": f"{best_metrics['recall']:.2%}", "desc": "Attack detection sensitivity"},
                        {"label": "F1-Score", "value": f"{best_metrics['f1_score']:.2%}", "desc": "Harmonic balance"}
                    ]
                    cm = best_metrics["confusion_matrix"]
                    content["confusionMatrix"] = {
                        "tn": f"{cm['tn']:,} Normal",
                        "fp": f"{cm['fp']:,} False Pos",
                        "fn": f"{cm['fn']:,} Missed",
                        "tp": f"{cm['tp']:,} Detected"
                    }
                    content["integrityNote"] = (
                        f"Verified Empirical Benchmark on CTU-13 Test Partition ({len(cm)} flows): "
                        f"Achieved using {best_model_name} on IBM LinuxONE compute environment."
                    )
                    break

            with open(pres_path, "w", encoding="utf-8") as f:
                json.dump(pres, f, indent=2)
            print(f"[Integration] Pitch Deck presentation state updated with {best_model_name} measured results!")
        except Exception as e:
            print(f"[Integration] Note: Pitch deck sync skipped: {e}")


def main():
    print("=" * 75)
    print("PROGRAM 7: FINAL MODEL COMPARISON & RESULTS DASHBOARD")
    print("IBM LinuxONE Community Cloud • Botnet Detection Pipeline")
    print("=" * 75)

    env_info = check_ibm_z_environment()
    print(f"[System] Platform: {env_info['runtime_notes']}")

    # 1. Load all model metrics JSON files
    metric_files = {
        "XGBoost Classifier": os.path.join(OUTPUTS_DIR, "xgboost", "xgboost_metrics.json"),
        "Neural Autoencoder": os.path.join(OUTPUTS_DIR, "autoencoder", "autoencoder_metrics.json"),
        "Random Forest": os.path.join(OUTPUTS_DIR, "randomforest", "randomforest_metrics.json"),
        "Isolation Forest": os.path.join(OUTPUTS_DIR, "isolationforest", "isolationforest_metrics.json"),
        "Hybrid Pipeline": os.path.join(OUTPUTS_DIR, "hybrid", "hybrid_metrics.json")
    }

    models_data = {}
    table_rows = []

    for name, path in metric_files.items():
        if not os.path.exists(path):
            print(f"[Warning] Metrics file missing for {name}: {path}")
            continue

        data = load_json_artifact(path)
        models_data[name] = data

        cm = data["confusion_matrix"]
        table_rows.append({
            "Model": name,
            "Accuracy": data["accuracy"],
            "Precision": data["precision"],
            "Recall": data["recall"],
            "F1_Score": data["f1_score"],
            "Balanced_Acc": data["balanced_accuracy"],
            "ROC_AUC": data["roc_auc"],
            "TN": cm["tn"],
            "FP": cm["fp"],
            "FN": cm["fn"],
            "TP": cm["tp"]
        })

    if not table_rows:
        raise RuntimeError("No model metrics available! Please run Programs 2 through 6 first.")

    df_comp = pd.DataFrame(table_rows)

    # 2. Display formatted comparison table
    print("\n" + "=" * 95)
    print("FINAL BENCHMARK COMPARISON TABLE (HELD-OUT TEST SET)")
    print("=" * 95)
    display_df = df_comp.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1_Score", "Balanced_Acc", "ROC_AUC"]:
        display_df[col] = display_df[col].apply(lambda v: f"{v:.4f}")
    for col in ["TN", "FP", "FN", "TP"]:
        display_df[col] = display_df[col].apply(lambda v: f"{v:,}")

    print(display_df.to_string(index=False))
    print("=" * 95)

    # 3. Identify best models
    best_acc_row = df_comp.loc[df_comp["Accuracy"].idxmax()]
    best_rec_row = df_comp.loc[df_comp["Recall"].idxmax()]
    best_f1_row = df_comp.loc[df_comp["F1_Score"].idxmax()]

    print("\n--- PERFORMANCE CHAMPIONS ---")
    print(f"  • Best Overall Accuracy : {best_acc_row['Model']} ({best_acc_row['Accuracy']:.4%})")
    print(f"  • Best Attack Recall    : {best_rec_row['Model']} ({best_rec_row['Recall']:.4%})")
    print(f"  • Best F1-Score Balance : {best_f1_row['Model']} ({best_f1_row['F1_Score']:.4f})")

    # 4. Save CSV and JSON summary
    csv_path = os.path.join(OUT_DIR, "model_comparison.csv")
    df_comp.to_csv(csv_path, index=False)
    print(f"\n[Storage] Saved comparison CSV: {csv_path}")

    summary_report = {
        "benchmark_summary": {
            "total_test_flows": int(df_comp["TN"].iloc[0] + df_comp["FP"].iloc[0] + df_comp["FN"].iloc[0] + df_comp["TP"].iloc[0]),
            "best_accuracy_model": str(best_acc_row["Model"]),
            "best_accuracy_score": float(best_acc_row["Accuracy"]),
            "best_recall_model": str(best_rec_row["Model"]),
            "best_recall_score": float(best_rec_row["Recall"]),
            "best_f1_model": str(best_f1_row["Model"]),
            "best_f1_score": float(best_f1_row["F1_Score"]),
        },
        "models_evaluated": df_comp.to_dict(orient="records"),
        "platform_info": env_info
    }
    save_json_artifact(summary_report, os.path.join(OUT_DIR, "summary_report.json"))

    # 5. Generate Visualizations
    print("[Visualization] Generating multi-model dashboard comparison charts...")
    plot_grouped_metrics_bar(df_comp, os.path.join(OUT_DIR, "grouped_metrics_comparison.png"))
    plot_accuracy_ranking(df_comp, os.path.join(OUT_DIR, "accuracy_ranking.png"))
    plot_precision_recall_tradeoff(df_comp, os.path.join(OUT_DIR, "precision_recall_tradeoff.png"))
    plot_confusion_matrices_grid(models_data, os.path.join(OUT_DIR, "confusion_matrices_grid.png"))

    # 6. Synchronize measured results with pitch deck
    best_model_name = best_f1_row["Model"]
    sync_with_pitch_deck(best_model_name, models_data[best_model_name])

    print("\n" + "=" * 75)
    print("PROGRAM 7 COMPLETED SUCCESSFULLY!")
    print(f"All comparison outputs saved in: {OUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
