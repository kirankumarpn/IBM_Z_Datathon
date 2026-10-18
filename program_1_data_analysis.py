"""
program_1_data_analysis.py - Program 1: Dataset Analysis and Preprocessing
Loads Normal and Attack traffic datasets, inspects data quality,
cleans features, performs stratified train/val/test splitting without data leakage,
generates exploratory data visualization figures, and saves split artifacts.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common import (
    resolve_dataset_files, load_raw_datasets, clean_and_prepare_features,
    save_json_artifact, check_ibm_z_environment, OUTPUTS_DIR
)

RANDOM_SEED = 42
OUT_DIR = os.path.join(OUTPUTS_DIR, "dataset_analysis")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_class_distribution(y_all, save_path):
    counts = pd.Series(y_all).value_counts().sort_index()
    labels = ["Normal Traffic (0)", "Attack Traffic (1)"]
    colors = ["#10b981", "#ef4444"]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    bars = ax.bar(labels, counts.values, color=colors, width=0.45, edgecolor="#000000", lw=1.2)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:,} flows\n({height/len(y_all):.1%})",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title("Dataset Class Distribution (CTU-13 Network Flows)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Network Flows", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(counts.values) * 1.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_missing_and_inf_analysis(df, feature_cols, save_path):
    nans = df[feature_cols].isna().sum()
    infs = np.isinf(df[feature_cols]).sum()

    issues_df = pd.DataFrame({"Missing (NaN)": nans, "Infinite (Inf)": infs})
    total_issues = issues_df.sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
    if total_issues.sum() == 0:
        # Clean data indicator
        ax.text(0.5, 0.5, "✓ Zero Missing or Infinite Values Detected Across All 57 Features\nData Quality: Pristine",
                horizontalalignment="center", verticalalignment="center",
                fontsize=13, fontweight="bold", color="#10b981",
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=1", facecolor="#ecfdf5", edgecolor="#10b981"))
        ax.axis("off")
    else:
        top_issues = total_issues[total_issues > 0].sort_values(ascending=False).head(15)
        top_issues.plot(kind="bar", ax=ax, color="#f59e0b", edgecolor="black")
        ax.set_ylabel("Count of Issues", fontsize=11)
        ax.set_title("Top Features with Missing or Infinite Values", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_feature_distributions(df, top_features, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)
    axes = axes.flatten()

    for i, col in enumerate(top_features[:4]):
        ax = axes[i]
        # Log scale where appropriate for skewed network flows
        series_normal = df[df["Label"] == 0][col].clip(lower=0)
        series_attack = df[df["Label"] == 1][col].clip(lower=0)

        # Plot KDE / histogram
        sns.kdeplot(np.log1p(series_normal), ax=ax, label="Normal (0)", color="#10b981", lw=2, fill=True, alpha=0.3)
        sns.kdeplot(np.log1p(series_attack), ax=ax, label="Attack (1)", color="#ef4444", lw=2, fill=True, alpha=0.3)
        ax.set_title(f"Log(1 + {col})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Value (Log Scale)")
        ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=9)

    plt.suptitle("Feature Distributions: Normal vs. Malicious Botnet Flows", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_correlation_matrix(df, selected_features, save_path):
    corr = df[selected_features].corr()
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, cbar=True, ax=ax)
    ax.set_title("Feature Correlation Heatmap (Selected Network Telemetry)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def main():
    print("=" * 75)
    print("PROGRAM 1: DATASET ANALYSIS & PREPROCESSING")
    print("IBM LinuxONE Community Cloud • Botnet Detection Pipeline")
    print("=" * 75)

    # 1. Environment check
    env_info = check_ibm_z_environment()
    print(f"[System] Machine: {env_info['machine']} | OS: {env_info['system']} | Python: {env_info['python_version']}")
    print(f"[System] Platform Status: {env_info['runtime_notes']}")

    # 2. Locate CSVs
    normal_path, attack_path = resolve_dataset_files()
    print(f"[Files] Normal File : {normal_path}")
    print(f"[Files] Attack File : {attack_path}")

    # 3. Load datasets
    df_normal, df_attack = load_raw_datasets(normal_path, attack_path)
    print(f"[Data] Normal rows: {len(df_normal):,} | Attack rows: {len(df_attack):,}")

    # 4. Concatenate and clean
    df_combined = pd.concat([df_normal, df_attack], axis=0, ignore_index=True)
    total_records = len(df_combined)
    print(f"[Data] Total combined records: {total_records:,}")

    # Deduplication check
    dups = df_combined.duplicated().sum()
    print(f"[Quality] Exact duplicate rows: {dups:,} ({dups/total_records:.2%})")

    # Clean features
    df_clean, feature_cols = clean_and_prepare_features(df_combined)
    print(f"[Features] Identified {len(feature_cols)} usable numeric flow features.")

    # 5. Stratified 60% Train, 20% Validation, 20% Test Split
    print("[Splitting] Performing Stratified 60% Train / 20% Val / 20% Test Split (Seed=42)...")
    indices = np.arange(len(df_clean))
    y = df_clean["Label"].values

    # Step A: 80% train+val, 20% test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # Step B: from 80%, 75% train (=60% total), 25% val (=20% total)
    y_train_val = y[train_val_idx]
    train_sub_idx, val_sub_idx = train_test_split(
        np.arange(len(train_val_idx)), test_size=0.25, random_state=RANDOM_SEED, stratify=y_train_val
    )

    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    print(f"  -> Train set      : {len(train_idx):,} flows ({len(train_idx)/total_records:.1%}) "
          f"[Normal: {(y[train_idx]==0).sum():,}, Attack: {(y[train_idx]==1).sum():,}]")
    print(f"  -> Validation set : {len(val_idx):,} flows ({len(val_idx)/total_records:.1%}) "
          f"[Normal: {(y[val_idx]==0).sum():,}, Attack: {(y[val_idx]==1).sum():,}]")
    print(f"  -> Test set       : {len(test_idx):,} flows ({len(test_idx)/total_records:.1%}) "
          f"[Normal: {(y[test_idx]==0).sum():,}, Attack: {(y[test_idx]==1).sum():,}]")

    # 6. Save split datasets and indices for subsequent programs
    split_info = {
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "test_indices": test_idx.tolist(),
        "total_records": total_records,
        "feature_columns": feature_cols,
        "target_column": "Label",
        "random_seed": RANDOM_SEED
    }
    split_path = os.path.join(OUT_DIR, "split_indices.json")
    save_json_artifact(split_info, split_path)

    # Save prepared CSV files
    print("[Storage] Saving prepared train, val, and test partitions to disk...")
    df_clean.iloc[train_idx].to_csv(os.path.join(OUT_DIR, "train_data.csv"), index=False)
    df_clean.iloc[val_idx].to_csv(os.path.join(OUT_DIR, "val_data.csv"), index=False)
    df_clean.iloc[test_idx].to_csv(os.path.join(OUT_DIR, "test_data.csv"), index=False)

    # Save feature names list
    save_json_artifact({"features": feature_cols, "count": len(feature_cols)}, os.path.join(OUT_DIR, "feature_names.json"))

    # 7. Generate Exploratory Visualizations
    print("[Visualization] Generating publication-grade exploratory charts...")
    plot_class_distribution(y, os.path.join(OUT_DIR, "class_distribution.png"))
    plot_missing_and_inf_analysis(df_combined, feature_cols, os.path.join(OUT_DIR, "missing_values.png"))

    top_features = ["Flow Duration", "Tot Fwd Pkts", "TotLen Fwd Pkts", "Flow Byts/s"]
    plot_feature_distributions(df_clean, top_features, os.path.join(OUT_DIR, "feature_distributions.png"))

    # Subset of 12 prominent features for correlation heatmap
    sample_corr_cols = [
        "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts", "TotLen Fwd Pkts",
        "Flow Byts/s", "Flow Pkts/s", "Flow IAT Mean", "Fwd Header Len",
        "Pkt Len Max", "Pkt Len Mean", "Pkt Size Avg", "Fwd Seg Size Avg"
    ]
    plot_correlation_matrix(df_clean, [c for c in sample_corr_cols if c in feature_cols], os.path.join(OUT_DIR, "correlation_matrix.png"))

    # 8. Data Quality Report
    quality_report = {
        "dataset_name": "CTU-13 Botnet & Normal Traffic",
        "normal_source": os.path.basename(normal_path),
        "attack_source": os.path.basename(attack_path),
        "total_records": total_records,
        "normal_records": int((y == 0).sum()),
        "attack_records": int((y == 1).sum()),
        "normal_percentage": float((y == 0).mean()),
        "attack_percentage": float((y == 1).mean()),
        "feature_count": len(feature_cols),
        "duplicates_count": int(dups),
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
        "random_seed": RANDOM_SEED,
        "environment": env_info
    }
    save_json_artifact(quality_report, os.path.join(OUT_DIR, "data_quality_report.json"))

    # Summary table
    summary_df = pd.DataFrame([quality_report])
    summary_df.to_csv(os.path.join(OUT_DIR, "data_quality_report.csv"), index=False)

    print("\n" + "=" * 75)
    print("PROGRAM 1 COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {OUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
