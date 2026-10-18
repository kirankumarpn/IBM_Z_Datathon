"""
program_3_autoencoder.py - Program 3: Neural Network Autoencoder
Implements a genuine deep neural network Autoencoder in PyTorch.
Trained strictly on normal network traffic, calculates reconstruction error,
selects the optimal anomaly threshold on validation data, and evaluates on test data.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.common import (
    calculate_comprehensive_metrics, plot_confusion_matrix, plot_roc_and_pr_curves,
    save_json_artifact, check_ibm_z_environment, OUTPUTS_DIR
)

OUT_DIR = os.path.join(OUTPUTS_DIR, "autoencoder")
os.makedirs(OUT_DIR, exist_ok=True)
DATA_ANALYSIS_DIR = os.path.join(OUTPUTS_DIR, "dataset_analysis")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class NetworkFlowAutoencoder(nn.Module):
    """
    Genuine Deep Neural Network Autoencoder Architecture.
    Encoder: Input -> 32 -> 16 -> Bottleneck (8)
    Decoder: Bottleneck (8) -> 16 -> 32 -> Output (Input dimension)
    """
    def __init__(self, input_dim: int, bottleneck_dim: int = 8):
        super(NetworkFlowAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, bottleneck_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def compute_reconstruction_errors(model, X_numpy, batch_size=512):
    """
    Computes per-sample Mean Squared Error (MSE) reconstruction loss.
    """
    model.eval()
    device = next(model.parameters()).device
    dataset = TensorDataset(torch.tensor(X_numpy, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    errors = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            # Row-wise MSE
            mse = torch.mean((batch_x - recon) ** 2, dim=1)
            errors.extend(mse.cpu().numpy())

    return np.array(errors)


def plot_reconstruction_error_distribution(errors, y_true, threshold, save_path):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    # Use log10 scale for reconstruction error visualization
    log_errors = np.log10(errors + 1e-6)
    log_thresh = np.log10(threshold + 1e-6)

    sns.kdeplot(log_errors[y_true == 0], ax=ax, label="Normal Flows (0)", color="#10b981", fill=True, alpha=0.3, lw=2)
    sns.kdeplot(log_errors[y_true == 1], ax=ax, label="Attack Flows (1)", color="#ef4444", fill=True, alpha=0.3, lw=2)

    ax.axvline(log_thresh, color="#f59e0b", lw=2, linestyle="--", label=f"Anomaly Threshold ({threshold:.4f})")
    ax.set_title("Autoencoder Reconstruction Error Distribution (Test Partition)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Log10(Reconstruction MSE)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Density", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")


def plot_threshold_tuning_curve(y_val, val_errors, save_path):
    percentiles = np.linspace(50, 99.9, 100)
    f1_scores = []
    threshold_vals = []

    for p in percentiles:
        t = np.percentile(val_errors[y_val == 0], p)
        preds = (val_errors >= t).astype(int)
        from sklearn.metrics import f1_score
        f1_scores.append(f1_score(y_val, preds, zero_division=0))
        threshold_vals.append(t)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.plot(percentiles, f1_scores, color="#8b5cf6", lw=2.5, label="Validation F1-Score")
    best_idx = np.argmax(f1_scores)
    best_p = percentiles[best_idx]
    best_f1 = f1_scores[best_idx]
    best_t = threshold_vals[best_idx]

    ax.scatter([best_p], [best_f1], color="#ef4444", s=80, zorder=5, label=f"Optimal: {best_p:.1f}th pct (F1={best_f1:.4f})")
    ax.set_xlabel("Normal Traffic Error Percentile Cutoff", fontsize=11, fontweight="bold")
    ax.set_ylabel("Validation F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("Autoencoder Anomaly Threshold Selection Curve", fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Viz] Saved: {save_path}")
    return best_t, best_f1


def main():
    print("=" * 75)
    print("PROGRAM 3: NEURAL NETWORK AUTOENCODER TRAINING & EVALUATION")
    print("IBM LinuxONE Community Cloud • Botnet Detection Pipeline")
    print("=" * 75)

    env_info = check_ibm_z_environment()
    print(f"[System] Platform: {env_info['runtime_notes']}")
    print(f"[Framework] PyTorch Version: {torch.__version__}")

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
    input_dim = len(feature_cols)
    print(f"[Features] Input dimension: {input_dim} features")

    # Filter normal training traffic only
    normal_mask = (df_train["Label"] == 0)
    X_train_normal = df_train.loc[normal_mask, feature_cols].values
    print(f"[Training Data] Autoencoder trains strictly on normal traffic: {len(X_train_normal):,} flows.")

    # 2. Fit StandardScaler on normal training data only
    print("[Preprocessing] Fitting StandardScaler on normal training flows only...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_normal)

    # Standardize validation and test sets using normal-fitted scaler
    X_val = df_val[feature_cols].values
    y_val = df_val["Label"].values.astype(int)
    X_val_scaled = scaler.transform(X_val)

    X_test = df_test[feature_cols].values
    y_test = df_test["Label"].values.astype(int)
    X_test_scaled = scaler.transform(X_test)

    # 3. Build and train Autoencoder model
    device = torch.device("cpu")
    model = NetworkFlowAutoencoder(input_dim=input_dim, bottleneck_dim=8).to(device)

    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-5)

    epochs = 15
    print(f"[Training] Training Autoencoder for {epochs} epochs on normal flows...")

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_x, in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch_x)

        epoch_loss = running_loss / len(X_train_scaled)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Reconstruction MSE: {epoch_loss:.6f}")

    print("[Training] Autoencoder training completed.")

    # 4. Compute reconstruction errors
    print("[Inference] Computing reconstruction losses across validation partition...")
    val_errors = compute_reconstruction_errors(model, X_val_scaled)

    # 5. Select anomaly threshold using validation data only
    print("[Threshold] Tuning anomaly threshold on validation set...")
    best_threshold, val_f1 = plot_threshold_tuning_curve(
        y_val, val_errors, os.path.join(OUT_DIR, "threshold_tuning_curve.png")
    )
    print(f"[Threshold] Selected Anomaly Threshold: {best_threshold:.6f} (Validation F1: {val_f1:.4f})")

    # 6. Evaluate on held-out test partition
    print("[Evaluation] Evaluating Autoencoder on held-out test partition...")
    test_errors = compute_reconstruction_errors(model, X_test_scaled)
    test_preds = (test_errors >= best_threshold).astype(int)

    # Normalize error to [0, 1] range for ROC-AUC computation
    max_err = np.percentile(val_errors, 99.5)
    test_probs = np.clip(test_errors / (max_err + 1e-8), 0.0, 1.0)
    val_probs = np.clip(val_errors / (max_err + 1e-8), 0.0, 1.0)

    metrics = calculate_comprehensive_metrics(y_test, test_preds, test_probs)
    metrics["anomaly_threshold"] = float(best_threshold)
    metrics["model_name"] = "Neural Network Autoencoder"
    metrics["validation_f1"] = float(val_f1)

    print("\n--- AUTOENCODER TEST BENCHMARK RESULTS ---")
    print(f"  Accuracy           : {metrics['accuracy']:.4f}")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  Attack Recall      : {metrics['recall']:.4f}")
    print(f"  F1-Score           : {metrics['f1_score']:.4f}")
    print(f"  Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"  Confusion Matrix   : TN={metrics['confusion_matrix']['tn']}, FP={metrics['confusion_matrix']['fp']}, "
          f"FN={metrics['confusion_matrix']['fn']}, TP={metrics['confusion_matrix']['tp']}")

    # 7. Save model and predictions
    print("[Storage] Saving Autoencoder weights, scaler, and prediction arrays...")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "autoencoder_model.pt"))
    joblib.dump(scaler, os.path.join(OUT_DIR, "ae_scaler.joblib"))

    # Save reconstruction error arrays
    np.save(os.path.join(OUT_DIR, "test_errors.npy"), test_errors)
    np.save(os.path.join(OUT_DIR, "val_errors.npy"), val_errors)
    np.save(os.path.join(OUT_DIR, "test_probabilities.npy"), test_probs)
    np.save(os.path.join(OUT_DIR, "val_probabilities.npy"), val_probs)

    pred_df = pd.DataFrame({
        "True_Label": y_test,
        "Reconstruction_Error": test_errors,
        "Predicted_Label": test_preds
    })
    pred_df.to_csv(os.path.join(OUT_DIR, "test_predictions.csv"), index=False)

    save_json_artifact(metrics, os.path.join(OUT_DIR, "autoencoder_metrics.json"))

    # 8. Generate Visualizations
    print("[Visualization] Generating Autoencoder evaluation charts...")
    plot_reconstruction_error_distribution(
        test_errors, y_test, best_threshold, os.path.join(OUT_DIR, "reconstruction_error_dist.png")
    )
    plot_confusion_matrix(metrics["confusion_matrix"], "Autoencoder — Test Confusion Matrix", os.path.join(OUT_DIR, "confusion_matrix.png"))
    plot_roc_and_pr_curves(y_test, test_probs, "Autoencoder", os.path.join(OUT_DIR, "roc_pr_curves.png"))

    print("\n" + "=" * 75)
    print("PROGRAM 3 COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {OUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    main()
