import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =======================
# PATHS
# =======================
INPUT_CSV = r"C:\Users\poula\Desktop\quad redo\gsk3b\outputs\gsk3b_rf_ensemble_uncertainty.csv"
OUT_DIR   = r"C:\Users\poula\Desktop\quad redo\gsk3b\outputs"

Y_COL    = "pIC50"
PRED_COL = "rf_mean_pred"
UNC_COL  = "rf_uncertainty"

# =======================
# Correlation helpers (no scipy)
# =======================
def pearson_r(x, y):
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    return float((x * y).sum() / denom)

def rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(a) + 1)

    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j+1]] = (i + j + 2) / 2
        i = j + 1
    return ranks

def spearman_rho(x, y):
    return pearson_r(rankdata(x), rankdata(y))

# =======================
# Main
# =======================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=[Y_COL, PRED_COL, UNC_COL]).copy()

    # Absolute error
    df["abs_error"] = (df[PRED_COL] - df[Y_COL]).abs()

    x = df[UNC_COL].to_numpy()
    y = df["abs_error"].to_numpy()

    pr = pearson_r(x, y)
    sr = spearman_rho(x, y)

    # Save augmented CSV
    out_csv = os.path.join(OUT_DIR, "gsk3b_test_with_error.csv")
    df.to_csv(out_csv, index=False)

    # ========= PLOT A =========
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel("RF ensemble uncertainty (sample SD)")
    plt.ylabel("|Prediction error|")
    plt.title(f"GSK3β: |error| vs uncertainty\nPearson r={pr:.3f}, Spearman ρ={sr:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "A_gsk3b_error_vs_uncertainty.png"), dpi=300)
    plt.close()

    # ========= PLOT B =========
    eps = 1e-6
    ylog = np.log10(y + eps)
    pr_log = pearson_r(x, ylog)
    sr_log = spearman_rho(x, ylog)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, ylog, alpha=0.7)
    plt.xlabel("RF ensemble uncertainty (sample SD)")
    plt.ylabel("log10(|error|)")
    plt.title(f"GSK3β: log|error| vs uncertainty\nPearson r={pr_log:.3f}, Spearman ρ={sr_log:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "B_gsk3b_logerror_vs_uncertainty.png"), dpi=300)
    plt.close()

    # ========= PLOT C =========
    plt.figure(figsize=(6, 6))
    plt.scatter(df[PRED_COL], x, alpha=0.7)
    plt.xlabel("Predicted pIC50 (RF mean)")
    plt.ylabel("RF ensemble uncertainty")
    plt.title("GSK3β: uncertainty vs predicted activity")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "C_gsk3b_uncertainty_vs_prediction.png"), dpi=300)
    plt.close()

    # ========= PLOT D =========
    plt.figure(figsize=(6, 6))
    plt.hist(x, bins=30)
    plt.xlabel("RF ensemble uncertainty")
    plt.ylabel("Count")
    plt.title("GSK3β: distribution of RF uncertainty")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "D_gsk3b_uncertainty_distribution.png"), dpi=300)
    plt.close()

    print("✅ ALL GSK3β uncertainty plots generated")
    print("Saved CSV:", out_csv)
    print(f"Pearson r (|error|): {pr:.4f}")
    print(f"Spearman ρ (|error|): {sr:.4f}")

if __name__ == "__main__":
    main()
