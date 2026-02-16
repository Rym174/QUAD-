import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =======================
# PATHS
# =======================
INPUT_CSV = r"C:\Users\poula\Desktop\quad redo\gsk3b\outputs\gsk3b_test_with_ad_risk.csv"
OUT_DIR   = r"C:\Users\poula\Desktop\quad redo\gsk3b\outputs"

# Columns
Y_COL     = "pIC50"
PRED_COL  = "rf_mean_pred"
UNC_COL   = "rf_uncertainty"
DIS_COL   = "disagreement"
AD_COL    = "ad_risk"

# =======================
# Helpers
# =======================
def minmax(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)

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
    df = df.dropna(subset=[Y_COL, PRED_COL, UNC_COL, DIS_COL, AD_COL]).copy()

    # Absolute error
    df["abs_error"] = (df[PRED_COL] - df[Y_COL]).abs()

    # Normalize components
    df["U_norm"]  = minmax(df[UNC_COL])
    df["D_norm"]  = minmax(df[DIS_COL])
    df["AD_norm"] = minmax(df[AD_COL])

    # QUAD score (equal weights)
    df["QUAD"] = (df["U_norm"] + df["D_norm"] + df["AD_norm"]) / 3.0

    # Correlation QUAD vs error
    x = df["QUAD"].to_numpy(dtype=float)
    y = df["abs_error"].to_numpy(dtype=float)

    pr = pearson_r(x, y)
    sr = spearman_rho(x, y)

    # Save CSV
    out_csv = os.path.join(OUT_DIR, "gsk3b_test_with_QUAD.csv")
    df.to_csv(out_csv, index=False)

    # ========== PLOT 1: error vs QUAD ==========
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, alpha=0.7)
    plt.xlabel("QUAD score")
    plt.ylabel("|Prediction error|")
    plt.title(f"GSK3β: |error| vs QUAD\nPearson r={pr:.3f}, Spearman ρ={sr:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "A_gsk3b_error_vs_QUAD.png"), dpi=300)
    plt.close()

    # ========== PLOT 2: comparison ==========
    plt.figure(figsize=(6, 6))
    plt.scatter(df["U_norm"],  y, alpha=0.5, label="Uncertainty")
    plt.scatter(df["D_norm"],  y, alpha=0.5, label="Disagreement")
    plt.scatter(df["AD_norm"], y, alpha=0.5, label="AD risk")
    plt.scatter(df["QUAD"],    y, alpha=0.8, label="QUAD")
    plt.xlabel("Normalized risk value")
    plt.ylabel("|Prediction error|")
    plt.legend()
    plt.title("GSK3β: Error vs individual risk components")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "B_gsk3b_error_vs_components.png"), dpi=300)
    plt.close()

    # ========== PLOT 3: QUAD distribution ==========
    plt.figure(figsize=(6, 6))
    plt.hist(df["QUAD"], bins=30)
    plt.xlabel("QUAD score")
    plt.ylabel("Count")
    plt.title("GSK3β: QUAD score distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "C_gsk3b_QUAD_distribution.png"), dpi=300)
    plt.close()

    print("✅ QUAD computed for GSK3β")
    print("Saved CSV:", out_csv)
    print(f"Pearson r (QUAD vs error): {pr:.4f}")
    print(f"Spearman ρ (QUAD vs error): {sr:.4f}")

if __name__ == "__main__":
    main()
