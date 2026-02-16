import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import matplotlib.pyplot as plt

# =======================
# EDIT THESE PATHS
# =======================
INPUT_CSV = r"C:\Users\poula\Desktop\quad redo\gsk3b\gsk3b_clean_ecfp4_2048.csv"
OUT_DIR  = r"C:\Users\poula\Desktop\quad redo\gsk3b\outputs"

# Column names (adjust ONLY if your CSV uses different names)
SMILES_COL = "smiles"
Y_COL = "pIC50"
FP_COL = "ecfp4_2048"
VALID_COL = "smiles_valid"   # if not present, we'll skip this filter

# Split settings (freeze these)
TEST_SIZE = 0.20
SPLIT_SEED = 42  # pick one number and NEVER change it

# Baseline RF settings (for regression plot)
RF_SEED = 123
RF_PARAMS = dict(
    n_estimators=500,
    random_state=RF_SEED,
    n_jobs=-1
)

# =======================
# Helpers
# =======================
def bitstring_to_array(bitstring: str) -> np.ndarray:
    # bitstring expected like "010010..."
    return np.fromiter((1 if c == "1" else 0 for c in bitstring), dtype=np.int8)

def ensure_2048(bitstring: str, nbits=2048) -> bool:
    return isinstance(bitstring, str) and len(bitstring) == nbits and set(bitstring).issubset({"0", "1"})

# =======================
# Main
# =======================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    # Basic column checks
    for col in [SMILES_COL, Y_COL, FP_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Found columns: {list(df.columns)}")

    # Optional: filter invalid smiles
    if VALID_COL in df.columns:
        df = df[df[VALID_COL] == 1].copy()

    # Drop missing targets / fps
    df = df.dropna(subset=[Y_COL, FP_COL]).copy()

    # Validate fingerprint format
    bad_fp = ~df[FP_COL].apply(ensure_2048)
    n_bad = int(bad_fp.sum())
    if n_bad > 0:
        # safer to remove bad rows than crash mid-training
        df = df[~bad_fp].copy()
        print(f"Warning: removed {n_bad} rows with invalid fingerprint strings (not 2048 bits).")

    # Build X, y
    X = np.vstack(df[FP_COL].apply(bitstring_to_array).to_numpy())
    y = df[Y_COL].to_numpy(dtype=float)

    # Create frozen split
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SPLIT_SEED, shuffle=True
    )

    df["split"] = "train"
    df.loc[df.index[test_idx], "split"] = "test"

    split_csv = os.path.join(OUT_DIR, "gsk3b_frozen_split_ecfp4_2048.csv")
    df.to_csv(split_csv, index=False)
    print(f"Saved frozen split CSV: {split_csv}")

    # Train baseline RF for regression plot
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestRegressor(**RF_PARAMS)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    metrics_txt = os.path.join(OUT_DIR, "gsk3b_baseline_rf_metrics.txt")
    with open(metrics_txt, "w", encoding="utf-8") as f:
        f.write(f"Dataset: gsk3b\n")
        f.write(f"Input CSV: {INPUT_CSV}\n")
        f.write(f"Split: 80/20, seed={SPLIT_SEED}\n")
        f.write(f"Model: RandomForestRegressor\n")
        f.write(f"Params: {RF_PARAMS}\n\n")
        f.write(f"R2:   {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"MAE:  {mae:.4f}\n")
    print(f"Saved metrics: {metrics_txt}")

    # Regression plot (Measured vs Predicted)
    plot_path = os.path.join(OUT_DIR, "gsk3b_regression_measured_vs_predicted.png")
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    # diagonal y=x
    minv = min(y_test.min(), y_pred.min())
    maxv = max(y_test.max(), y_pred.max())
    plt.plot([minv, maxv], [minv, maxv], linewidth=1)

    plt.xlabel("Measured pIC50")
    plt.ylabel("Predicted pIC50")
    plt.title(f"gsk3b Test Set: Measured vs Predicted\nR2={r2:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved regression plot: {plot_path}")

if __name__ == "__main__":
    main()
