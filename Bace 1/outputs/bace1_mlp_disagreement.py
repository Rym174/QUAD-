import os
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# =======================
# PATHS
# =======================
# IMPORTANT:
# This file must include train+test rows (from the frozen split),
# and MUST include rf_mean_pred for test rows.
INPUT_SPLIT_CSV = r"C:\Users\poula\Desktop\quad redo\Bace 1\outputs\bace1_frozen_split_ecfp4_2048.csv"
RF_TEST_CSV     = r"C:\Users\poula\Desktop\quad redo\Bace 1\outputs\bace1_rf_ensemble_uncertainty.csv"
OUT_DIR         = r"C:\Users\poula\Desktop\quad redo\Bace 1\outputs"

# Columns
Y_COL      = "pIC50"
FP_COL     = "ecfp4_2048"
SPLIT_COL  = "split"
VALID_COL  = "smiles_valid"

RF_MEAN_COL = "rf_mean_pred"

# =======================
# FIXED MLP SETTINGS (NO TUNING)
# =======================
MLP_PARAMS = dict(
    hidden_layer_sizes=(256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    max_iter=500,
    random_state=123,
    early_stopping=True
)

# =======================
# Helpers
# =======================
def bitstring_to_array(bitstring: str) -> np.ndarray:
    return np.fromiter((1 if c == "1" else 0 for c in bitstring), dtype=np.int8)

def ensure_2048(bitstring: str, nbits=2048) -> bool:
    return isinstance(bitstring, str) and len(bitstring) == nbits and set(bitstring).issubset({"0", "1"})

# =======================
# Main
# =======================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Load full split (train + test)
    df = pd.read_csv(INPUT_SPLIT_CSV)

    if VALID_COL in df.columns:
        df = df[df[VALID_COL] == 1].copy()

    df = df.dropna(subset=[Y_COL, FP_COL, SPLIT_COL]).copy()

    bad_fp = ~df[FP_COL].apply(ensure_2048)
    if bad_fp.any():
        raise ValueError("Invalid fingerprint detected")

    # ---- Build X/y
    X = np.vstack(df[FP_COL].apply(bitstring_to_array).to_numpy())
    y = df[Y_COL].to_numpy(dtype=float)

    train_mask = (df[SPLIT_COL] == "train").to_numpy()
    test_mask  = (df[SPLIT_COL] == "test").to_numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_test           = X[test_mask]

    # ---- Scale (important for MLP; fit on TRAIN only)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ---- Train MLP
    print("Training fixed MLP for disagreement...")
    mlp = MLPRegressor(**MLP_PARAMS)
    mlp.fit(X_train_s, y_train)

    # ---- Predict
    print("Predicting with MLP...")
    y_mlp = mlp.predict(X_test_s)

    # ---- Load RF mean predictions (test only)
    rf_test = pd.read_csv(RF_TEST_CSV)

    if RF_MEAN_COL not in rf_test.columns:
        raise ValueError(f"Missing '{RF_MEAN_COL}' in RF ensemble CSV")

    # Ensure alignment by row order (both are test-set only and originate from same split)
    rf_mean = rf_test[RF_MEAN_COL].to_numpy(dtype=float)

    if len(rf_mean) != len(y_mlp):
        raise ValueError("Length mismatch between RF and MLP predictions")

    # ---- Disagreement
    disagreement = np.abs(rf_mean - y_mlp)

    # ---- Save output
    out_df = rf_test.copy()
    out_df["mlp_pred"] = y_mlp
    out_df["disagreement"] = disagreement

    out_csv = os.path.join(OUT_DIR, "bace1_test_with_disagreement.csv")
    out_df.to_csv(out_csv, index=False)

    print("\n✅ MLP disagreement computed for BACE1")
    print("Saved CSV:", out_csv)
    print("Columns added:")
    print("  mlp_pred")
    print("  disagreement = |rf_mean_pred - mlp_pred|")

if __name__ == "__main__":
    main()
