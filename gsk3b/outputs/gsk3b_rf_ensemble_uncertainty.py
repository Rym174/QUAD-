import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

# =======================
# PATHS (EDIT IF NEEDED)
# =======================
INPUT_SPLIT_CSV = r"C:\Users\poula\Desktop\quad redo\gsk3b\outputs\gsk3b_frozen_split_ecfp4_2048.csv"
OUT_DIR = r"C:\Users\poula\Desktop\quad redo\gsk3b\outputs"

# Columns
Y_COL = "pIC50"
FP_COL = "ecfp4_2048"
SPLIT_COL = "split"
VALID_COL = "smiles_valid"

# =======================
# RF ENSEMBLE SETTINGS
# =======================
N_ENSEMBLE = 40          # 🔒 FIXED
BASE_RF_SEED = 1000     # base seed; each model uses seed+ i

RF_PARAMS = dict(
    n_estimators=500,
    n_jobs=-1,
    bootstrap=True,
    max_features=1.0
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

    # ---- Load
    df = pd.read_csv(INPUT_SPLIT_CSV)
    print("Loaded:", INPUT_SPLIT_CSV)
    print("Rows:", len(df))

    # ---- Basic checks
    for col in [Y_COL, FP_COL, SPLIT_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'")

    if VALID_COL in df.columns:
        df = df[df[VALID_COL] == 1].copy()

    df = df.dropna(subset=[Y_COL, FP_COL, SPLIT_COL]).copy()

    bad_fp = ~df[FP_COL].apply(ensure_2048)
    if bad_fp.any():
        raise ValueError("Invalid fingerprint detected (not 2048-bit 0/1 string)")

    # ---- Build X/y
    X = np.vstack(df[FP_COL].apply(bitstring_to_array).to_numpy())
    y = df[Y_COL].to_numpy(dtype=float)

    train_mask = (df[SPLIT_COL] == "train").to_numpy()
    test_mask  = (df[SPLIT_COL] == "test").to_numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_test = X[test_mask]

    print("Train size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])

    # ---- RF Ensemble
    all_preds = []

    print(f"Training RF ensemble (n = {N_ENSEMBLE})...")
    for i in range(N_ENSEMBLE):
        seed = BASE_RF_SEED + i
        rf = RandomForestRegressor(
            random_state=seed,
            **RF_PARAMS
        )
        print(f"  RF {i+1}/{N_ENSEMBLE} (seed={seed})")
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        all_preds.append(preds)

    preds_array = np.vstack(all_preds)  # shape (n_ensemble, n_test)

    # ---- Ensemble statistics
    mean_pred = preds_array.mean(axis=0)
    std_pred  = preds_array.std(axis=0, ddof=1)  # 🔒 sample SD

    # ---- Save output
    out_df = df.loc[test_mask].copy()
    out_df["rf_mean_pred"] = mean_pred
    out_df["rf_uncertainty"] = std_pred

    out_csv = os.path.join(OUT_DIR, "gsk3b_rf_ensemble_uncertainty.csv")
    out_df.to_csv(out_csv, index=False)

    print("\nSaved RF ensemble uncertainty CSV:")
    print(out_csv)
    print("\nColumns added:")
    print("  rf_mean_pred")
    print("  rf_uncertainty (sample SD, ddof=1)")

if __name__ == "__main__":
    main()
