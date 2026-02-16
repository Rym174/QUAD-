import os
import numpy as np
import pandas as pd

from rdkit import DataStructs

# =======================
# PATHS (GSK3β)
# =======================
SPLIT_CSV = r"C:\Users\poula\Desktop\quad redo\gsk3b\outputs\gsk3b_frozen_split_ecfp4_2048.csv"

# Test-set file that already has RF mean/uncertainty + MLP disagreement
TEST_WITH_DISAGREE_CSV = r"C:\Users\poula\Desktop\quad redo\gsk3b\outputs\gsk3b_test_with_disagreement.csv"

OUT_DIR = r"C:\Users\poula\Desktop\quad redo\gsk3b\outputs"

# Column names
SMILES_COL = "smiles"
FP_COL = "ecfp4_2048"
SPLIT_COL = "split"
VALID_COL = "smiles_valid"

# =======================
# Helpers
# =======================
def bitstring_to_bv(bitstring: str):
    return DataStructs.CreateFromBitString(bitstring)

def ensure_2048(bitstring: str, nbits=2048) -> bool:
    return isinstance(bitstring, str) and len(bitstring) == nbits and set(bitstring).issubset({"0", "1"})

# =======================
# Main
# =======================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- Load split (train+test)
    df = pd.read_csv(SPLIT_CSV)

    # Optional validity filter
    if VALID_COL in df.columns:
        df = df[df[VALID_COL] == 1].copy()

    # Basic checks
    for col in [SMILES_COL, FP_COL, SPLIT_COL]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in split CSV")

    df = df.dropna(subset=[SMILES_COL, FP_COL, SPLIT_COL]).copy()

    bad_fp = ~df[FP_COL].apply(ensure_2048)
    if bad_fp.any():
        raise ValueError("Invalid fingerprint detected in split CSV (not 2048-bit 0/1 string).")

    # ---- Build training fingerprint list
    train_df = df[df[SPLIT_COL] == "train"].copy()
    test_df  = df[df[SPLIT_COL] == "test"].copy()

    print("Train rows:", len(train_df))
    print("Test rows (from split):", len(test_df))

    print("Converting train fingerprints to RDKit bit vectors...")
    train_fps = [bitstring_to_bv(s) for s in train_df[FP_COL].tolist()]

    # ---- Load your existing test-with-disagreement file
    test_out = pd.read_csv(TEST_WITH_DISAGREE_CSV)

    if SMILES_COL not in test_out.columns:
        raise ValueError(f"'{SMILES_COL}' not found in {TEST_WITH_DISAGREE_CSV}")

    # ---- Merge fingerprints for test smiles
    test_fp_table = test_df[[SMILES_COL, FP_COL]].copy()

    merged = test_out.merge(test_fp_table, on=SMILES_COL, how="left", suffixes=("", "_from_split"))
    if merged[FP_COL].isna().any():
        missing = merged[merged[FP_COL].isna()][SMILES_COL].head(10).tolist()
        raise ValueError(
            "Some test SMILES in test_with_disagreement.csv were not found in the split CSV test set.\n"
            f"Examples: {missing}\n"
            "This usually means files came from different runs/splits."
        )

    # ---- Compute max Tanimoto similarity for each test molecule vs training set
    print("Computing max Tanimoto similarity (test -> train)...")
    max_sims = []
    n_test = len(merged)

    for i, fp_bits in enumerate(merged[FP_COL].tolist(), start=1):
        bv = bitstring_to_bv(fp_bits)
        sims = DataStructs.BulkTanimotoSimilarity(bv, train_fps)
        max_sim = float(max(sims)) if sims else float("nan")
        max_sims.append(max_sim)

        if i % 100 == 0 or i == n_test:
            print(f"  {i}/{n_test} done")

    merged["ad_max_tanimoto"] = max_sims
    merged["ad_risk"] = 1.0 - merged["ad_max_tanimoto"]

    # ---- Save
    out_csv = os.path.join(OUT_DIR, "gsk3b_test_with_ad_risk.csv")
    merged.to_csv(out_csv, index=False)

    print("\n✅ AD risk computed (GSK3β)")
    print("Saved:", out_csv)
    print("Columns added:")
    print("  ad_max_tanimoto (max similarity to training set)")
    print("  ad_risk = 1 - ad_max_tanimoto")

if __name__ == "__main__":
    main()
