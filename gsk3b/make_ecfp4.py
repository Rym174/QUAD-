import os
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

INPUT_CSV = r"C:\Users\poula\Desktop\quad redo\gsk3b\gsk3b_clean.csv"
OUTPUT_CSV = r"C:\Users\poula\Desktop\quad redo\gsk3b\gsk3b_clean_ecfp4_2048.csv"

SMILES_COL = "smiles"
RADIUS = 2          # ECFP4 => radius 2
NBITS = 2048

def smiles_to_ecfp_bitstring(smiles: str, radius=RADIUS, nbits=NBITS):
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return fp.ToBitString()

def main():
    df = pd.read_csv(INPUT_CSV)

    if SMILES_COL not in df.columns:
        raise ValueError(f"Couldn't find column '{SMILES_COL}'. Columns are: {list(df.columns)}")

    # Generate fingerprints
    df["ecfp4_2048"] = df[SMILES_COL].apply(smiles_to_ecfp_bitstring)

    # Optional: keep track of invalid SMILES rows
    df["smiles_valid"] = df["ecfp4_2048"].notna().astype(int)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)

    n_invalid = (df["smiles_valid"] == 0).sum()
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Rows: {len(df)} | Invalid SMILES: {n_invalid}")

if __name__ == "__main__":
    main()
