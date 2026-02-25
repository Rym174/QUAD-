import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ======================
# Load dataset
# ======================
path = r"C:\Users\poula\Desktop\quad redo\gsk3b\gsk3b_clean.csv"
df = pd.read_csv(path)

# ---- strip column names (handles hidden spaces)
df.columns = df.columns.str.strip()

# ---- set correct columns
smiles_col = "smiles"
target_col = "pIC50"

# ======================
# Clean target + drop missing rows
# ======================
df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
df = df.dropna(subset=[smiles_col, target_col]).reset_index(drop=True)

print("Columns:", list(df.columns))
print("Rows after cleaning SMILES+pIC50:", len(df))
print("NaNs remaining in target:", int(df[target_col].isna().sum()))

# ======================
# Featurize with RDKit, drop invalid SMILES
# ======================
def featurize_ecfp4(smiles, n_bits=2048, radius=2):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp, dtype=np.int8)

X_list, y_list = [], []

for smi, val in zip(df[smiles_col], df[target_col]):
    arr = featurize_ecfp4(smi)
    if arr is None:
        continue
    X_list.append(arr)
    y_list.append(float(val))

X = np.vstack(X_list)
y = np.array(y_list, dtype=float)

print("Usable molecules after RDKit parsing:", len(y))

# ======================
# Train-test split (match your paper)
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Model (match your paper)
# ======================
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=123,
    n_jobs=-1
)

# ======================
# Q2CV (5-fold CV on TRAIN only)
# ======================
cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_cv_pred = cross_val_predict(rf, X_train, y_train, cv=cv, n_jobs=-1)

q2_cv = r2_score(y_train, y_cv_pred)

# ======================
# Q2ext (external predictive Q2)
# IMPORTANT: denominator uses TRAIN mean
# ======================
rf.fit(X_train, y_train)
y_test_pred = rf.predict(X_test)

y_train_mean = float(np.mean(y_train))
q2_ext = 1 - np.sum((y_test - y_test_pred) ** 2) / np.sum((y_test - y_train_mean) ** 2)

print("Q2CV:", q2_cv)
print("Q2ext:", q2_ext)