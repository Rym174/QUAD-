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
path = r"C:\Users\poula\Desktop\quad redo\Bace 1\bace1_clean.csv"
df = pd.read_csv(path)

# adjust column names if needed
smiles_col = "smiles"
target_col = "pIC50"

# ======================
# Generate ECFP4
# ======================
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return np.array(fp)

X = np.vstack(df[smiles_col].apply(featurize))
y = df[target_col].values

# ======================
# Train-test split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Model
# ======================
rf = RandomForestRegressor(n_estimators=500, random_state=123)

# ======================
# Q2CV
# ======================
cv = KFold(n_splits=5, shuffle=True, random_state=42)

y_cv_pred = cross_val_predict(rf, X_train, y_train, cv=cv)

q2_cv = r2_score(y_train, y_cv_pred)

# ======================
# Train final model
# ======================
rf.fit(X_train, y_train)
y_test_pred = rf.predict(X_test)

# ======================
# Q2ext
# IMPORTANT: use TRAIN mean
# ======================
y_train_mean = np.mean(y_train)

q2_ext = 1 - np.sum((y_test - y_test_pred)**2) / np.sum((y_test - y_train_mean)**2)

print("Q2CV:", q2_cv)
print("Q2ext:", q2_ext)