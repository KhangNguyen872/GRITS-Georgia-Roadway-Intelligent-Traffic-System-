# train_for_1_6.py
import re, json, numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, classification_report

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

def load_csv(p):
    return pd.read_csv(p, low_memory=False)

def std(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Z0-9_]+', '', c.upper().strip().replace(" ", "_")) for c in df.columns]
    return df

# --- Load your three core files (same as prototype) ---
aadt_new = std(load_csv(DATA / "aadt_and_truckpct.csv"))
aadt_old = std(load_csv(DATA / "GDOT_Traffic_Counts_(AADT_and_Truck_Percent)_2008_to_2017.csv"))
annual   = std(load_csv(DATA / "annualized_statistics.csv"))

# Normalize old TRUCKPCT_* to TRUCK_*
aadt_old = aadt_old.rename(columns={c: c.replace("TRUCKPCT_", "TRUCK_") for c in aadt_old.columns if c.startswith("TRUCKPCT_")})

# Merge old+new on STATION_ID + FUNCTIONAL_CLASS (outer keep)
aadt_all = aadt_old.merge(aadt_new, on=["STATION_ID","FUNCTIONAL_CLASS"], how="outer", suffixes=("_OLD","_NEW"))

# Keep K/D factors & station type
annual_keep = annual[["STATION_ID","KFACTOR","DFACTOR","STATION_TYPE"]].drop_duplicates()
aadt_all = aadt_all.merge(annual_keep, on="STATION_ID", how="left")

# Lat/Lon
if "LAT" in aadt_all.columns: aadt_all["LAT"] = pd.to_numeric(aadt_all["LAT"], errors="coerce")
if "LATITUDE" in aadt_all.columns: aadt_all["LAT"] = pd.to_numeric(aadt_all["LATITUDE"], errors="coerce")
for loncand in ["LON","LONG","LONGITUDE"]:
    if loncand in aadt_all.columns:
        aadt_all["LON"] = pd.to_numeric(aadt_all[loncand], errors="coerce")
        break

# Years, features
aadt_year_cols  = sorted([c for c in aadt_all.columns if re.match(r"^AADT_\d{4}$", c)])
truck_year_cols = sorted([c for c in aadt_all.columns if re.match(r"^TRUCK_\d{4}$", c)])
for c in aadt_year_cols + truck_year_cols + ["KFACTOR","DFACTOR"]:
    if c in aadt_all.columns:
        aadt_all[c] = pd.to_numeric(aadt_all[c], errors="coerce")

years = [int(c.split("_")[1]) for c in aadt_year_cols] if aadt_year_cols else []
latest_year = max(years) if years else None
if latest_year is None:
    raise SystemExit("No AADT year columns found.")

aadt_all["AADT_LATEST"] = aadt_all.get(f"AADT_{latest_year}")

def slope(row, years_back=6):
    ys = sorted(years)[-years_back:]
    vals = [row.get(f"AADT_{y}", np.nan) for y in ys]
    xs, vs = zip(*[(y, v) for y, v in zip(ys, vals) if pd.notna(v)]) if any(pd.notna(v) for v in vals) else ([],[])
    if len(xs) >= 2:
        x = np.array(xs); v = np.array(vs)
        return float(np.polyfit(x, v, 1)[0])
    return np.nan

aadt_all["AADT_TREND_SLOPE"] = aadt_all.apply(slope, axis=1)

def roll3(row):
    ys = sorted(years)[-3:]
    vals = [row.get(f"AADT_{y}", np.nan) for y in ys]
    vals = [v for v in vals if pd.notna(v)]
    return pd.Series({"AADT_MEAN_3YR": np.mean(vals) if vals else np.nan,
                      "AADT_STD_3YR":  np.std(vals) if vals else np.nan})
aadt_all[["AADT_MEAN_3YR","AADT_STD_3YR"]] = aadt_all.apply(roll3, axis=1)

def mean_or_nan(row, cols):
    xs = [row.get(c, np.nan) for c in cols]
    xs = [float(x) for x in xs if pd.notna(x)]
    return float(np.mean(xs)) if xs else np.nan

aadt_all["TRUCK_PCT_MEAN"]   = aadt_all.apply(lambda r: mean_or_nan(r, truck_year_cols), axis=1)
aadt_all["TRUCK_PCT_LATEST"] = aadt_all.get(f"TRUCK_{latest_year}", np.nan)

valid = aadt_all[pd.notna(aadt_all["AADT_LATEST"])].copy()
thr = np.nanpercentile(valid["AADT_LATEST"], 75)
aadt_all["HIGH_CONGESTION_PROXY"] = (aadt_all["AADT_LATEST"] >= thr).astype(int)

num_feats = ["LAT","LON","AADT_MEAN_3YR","AADT_STD_3YR","AADT_TREND_SLOPE","TRUCK_PCT_MEAN","TRUCK_PCT_LATEST","KFACTOR","DFACTOR"]
cat_feats = [c for c in ["FUNCTIONAL_CLASS","STATION_TYPE"] if c in aadt_all.columns]
feature_cols = num_feats + cat_feats

dataset = aadt_all[feature_cols + ["HIGH_CONGESTION_PROXY"]].copy()

# Train/test
X = dataset[feature_cols]
y = dataset["HIGH_CONGESTION_PROXY"].astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pre = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_feats),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_feats)
    ],
    remainder="drop"
)

clf = Pipeline([
    ("preprocess", pre),
    ("model", GradientBoostingClassifier(random_state=42))
])

clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte)[:,1]
roc = roc_auc_score(yte, proba)
prec, rec, thr = precision_recall_curve(yte, proba)
pr_auc = auc(rec, prec)

print(f"ROC_AUC={roc:.6f}  PR_AUC={pr_auc:.6f}")
print(classification_report(yte, (proba>=0.5).astype(int), digits=3))

# Save artifacts
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)
joblib.dump(clf, "models/latest_model.pkl")
with open("outputs/feature_schema.json","w") as f:
    json.dump({"numeric_features": num_feats, "categorical_features": cat_feats,
               "label":"HIGH_CONGESTION_PROXY",
               "label_definition": f"1 if AADT_{latest_year} >= 75th percentile; else 0",
               "latest_year": latest_year}, f, indent=2)
with open("outputs/metrics.json","w") as f:
    json.dump({"ROC_AUC": float(roc), "PR_AUC": float(pr_auc)}, f, indent=2)

print("Saved models/latest_model.pkl (sklearn 1.6.x compatible)")
