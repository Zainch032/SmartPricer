"""
Phone price model — "how little can we ask the user?" experiment + final model.

Run:
    python train.py --data data/Clean2_Smart_Phones.csv

What it does
------------
1. Loads and lightly sanitises your cleaned CSV (see load_clean()).
2. ABLATION: cross-validates the same model on progressively smaller input sets,
   so you can SEE how much accuracy each group of questions buys you.
3. Trains the final model on the questions the app actually asks, using
   "feature dropout" so that it still gives a sensible answer when the user
   skips the optional ones.
4. Measures an honest error band for each mode (basic / detailed), so the
   app can say "≈ ₹21,000  (likely ₹17,500 – ₹25,500)" instead of a fake-precise number.

Everything (imputation, encoding) lives inside the Pipeline, so nothing is fitted on test data.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

# --------------------------------------------------------------------------
# Feature groups: what a normal phone owner can answer vs. what only a spec-nerd knows
# --------------------------------------------------------------------------
REQUIRED = ["Brand", "Ram", "Memory", "Has_5G"]  # 4 questions, all visible in Settings > About phone
OPTIONAL = [                                      # each one can be skipped
    "Screen_size",      # inches   (phone box / Settings)
    "Battery",          # mAh      (Settings > Battery or box)
    "Back_Camera_MP",   # main camera MP (box / camera app)
    "Has_NFC",          # Google Pay / tap-to-pay works?
    "Fast_Charge",      # watts, 0 = none (charger label)
    "os_version",       # Android / iOS major version (Settings > About)
]
CATEGORICAL = ["Brand"]
# Things most owners can't answer. Used ONLY in the ablation, to show what they add.
NERD = ["Processor_Model", "Processor_Core", "Processor_Speed", "Refresh_Rate",
        "Resolution", "Num_Back_Cam", "Num_Front_Cam", "Front_CAM_MP", "Has_IR_Blaster",
        "card_support", "slot_type", "capacity_gb", "OS_Name"]
CATEGORICAL_NERD = ["Processor_Model", "Resolution", "slot_type", "OS_Name"]

MAX_PRICE = 200_000      # beyond this are 1-off luxury/mis-priced rows (Vertu 650k, "Redmi K20 Pro Signature" 480k)
MIN_BRAND_COUNT = 5      # rarer brands become "Other"


def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    df["Brand"] = df["Brand"].astype(str).str.strip().str.title()
    # Redmi / Poco are Xiaomi sub-brands: users think of them as one family
    df["Brand"] = df["Brand"].replace({"Redmi": "Xiaomi", "Poco": "Xiaomi", "Iqoo": "iQOO",
                                       "Oneplus": "OnePlus", "Lg": "LG", "Zte": "ZTE", "Tcl": "TCL"})
    vc = df["Brand"].value_counts()
    df.loc[df["Brand"].isin(vc[vc < MIN_BRAND_COUNT].index), "Brand"] = "Other"

    for c in ["Has_5G", "Has_NFC", "Has_IR_Blaster"]:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int)

    numeric = ["price", "Ram", "Memory", "Battery", "Screen_size", "Back_Camera_MP", "Fast_Charge",
               "os_version", "Processor_Core", "Processor_Speed", "Refresh_Rate", "Num_Back_Cam",
               "Num_Front_Cam", "Front_CAM_MP", "card_support", "capacity_gb", "rating"]
    for c in numeric:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 0 rating == "no rating" in the source data
    if "rating" in df:
        df.loc[df["rating"] == 0, "rating"] = np.nan

    df = df[(df["price"] > 0) & (df["price"] <= MAX_PRICE)].reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def make_model(features, categorical):
    cats = [f for f in features if f in categorical]
    nums = [f for f in features if f not in categorical]
    prep = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), cats),
         ("num", "passthrough", nums)],
        verbose_feature_names_out=False,
    )
    hgb = HistGradientBoostingRegressor(
        categorical_features=[True] * len(cats) + [False] * len(nums),
        learning_rate=0.06, max_iter=400, max_leaf_nodes=15, min_samples_leaf=8,
        l2_regularization=1.0, random_state=0,
    )
    # Predict log(price): phone prices are heavily right-skewed, so errors are relative, not absolute.
    return TransformedTargetRegressor(Pipeline([("prep", prep), ("hgb", hgb)]),
                                      func=np.log1p, inverse_func=np.expm1)


def drop_features(X: pd.DataFrame, cols_to_hide, rng, copies=3):
    """Feature-dropout augmentation: each copy hides a random share of the optional columns per row,
    so the model learns to cope with 'user skipped this question'."""
    parts = [X]
    for _ in range(copies):
        Xc = X.copy()
        p = rng.uniform(0.0, 1.0, size=len(Xc))  # row-specific skip rate
        for c in cols_to_hide:
            mask = rng.uniform(size=len(Xc)) < p
            Xc[c] = Xc[c].astype("object").where(~mask, np.nan) if c in CATEGORICAL else Xc[c].where(~mask, np.nan)
        parts.append(Xc)
    return pd.concat(parts, ignore_index=True)


def hide(X, keep):
    Xh = X.copy()
    for c in Xh.columns:
        if c not in keep:
            Xh[c] = np.nan
    return Xh


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ape = np.abs(p - y) / y
    ss_res, ss_tot = ((y - p) ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return {"R2": 1 - ss_res / ss_tot, "MAE": np.abs(p - y).mean(),
            "MdAPE%": 100 * np.median(ape), "within20%": 100 * (ape <= 0.20).mean()}


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def ablation(df, n_repeats=3):
    all_known = REQUIRED + OPTIONAL
    sets = {
        "A. ORIGINAL: everything incl. rating (what your app asks today: 24 fields)": (all_known + NERD + (["rating"] if "rating" in df else []), CATEGORICAL + CATEGORICAL_NERD),
        "B. Everything, rating removed": (all_known + NERD, CATEGORICAL + CATEGORICAL_NERD),
        "C. Easy-10 (4 required + 6 optional)": (all_known, CATEGORICAL),
        "D. Essentials-4 (brand, RAM, storage, 5G)": (REQUIRED, CATEGORICAL),
        "E. Brand + RAM + storage only (3 fields)": (["Brand", "Ram", "Memory"], CATEGORICAL),
    }
    y = df["price"].values
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=1)
    rows = {k: [] for k in sets}
    rows["F. FINAL: one model, trained with dropout — user answers 4 fields"] = []
    rows["G. FINAL: same model — user answers all 10 fields"] = []
    rng = np.random.default_rng(0)

    for tr, te in cv.split(df):
        for name, (feats, cats) in sets.items():
            feats = [f for f in feats if f in df.columns]
            m = make_model(feats, [c for c in cats if c in feats]).fit(df.iloc[tr][feats], y[tr])
            rows[name].append(metrics(y[te], m.predict(df.iloc[te][feats])))
        # final-style model
        X = df[all_known]
        m = make_model(all_known, CATEGORICAL).fit(drop_features(X.iloc[tr], OPTIONAL, rng), np.tile(y[tr], 4))
        rows["F. FINAL: one model, trained with dropout — user answers 4 fields"].append(
            metrics(y[te], m.predict(hide(X.iloc[te], REQUIRED))))
        rows["G. FINAL: same model — user answers all 10 fields"].append(metrics(y[te], m.predict(X.iloc[te])))

    out = pd.DataFrame({k: pd.DataFrame(v).mean() for k, v in rows.items()}).T
    out.insert(0, "fields asked", [len([f for f in sets[k][0] if f in df.columns]) if k in sets else (4 if k.startswith("F") else 10) for k in out.index])
    return out.round(2)


def final_model(df, n_repeats=3):
    feats = REQUIRED + OPTIONAL
    X, y = df[feats], df["price"].values
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats, random_state=7)
    rng = np.random.default_rng(1)
    log_err = {"basic": [], "detailed": []}
    for tr, te in cv.split(X):
        m = make_model(feats, CATEGORICAL).fit(drop_features(X.iloc[tr], OPTIONAL, rng), np.tile(y[tr], 4))
        log_err["basic"] += list(np.log(m.predict(hide(X.iloc[te], REQUIRED)) / y[te]))
        log_err["detailed"] += list(np.log(m.predict(X.iloc[te]) / y[te]))
    bands = {k: {"lo": float(np.quantile(v, 0.10)), "hi": float(np.quantile(v, 0.90))} for k, v in log_err.items()}

    model = make_model(feats, CATEGORICAL).fit(drop_features(X, OPTIONAL, np.random.default_rng(2)), np.tile(y, 4))
    return model, bands


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/Clean2_Smart_Phones.csv")
    ap.add_argument("--out", default="model/model.joblib")
    ap.add_argument("--skip-ablation", action="store_true")
    a = ap.parse_args()

    df = load_clean(a.data)
    print(f"{len(df)} phones, price ₹{df.price.min():,.0f}–₹{df.price.max():,.0f}, median ₹{df.price.median():,.0f}\n")

    if not a.skip_ablation:
        res = ablation(df)
        pd.set_option("display.width", 200, "display.max_colwidth", 90)
        print("5-fold x3 cross-validation (MdAPE% = typical % error; within20% = share of phones priced within ±20%)\n")
        print(res.to_string())
        res.to_csv("model/ablation_results.csv")
        print()

    model, bands = final_model(df)
    brands = sorted(df["Brand"].unique().tolist(), key=lambda b: (b == "Other", b))
    joblib.dump({"model": model, "required": REQUIRED, "optional": OPTIONAL, "bands": bands, "brands": brands,
                 "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in ["Ram", "Memory", "Battery", "Screen_size"]}},
                a.out)
    print("error bands (log-ratio, 10th–90th pct):", json.dumps(bands))
    print("saved", a.out)


if __name__ == "__main__":
    main()
