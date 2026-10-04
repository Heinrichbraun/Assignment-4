"""The model without pickles: preprocessing as plain numbers (preprocess.json) + XGBoost's own format (booster.json).

A pickled scikit-learn pipeline only loads with the exact library versions that saved it. These two files load with
any recent pandas and xgboost, on any Python, so the app needs no version pins.

  export(pipe, folder)   -> writes booster.json + preprocess.json   (needs scikit-learn; run where the model was trained)
  Model(folder)          -> .transform(df), .predict_proba(df), .contributions(df)   (needs only pandas, numpy, xgboost)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def export(pipe, folder):
    """Turn a make_pipeline(ColumnTransformer[num: imputer+scaler, cat: one-hot], XGBClassifier) into portable files."""
    folder = Path(folder)
    prep, clf = pipe[0], pipe[-1]
    num_pipe, num_cols = prep.named_transformers_["num"], prep.transformers_[0][2]
    imputer, scaler = num_pipe[0], num_pipe[-1]
    onehot, cat_cols = prep.named_transformers_["cat"], prep.transformers_[1][2]
    infrequent = onehot.infrequent_categories_ or [None] * len(cat_cols)
    spec = {
        "numeric": [{"name": c, "median": float(imputer.statistics_[i]), "mean": float(scaler.mean_[i]),
                     "scale": float(scaler.scale_[i])} for i, c in enumerate(num_cols)],
        "categorical": [{"name": c,
                         "frequent": [str(v) for v in onehot.categories_[i] if infrequent[i] is None or v not in infrequent[i]],
                         "has_infrequent": infrequent[i] is not None}
                        for i, c in enumerate(cat_cols)],
        "columns": [n.split("__", 1)[1] for n in prep.get_feature_names_out()],
    }
    (folder / "preprocess.json").write_text(json.dumps(spec, indent=1))
    clf.get_booster().save_model(folder / "booster.json")


class Model:
    def __init__(self, folder):
        folder = Path(folder)
        self.spec = json.loads((folder / "preprocess.json").read_text())
        self.booster = xgb.Booster()
        self.booster.load_model(folder / "booster.json")
        self.features = [n["name"] for n in self.spec["numeric"]] + [c["name"] for c in self.spec["categorical"]]

    def transform(self, df):
        """Same numbers the scikit-learn preprocessing produces: median-fill + standardise; one-hot with rare and
        unseen levels pooled into '<feature>_infrequent_sklearn' (as OneHotEncoder(handle_unknown='infrequent_if_exist'))."""
        parts = []
        for n in self.spec["numeric"]:
            x = pd.to_numeric(df[n["name"]], errors="coerce").astype(float).fillna(n["median"])
            parts.append(((x - n["mean"]) / n["scale"]).to_numpy()[:, None])
        for c in self.spec["categorical"]:
            v = df[c["name"]].astype(str).to_numpy()
            parts.append(np.stack([v == level for level in c["frequent"]], axis=1).astype(float))
            if c["has_infrequent"]:
                parts.append((~np.isin(v, c["frequent"])).astype(float)[:, None])
        return np.hstack(parts)

    def _dmatrix(self, df):
        return xgb.DMatrix(self.transform(df), feature_names=None)

    def predict_proba(self, df):
        """Probability of cancellation for each row (1-D array)."""
        return self.booster.predict(self._dmatrix(df))

    def contributions(self, df):
        """Per-row contribution of each original feature to the log-odds (one-hot columns summed back)."""
        contrib = self.booster.predict(self._dmatrix(df), pred_contribs=True)[:, :-1]
        owner = [next(f for f in self.features if col == f or col.startswith(f + "_")) for col in self.spec["columns"]]
        return pd.DataFrame(contrib, columns=owner, index=df.index).T.groupby(level=0).sum().T[self.features]
