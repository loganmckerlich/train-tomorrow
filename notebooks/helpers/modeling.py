import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score

from model import _recency_weights

TARGET = "will_train_tomorrow"
TUNABLE = ("max_depth", "eta", "n_estimators", "min_child_weight", "reg_lambda", "subsample", "colsample_bytree")


def resolve_features(historical: pd.DataFrame, wanted: list[str], use_live: bool) -> list[str]:
    """Keep wanted features present in the data; stale cached CSVs may lack newer columns."""
    available = [f for f in wanted if f in historical.columns]
    missing = sorted(set(wanted) - set(available))
    if missing:
        if use_live:
            raise ValueError(f"Prepared data is missing configured model features: {missing}")
        print(f"Cached historical data lacks {missing}; using available features only. Set USE_LIVE_DATA=True to refresh it.")
    if not available:
        raise ValueError("No configured model features are available in the historical data")
    return available


def time_holdout_split(historical: pd.DataFrame, features: list[str], holdout_frac: float = 0.2):
    """Chronological split; the newest rows are the holdout. Returns X_search, X_hold, y_search, y_hold.

    X keeps an `as_of_date` column because RecencyXGB needs it for sample weights.
    """
    missing = {TARGET, "as_of_date"} - set(historical.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    data = historical.copy()
    data["as_of_date"] = pd.to_datetime(data["as_of_date"], errors="coerce")
    data = data.dropna(subset=["as_of_date", TARGET]).sort_values("as_of_date").reset_index(drop=True)
    X, y = data[["as_of_date", *features]], data[TARGET].astype(int)
    cut = int(len(X) * (1 - holdout_frac))
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]


# ClassifierMixin must precede BaseEstimator or sklearn >=1.6 doesn't see this as a classifier
# and the roc_auc scorer receives the full (n, 2) predict_proba output.
class RecencyXGB(ClassifierMixin, BaseEstimator):
    """XGBClassifier whose recency-weight half-life is a searchable hyperparameter."""

    def __init__(self, features=None, base_params=None, half_life_days=180.0, max_depth=3, eta=0.05,
                 n_estimators=300, min_child_weight=1, reg_lambda=1.0, subsample=0.9, colsample_bytree=0.9):
        self.features = features
        self.base_params = base_params
        self.half_life_days = half_life_days
        self.max_depth = max_depth
        self.eta = eta
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def fit(self, X, y):
        # weights are relative to the newest row of whatever data is passed (each CV fold's own train set)
        self.model_ = xgb.XGBClassifier(**{
            **(self.base_params or {}),
            **{k: getattr(self, k) for k in TUNABLE},
            "objective": "binary:logistic",
            "eval_metric": "logloss",
        })
        self.model_.fit(
            X[self.features], y, sample_weight=_recency_weights(X["as_of_date"], self.half_life_days)
        )
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X[self.features])

    def predict(self, X):
        return self.model_.predict(X[self.features])


def holdout_report(split, features: list[str], xgb_params: dict, half_life_days: float) -> dict:
    """Fit on the search window, score on the holdout; a big train/holdout AUC gap means overfitting."""
    X_search, X_hold, y_search, y_hold = split
    model = RecencyXGB(
        features=features,
        base_params=xgb_params,
        half_life_days=half_life_days,
        **{k: xgb_params[k] for k in TUNABLE if k in xgb_params},
    ).fit(X_search, y_search)
    return {
        "train_auc": roc_auc_score(y_search, model.predict_proba(X_search)[:, 1]),
        "holdout_auc": roc_auc_score(y_hold, model.predict_proba(X_hold)[:, 1]),
    }
