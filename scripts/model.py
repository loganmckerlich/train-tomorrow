from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from features import FEATURE_COLUMNS

CLASSIFIER_MODEL_PATH = "classifier.json"
REGRESSOR_MODEL_PATH = "regressor.json"


@dataclass
class TrainedModels:
    classifier: xgb.XGBClassifier
    regressor: xgb.XGBRegressor


def _time_split(df: pd.DataFrame, frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 3:
        raise ValueError("Need at least 3 labeled rows for a time-based split")
    split_idx = max(1, int(len(df) * frac))
    split_idx = min(split_idx, len(df) - 1)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    positives = y_true == 1
    negatives = y_true == 0
    n_pos = int(positives.sum())
    n_neg = int(negatives.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    rank_sum_pos = float(ranks[positives].sum())
    return (rank_sum_pos - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg)


def train_and_save_models(dataset: pd.DataFrame, model_dir: Path) -> TrainedModels:
    ordered = dataset.sort_values("as_of_date").reset_index(drop=True)

    clf_train, clf_test = _time_split(ordered)
    X_train = clf_train[FEATURE_COLUMNS]
    y_train = clf_train["will_train_tomorrow"]
    X_test = clf_test[FEATURE_COLUMNS]
    y_test = clf_test["will_train_tomorrow"]

    classifier = xgb.XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
    )
    classifier.fit(X_train, y_train)

    probs = classifier.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    accuracy = float((preds == y_test.to_numpy()).mean())
    auc = _binary_auc(y_test.to_numpy(), probs)
    print(f"[model] classifier_accuracy={accuracy:.3f}")
    print(f"[model] classifier_auc={auc:.3f}" if not np.isnan(auc) else "[model] classifier_auc=nan")

    reg_rows = ordered[(ordered["will_train_tomorrow"] == 1) & ordered["next_day_relative_effort"].notna()].copy()
    if len(reg_rows) < 3:
        raise ValueError("Need at least 3 positive-label rows to train the effort regressor")

    reg_train, reg_test = _time_split(reg_rows)
    Xr_train = reg_train[FEATURE_COLUMNS]
    yr_train = reg_train["next_day_relative_effort"]
    Xr_test = reg_test[FEATURE_COLUMNS]
    yr_test = reg_test["next_day_relative_effort"]

    regressor = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
    )
    regressor.fit(Xr_train, yr_train)

    reg_preds = regressor.predict(Xr_test)
    mae = float(np.mean(np.abs(reg_preds - yr_test.to_numpy())))
    print(f"[model] regressor_mae={mae:.3f}")

    model_dir.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(model_dir / CLASSIFIER_MODEL_PATH))
    regressor.save_model(str(model_dir / REGRESSOR_MODEL_PATH))

    return TrainedModels(classifier=classifier, regressor=regressor)


def load_models(model_dir: Path) -> TrainedModels:
    classifier = xgb.XGBClassifier()
    regressor = xgb.XGBRegressor()
    classifier.load_model(str(model_dir / CLASSIFIER_MODEL_PATH))
    regressor.load_model(str(model_dir / REGRESSOR_MODEL_PATH))
    return TrainedModels(classifier=classifier, regressor=regressor)


def predict_tomorrow(models: TrainedModels, tomorrow_features: pd.DataFrame) -> dict[str, Any]:
    X = tomorrow_features[FEATURE_COLUMNS]
    probability = float(models.classifier.predict_proba(X)[0, 1])
    will_train = probability >= 0.5
    predicted_effort = float(models.regressor.predict(X)[0]) if will_train else None
    return {
        "will_train": bool(will_train),
        "probability": probability,
        "predicted_effort": predicted_effort,
    }


def feature_contributions(model: xgb.XGBClassifier, feature_row: pd.DataFrame) -> dict[str, float]:
    X = feature_row[FEATURE_COLUMNS]
    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_COLUMNS)
    contribs = model.get_booster().predict(dmatrix, pred_contribs=True)[0]
    # Last item is the bias term.
    return {name: float(value) for name, value in zip(FEATURE_COLUMNS, contribs[:-1], strict=True)}
