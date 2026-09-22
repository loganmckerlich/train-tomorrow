from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support

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


def _recency_weights(as_of_dates: pd.Series, half_life_days: float = 180.0) -> np.ndarray:
    """Exponentially downweight older rows so training tracks current habits, not stale regimes.

    Training frequency can drift a lot over a multi-year history (season, injury, life changes);
    weighting by recency keeps the model responsive to how you train *now* rather than an average
    over behavior that no longer applies.
    """
    dates = pd.to_datetime(as_of_dates)
    age_days = (dates.max() - dates).dt.days.to_numpy()
    return np.power(0.5, age_days / half_life_days)


def walk_forward_evaluate(dataset: pd.DataFrame, n_folds: int = 5) -> pd.DataFrame:
    """Expanding-window backtest across multiple test windows, not just one 80/20 split.

    A single holdout can look better or worse than reality purely because of which season/period
    landed in the test slice. This retrains on an expanding window and evaluates on the next chunk
    each fold, so you can see whether classifier AUC is stable or swings a lot fold-to-fold.
    """
    ordered = dataset.sort_values("as_of_date").reset_index(drop=True)
    n = len(ordered)
    fold_edges = np.linspace(int(n * 0.5), n, n_folds + 1).astype(int)

    rows: list[dict[str, Any]] = []
    for fold in range(n_folds):
        train_end = fold_edges[fold]
        test_end = fold_edges[fold + 1]
        train = ordered.iloc[:train_end]
        test = ordered.iloc[train_end:test_end]
        if test.empty or train["will_train_tomorrow"].nunique() < 2:
            continue

        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            eta=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            n_estimators=300,
            random_state=42,
        )
        clf.fit(train[FEATURE_COLUMNS], train["will_train_tomorrow"], sample_weight=_recency_weights(train["as_of_date"]))
        probs = clf.predict_proba(test[FEATURE_COLUMNS])[:, 1]
        y_test = test["will_train_tomorrow"].to_numpy()
        preds = (probs >= 0.5).astype(int)

        rows.append(
            {
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "positive_rate": float(y_test.mean()),
                "auc": _binary_auc(y_test, probs),
                "accuracy": float((preds == y_test).mean()),
                "baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
            }
        )

    return pd.DataFrame(rows)


def train_and_save_models(dataset: pd.DataFrame, model_dir: Path) -> TrainedModels:
    ordered = dataset.sort_values("as_of_date").reset_index(drop=True)

    clf_train, clf_test = _time_split(ordered)
    y_train = clf_train["will_train_tomorrow"]
    y_test = clf_test["will_train_tomorrow"]

    classifier = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        eta=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        n_estimators=300,
        random_state=42,
    )
    classifier.fit(clf_train[FEATURE_COLUMNS], y_train, sample_weight=_recency_weights(clf_train["as_of_date"]))

    probs = classifier.predict_proba(clf_test[FEATURE_COLUMNS])[:, 1]
    preds = (probs >= 0.5).astype(int)
    accuracy = float((preds == y_test.to_numpy()).mean())
    baseline_accuracy = float(max(y_test.mean(), 1 - y_test.mean()))
    auc = _binary_auc(y_test.to_numpy(), probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test.to_numpy(), preds, average="binary", zero_division=0
    )
    print(f"[model] classifier_accuracy={accuracy:.3f} (baseline={baseline_accuracy:.3f})")
    print(f"[model] classifier_auc={auc:.3f}" if not np.isnan(auc) else "[model] classifier_auc=nan")
    print(f"[model] classifier_precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")

    reg_rows = ordered[ordered["will_train_tomorrow"] == 1].copy() # only predict effort when there is an effort, Binary model runs first
    if len(reg_rows) < 3:
        raise ValueError("Need at least 3 positive-label rows to train the effort regressor")

    reg_train, reg_test = _time_split(reg_rows)
    yr_train = reg_train["next_day_relative_effort"]
    yr_test = reg_test["next_day_relative_effort"]

    regressor = xgb.XGBRegressor(
        objective="reg:absoluteerror",  # optimize MAE directly, robust to the long-tail effort outliers
        eta=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        n_estimators=300,
        random_state=42,
    )
    regressor.fit(reg_train[FEATURE_COLUMNS], yr_train, sample_weight=_recency_weights(reg_train["as_of_date"]))

    reg_preds = regressor.predict(reg_test[FEATURE_COLUMNS])
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


def evaluate_models(models: TrainedModels, dataset: pd.DataFrame) -> dict[str, np.ndarray]:
    """Reproduce the same time-based test split used during training, for notebook analysis."""
    ordered = dataset.sort_values("as_of_date").reset_index(drop=True)

    _, clf_test = _time_split(ordered)
    probs = models.classifier.predict_proba(clf_test[FEATURE_COLUMNS])[:, 1]

    reg_rows = ordered[ordered["will_train_tomorrow"] == 1].copy()
    _, reg_test = _time_split(reg_rows)
    reg_preds = models.regressor.predict(reg_test[FEATURE_COLUMNS])

    return {
        "y_test": clf_test["will_train_tomorrow"].to_numpy(),
        "probs": probs,
        "reg_true": reg_test["next_day_relative_effort"].to_numpy(),
        "reg_pred": reg_preds,
    }


def predict_tomorrow(models: TrainedModels, tomorrow_features: pd.DataFrame) -> dict[str, Any]:
    features = tomorrow_features[FEATURE_COLUMNS]
    probability = float(models.classifier.predict_proba(features)[0, 1])
    will_train = probability >= 0.5
    predicted_effort = float(models.regressor.predict(features)[0]) if will_train else None
    return {
        "will_train": bool(will_train),
        "probability": probability,
        "predicted_effort": predicted_effort,
    }


def feature_contributions(model: xgb.XGBModel, feature_row: pd.DataFrame) -> dict[str, float]:
    matrix = xgb.DMatrix(feature_row[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
    contribs = model.get_booster().predict(matrix, pred_contribs=True)[0]
    return {name: float(value) for name, value in zip(FEATURE_COLUMNS, contribs[:-1], strict=True)}
