from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_fscore_support, roc_curve

from features import FEATURE_COLUMNS

CLASSIFIER_MODEL_PATH = "classifier.json"
REGRESSOR_MODEL_PATH = "regressor.json"

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "eta": 0.05,
    "max_depth": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "n_estimators": 300,
    "random_state": 42,
}

logger = logging.getLogger(__name__)


def _feature_list(features: list[str] | None) -> list[str]:
    return FEATURE_COLUMNS if features is None else features


@dataclass
class TrainedModels:
    classifier: xgb.XGBClassifier
    regressor: xgb.XGBRegressor
    validation_predictions: list[dict[str, Any]] | None = None
    metrics: dict[str, Any] | None = None


def _finite(value: float) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), 4)


def _classifier_holdout_metrics(
    y_true: np.ndarray, probs: np.ndarray, clf_train: pd.DataFrame, clf_test: pd.DataFrame
) -> dict[str, Any]:
    preds = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    two_classes = len(np.unique(y_true)) == 2
    roc: list[dict[str, float]] = []
    if two_classes:
        fpr, tpr, _ = roc_curve(y_true, probs)
        step = max(1, len(fpr) // 50)
        roc = [{"fpr": round(float(a), 4), "tpr": round(float(b), 4)} for a, b in zip(fpr[::step], tpr[::step])]
        roc.append({"fpr": 1.0, "tpr": 1.0})
    # Equal-width probability bins; empty bins are skipped.
    edges = np.linspace(0, 1, 6)
    bins = np.clip(np.digitize(probs, edges[1:-1]), 0, 4)
    calibration = [
        {
            "mean_predicted": round(float(probs[bins == b].mean()), 4),
            "observed_rate": round(float(y_true[bins == b].mean()), 4),
            "n": int((bins == b).sum()),
        }
        for b in range(5)
        if (bins == b).any()
    ]
    return {
        "n_train": len(clf_train),
        "n_holdout": len(clf_test),
        "holdout_start": str(clf_test["as_of_date"].min()),
        "holdout_end": str(clf_test["as_of_date"].max()),
        "positive_rate": _finite(y_true.mean()),
        "accuracy": _finite((preds == y_true).mean()),
        "baseline_accuracy": _finite(max(y_true.mean(), 1 - y_true.mean())),
        "auc": _finite(_binary_auc(y_true, probs)),
        "precision": _finite(precision),
        "recall": _finite(recall),
        "f1": _finite(f1),
        "log_loss": _finite(log_loss(y_true, probs, labels=[0, 1])),
        "brier": _finite(brier_score_loss(y_true, probs)),
        "roc_curve": roc,
        "calibration": calibration,
    }


def _time_split(
    df: pd.DataFrame, frac: float = 0.8, validation_period_days: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 3:
        raise ValueError("Need at least 3 labeled rows for a time-based split")
    if validation_period_days is not None:
        if validation_period_days < 1:
            raise ValueError("validation_period_days must be at least 1")
        dates = pd.to_datetime(df["as_of_date"])
        validation_start = dates.max().normalize() - pd.Timedelta(days=validation_period_days - 1)
        split_idx = int((dates < validation_start).sum())
    else:
        split_idx = int(len(df) * frac)
    split_idx = max(1, split_idx)
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


def walk_forward_evaluate(
    dataset: pd.DataFrame,
    n_folds: int = 5,
    xgb_params: dict[str, Any] | None = None,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Expanding-window backtest across multiple test windows, not just one 80/20 split.

    A single holdout can look better or worse than reality purely because of which season/period
    landed in the test slice. This retrains on an expanding window and evaluates on the next chunk
    each fold, so you can see whether classifier AUC is stable or swings a lot fold-to-fold.
    """
    ordered = dataset.sort_values("as_of_date").reset_index(drop=True)
    n = len(ordered)
    fold_edges = np.linspace(int(n * 0.5), n, n_folds + 1).astype(int)
    merged_xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    features = _feature_list(features)

    rows: list[dict[str, Any]] = []
    for fold in range(n_folds):
        train_end = fold_edges[fold]
        test_end = fold_edges[fold + 1]
        train = ordered.iloc[:train_end]
        test = ordered.iloc[train_end:test_end]
        if test.empty or train["will_train_tomorrow"].nunique() < 2:
            continue

        clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **merged_xgb_params)
        clf.fit(train[features], train["will_train_tomorrow"], sample_weight=_recency_weights(train["as_of_date"]))
        probs = clf.predict_proba(test[features])[:, 1]
        train_probs = clf.predict_proba(train[features])[:, 1]
        y_test = test["will_train_tomorrow"].to_numpy()
        preds = (probs >= 0.5).astype(int)

        rows.append(
            {
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "positive_rate": float(y_test.mean()),
                "train_auc": _binary_auc(train["will_train_tomorrow"].to_numpy(), train_probs),
                "auc": _binary_auc(y_test, probs),
                "accuracy": float((preds == y_test).mean()),
                "baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
            }
        )

    return pd.DataFrame(rows)


def train_and_save_models(
    dataset: pd.DataFrame,
    model_dir: Path,
    split_frac: float = 0.8,
    half_life_days: float = 180.0,
    xgb_params: dict[str, Any] | None = None,
    features: list[str] | None = None,
    validation_period_days: int | None = None,
) -> TrainedModels:
    ordered = dataset.sort_values("as_of_date").reset_index(drop=True)
    merged_xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
    features = _feature_list(features)

    clf_train, clf_test = _time_split(
        ordered, frac=split_frac, validation_period_days=validation_period_days
    )
    y_train = clf_train["will_train_tomorrow"]
    y_test = clf_test["will_train_tomorrow"]

    classifier = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **merged_xgb_params)
    classifier.fit(
        clf_train[features], y_train, sample_weight=_recency_weights(clf_train["as_of_date"], half_life_days)
    )

    probs = classifier.predict_proba(clf_test[features])[:, 1]
    preds = (probs >= 0.5).astype(int)
    validation_predictions = [
        {
            "date": str(target_date),
            "probability": float(probability),
            "actual_will_train": bool(actual),
        }
        for target_date, probability, actual in zip(
            clf_test["target_date"], probs, y_test.to_numpy(), strict=True
        )
    ]
    accuracy = float((preds == y_test.to_numpy()).mean())
    baseline_accuracy = float(max(y_test.mean(), 1 - y_test.mean()))
    auc = _binary_auc(y_test.to_numpy(), probs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test.to_numpy(), preds, average="binary", zero_division=0
    )
    logger.info("classifier_accuracy=%.3f (baseline=%.3f)", accuracy, baseline_accuracy)
    logger.info("classifier_auc=%s", f"{auc:.3f}" if not np.isnan(auc) else "nan")
    logger.info("classifier_precision=%.3f recall=%.3f f1=%.3f", precision, recall, f1)

    # Metrics above reflect the held-out split; refit on all rows so the saved model also learns
    # from the newest data instead of only ever seeing it as an evaluation set.
    logger.info("refitting classifier on full dataset (n=%d) before saving", len(ordered))
    classifier.fit(
        ordered[features],
        ordered["will_train_tomorrow"],
        sample_weight=_recency_weights(ordered["as_of_date"], half_life_days),
    )

    reg_rows = ordered[ordered["will_train_tomorrow"] == 1].copy() # only predict effort when there is an effort, Binary model runs first
    if len(reg_rows) < 3:
        raise ValueError("Need at least 3 positive-label rows to train the effort regressor")

    reg_train, reg_test = _time_split(
        reg_rows, frac=split_frac, validation_period_days=validation_period_days
    )
    yr_train = reg_train["next_day_relative_effort"]
    yr_test = reg_test["next_day_relative_effort"]

    regressor = xgb.XGBRegressor(
        objective="reg:absoluteerror",  # optimize MAE directly, robust to the long-tail effort outliers
        **merged_xgb_params,
    )
    regressor.fit(
        reg_train[features], yr_train, sample_weight=_recency_weights(reg_train["as_of_date"], half_life_days)
    )

    reg_preds = regressor.predict(reg_test[features])
    mae = float(np.mean(np.abs(reg_preds - yr_test.to_numpy())))
    logger.info("regressor_mae=%.3f", mae)
    metrics = {
        "classifier": _classifier_holdout_metrics(y_test.to_numpy(), probs, clf_train, clf_test),
        "regressor": {
            "n_train": len(reg_train),
            "n_holdout": len(reg_test),
            "mae": _finite(mae),
            "baseline_mae": _finite(np.mean(np.abs(yr_test.to_numpy() - np.median(yr_train.to_numpy())))),
        },
    }

    logger.info("refitting regressor on full positive-label dataset (n=%d) before saving", len(reg_rows))
    regressor.fit(
        reg_rows[features],
        reg_rows["next_day_relative_effort"],
        sample_weight=_recency_weights(reg_rows["as_of_date"], half_life_days),
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(model_dir / CLASSIFIER_MODEL_PATH))
    regressor.save_model(str(model_dir / REGRESSOR_MODEL_PATH))

    return TrainedModels(
        classifier=classifier,
        regressor=regressor,
        validation_predictions=validation_predictions,
        metrics=metrics,
    )


def load_models(model_dir: Path) -> TrainedModels:
    classifier = xgb.XGBClassifier()
    regressor = xgb.XGBRegressor()
    classifier.load_model(str(model_dir / CLASSIFIER_MODEL_PATH))
    regressor.load_model(str(model_dir / REGRESSOR_MODEL_PATH))
    return TrainedModels(classifier=classifier, regressor=regressor)


def evaluate_models(
    models: TrainedModels,
    dataset: pd.DataFrame,
    features: list[str] | None = None,
    validation_period_days: int | None = None,
    half_life_days: float = 180.0,
    xgb_params: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Score the held-out split with models fit on the train split only.

    `models` is refit on all rows when saved, so scoring it here would be in-sample; it is kept
    in the signature for notebook compatibility but unused.
    """
    ordered = dataset.sort_values("as_of_date").reset_index(drop=True)
    features = _feature_list(features)
    merged_xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    clf_train, clf_test = _time_split(ordered, validation_period_days=validation_period_days)
    classifier = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **merged_xgb_params)
    classifier.fit(
        clf_train[features],
        clf_train["will_train_tomorrow"],
        sample_weight=_recency_weights(clf_train["as_of_date"], half_life_days),
    )
    probs = classifier.predict_proba(clf_test[features])[:, 1]

    reg_rows = ordered[ordered["will_train_tomorrow"] == 1].copy()
    reg_train, reg_test = _time_split(reg_rows, validation_period_days=validation_period_days)
    regressor = xgb.XGBRegressor(objective="reg:absoluteerror", **merged_xgb_params)
    regressor.fit(
        reg_train[features],
        reg_train["next_day_relative_effort"],
        sample_weight=_recency_weights(reg_train["as_of_date"], half_life_days),
    )
    reg_preds = regressor.predict(reg_test[features])

    return {
        "y_test": clf_test["will_train_tomorrow"].to_numpy(),
        "probs": probs,
        "reg_true": reg_test["next_day_relative_effort"].to_numpy(),
        "reg_pred": reg_preds,
    }


def predict_tomorrow(
    models: TrainedModels, tomorrow_features: pd.DataFrame, features: list[str] | None = None
) -> dict[str, Any]:
    features = _feature_list(features)
    feature_frame = tomorrow_features[features]
    probability = float(models.classifier.predict_proba(feature_frame)[0, 1])
    will_train = probability >= 0.5
    predicted_effort = float(models.regressor.predict(feature_frame)[0]) if will_train else None
    return {
        "will_train": bool(will_train),
        "probability": probability,
        "predicted_effort": predicted_effort,
    }
