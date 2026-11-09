"""Train and select an XGBoost repair-risk classifier."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


LOGGER = logging.getLogger("factoryiq.train")
ID_COLUMN = "vehicle_id"
TARGET_COLUMN = "target"
RANDOM_STATE = 42
DATASET_PATH = Path("data/modeling_dataset.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "xgb_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"
SPLITS_PATH = Path("data/split_assignments.csv")


def build_pipeline(numeric_columns: list[str], categorical_columns: list[str], scale_pos_weight: float) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median", keep_empty_features=True),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                                keep_empty_features=True,
                            ),
                        ),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=350,
        max_depth=4,
        learning_rate=0.04,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        n_jobs=4,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def choose_f1_threshold(targets: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(np.concatenate(([0.0], probabilities, [1.0])))
    scores = np.asarray([f1_score(targets, probabilities >= value, zero_division=0) for value in thresholds])
    best = np.flatnonzero(scores == scores.max())
    selected_index = best[-1]
    return float(thresholds[selected_index]), float(scores[selected_index])


def train_model(dataset_path: Path = DATASET_PATH, model_dir: Path = MODEL_DIR) -> dict[str, object]:
    data = pd.read_csv(dataset_path)
    if data[ID_COLUMN].duplicated().any():
        raise ValueError("Modeling data must contain one row per vehicle")
    if not data[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Training target must contain only known binary labels")

    excluded = {ID_COLUMN, TARGET_COLUMN, "length_of_study_time_step", "in_study_repair"}
    feature_columns = [column for column in data.columns if column not in excluded]
    if excluded.intersection(feature_columns):
        raise RuntimeError("Identifier or repair outcome leaked into model inputs")
    categorical_columns = [column for column in feature_columns if column.startswith("Spec_")]
    numeric_columns = [column for column in feature_columns if column not in categorical_columns]
    if not numeric_columns or len(categorical_columns) != 8:
        raise ValueError("Expected numeric operational features and Spec_0 through Spec_7")
    if not data[numeric_columns].apply(pd.to_numeric, errors="coerce").notna().any().any():
        raise ValueError("No valid numeric model features were found")

    targets = data[TARGET_COLUMN].astype("int8")
    train_ids, remaining_ids = train_test_split(
        data[ID_COLUMN],
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=targets,
    )
    remaining_targets = data.set_index(ID_COLUMN).loc[remaining_ids, TARGET_COLUMN]
    validation_ids, test_ids = train_test_split(
        remaining_ids,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=remaining_targets,
    )
    train_mask = data[ID_COLUMN].isin(train_ids)
    validation_mask = data[ID_COLUMN].isin(validation_ids)
    train_x = data.loc[train_mask, feature_columns]
    train_y = data.loc[train_mask, TARGET_COLUMN].astype("int8")
    validation_x = data.loc[validation_mask, feature_columns]
    validation_y = data.loc[validation_mask, TARGET_COLUMN].astype("int8").to_numpy()

    positives = int(train_y.sum())
    negatives = len(train_y) - positives
    imbalance_ratio = negatives / positives
    candidate_results: list[dict[str, object]] = []
    candidates: list[tuple[float, Pipeline]] = []
    for candidate_name, class_weight in (("unweighted", 1.0), ("balanced", imbalance_ratio)):
        pipeline = build_pipeline(numeric_columns, categorical_columns, class_weight)
        LOGGER.info("Fitting %s XGBoost candidate", candidate_name)
        pipeline.fit(train_x, train_y)
        validation_probabilities = pipeline.predict_proba(validation_x)[:, 1]
        threshold, validation_f1 = choose_f1_threshold(validation_y, validation_probabilities)
        result = {
            "name": candidate_name,
            "scale_pos_weight": class_weight,
            "validation_average_precision": float(average_precision_score(validation_y, validation_probabilities)),
            "validation_roc_auc": float(roc_auc_score(validation_y, validation_probabilities)),
            "validation_f1_at_selected_threshold": validation_f1,
            "validation_threshold": threshold,
        }
        candidate_results.append(result)
        candidates.append((class_weight, pipeline))
        LOGGER.info("%s validation results: %s", candidate_name, result)

    selected_index = max(
        range(len(candidate_results)),
        key=lambda index: (
            candidate_results[index]["validation_average_precision"],
            candidate_results[index]["validation_f1_at_selected_threshold"],
        ),
    )
    selected_model = candidates[selected_index][1]
    selected_result = candidate_results[selected_index]
    transformed_names = selected_model.named_steps["preprocessor"].get_feature_names_out().tolist()
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, model_dir / MODEL_PATH.name)
    metadata: dict[str, object] = {
        "model_version": "factoryiq-xgb-1.0.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "Component X repair observed after time step 100 and by time step 200",
        "prediction_cutoff": 100,
        "prediction_horizon_steps": 100,
        "prediction_window": "time steps 100 to 200 (repair time >100 and <=200)",
        "risk_thresholds": {"low_max": 0.20, "medium_max": 0.50, "high_min": 0.50},
        "risk_thresholds_note": "Demonstration thresholds; not calibrated clinical or operational cutoffs.",
        "selected_validation_threshold": selected_result["validation_threshold"],
        "selected_candidate": selected_result["name"],
        "candidate_validation_results": candidate_results,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "excluded_columns": sorted(excluded),
        "split_seed": RANDOM_STATE,
        "split_sizes": {
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(data[ID_COLUMN].isin(test_ids).sum()),
        },
        "training_positive_prevalence": float(train_y.mean()),
        "versions": {"python": "3.12", "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    (model_dir / METADATA_PATH.name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    feature_names = {"input_features": feature_columns, "transformed_features": transformed_names}
    (model_dir / FEATURE_NAMES_PATH.name).write_text(json.dumps(feature_names, indent=2), encoding="utf-8")

    split_assignments = pd.concat(
        [
            pd.DataFrame({ID_COLUMN: train_ids, "split": "train"}),
            pd.DataFrame({ID_COLUMN: validation_ids, "split": "validation"}),
            pd.DataFrame({ID_COLUMN: test_ids, "split": "test"}),
        ],
        ignore_index=True,
    )
    split_assignments.to_csv(SPLITS_PATH, index=False)
    LOGGER.info("Saved selected model to %s; validation-selected threshold %.4f", model_dir / MODEL_PATH.name, metadata["selected_validation_threshold"])
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_model(args.dataset, args.model_dir)


if __name__ == "__main__":
    main()