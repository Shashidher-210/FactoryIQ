"""Evaluate the saved model once on the held-out vehicle test split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


LOGGER = logging.getLogger("factoryiq.evaluate")
DATASET_PATH = Path("data/modeling_dataset.csv")
SPLITS_PATH = Path("data/split_assignments.csv")
MODEL_DIR = Path("models")
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"


def calculate_metrics(targets: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    return {
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc_average_precision": float(average_precision_score(targets, probabilities)),
        "confusion_matrix_labels_0_1": confusion_matrix(targets, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=[0, 1],
            target_names=["no_repair", "repair"],
            output_dict=True,
            zero_division=0,
        ),
        "positive_class_prevalence": float(np.mean(targets)),
        "accuracy": float(accuracy_score(targets, predictions)),
    }


def evaluate_model(
    dataset_path: Path = DATASET_PATH,
    splits_path: Path = SPLITS_PATH,
    model_dir: Path = MODEL_DIR,
    output_path: Path = METRICS_PATH,
) -> dict[str, object]:
    dataset = pd.read_csv(dataset_path)
    splits = pd.read_csv(splits_path)
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(model_dir / "xgb_pipeline.joblib")
    test_ids = splits.loc[splits["split"].eq("test"), "vehicle_id"]
    train_ids = splits.loc[splits["split"].eq("train"), "vehicle_id"]
    if test_ids.isin(train_ids).any():
        raise ValueError("Training and test vehicle IDs overlap")
    test_data = dataset.loc[dataset["vehicle_id"].isin(test_ids)].copy()
    if len(test_data) != len(test_ids) or test_data.empty:
        raise ValueError("The saved test split does not match the modeling dataset")

    feature_columns = metadata["feature_columns"]
    targets = test_data["target"].to_numpy(dtype=np.int8)
    probabilities = model.predict_proba(test_data[feature_columns])[:, 1]
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Model produced invalid probabilities")
    threshold = float(metadata["selected_validation_threshold"])
    predictions = (probabilities >= threshold).astype(np.int8)
    model_metrics = calculate_metrics(targets, predictions, probabilities)

    majority = DummyClassifier(strategy="most_frequent")
    train_data = dataset.loc[dataset["vehicle_id"].isin(train_ids)]
    majority.fit(np.zeros((len(train_data), 1)), train_data["target"])
    baseline_predictions = majority.predict(np.zeros((len(test_data), 1)))
    baseline_probabilities = np.full(len(test_data), float(train_data["target"].mean()))
    baseline_metrics = calculate_metrics(targets, baseline_predictions, baseline_probabilities)

    results: dict[str, object] = {
        "model_version": metadata["model_version"],
        "test_split_size": len(test_data),
        "threshold_source": "validation split only; max-F1 operating point",
        "classification_threshold": threshold,
        "positive_class_prevalence": float(np.mean(targets)),
        "test": model_metrics,
        "majority_baseline": baseline_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Held-out test metrics: %s", json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--output", type=Path, default=METRICS_PATH)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    evaluate_model(args.dataset, args.splits, args.model_dir, args.output)


if __name__ == "__main__":
    main()