"""Load persisted model artifacts and perform validated inference."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from backend.config import SPLIT_ASSIGNMENTS_PATH
from backend.schemas import FleetRiskResponse, ImportantFeature, PredictionResponse
from backend.services.vehicle_service import VehicleService


class PredictionService:
    def __init__(
        self,
        model_dir: Path,
        vehicle_service: VehicleService,
        split_assignments_path: Path = SPLIT_ASSIGNMENTS_PATH,
    ) -> None:
        model_path = model_dir / "xgb_pipeline.joblib"
        metadata_path = model_dir / "model_metadata.json"
        names_path = model_dir / "feature_names.json"
        metrics_path = model_dir / "evaluation_metrics.json"
        for path in (model_path, metadata_path, names_path, metrics_path, split_assignments_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required model artifact is missing: {path}")
        self.model = joblib.load(model_path)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        evaluation_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.feature_names = json.loads(names_path.read_text(encoding="utf-8"))["transformed_features"]
        self.feature_columns: list[str] = self.metadata["feature_columns"]
        self.numeric_columns: set[str] = set(self.metadata["numeric_columns"])
        self.categorical_columns: set[str] = set(self.metadata["categorical_columns"])
        self.vehicle_service = vehicle_service
        self.classification_threshold = float(self.metadata["selected_validation_threshold"])
        if not isclose(
            self.classification_threshold,
            float(evaluation_metrics["classification_threshold"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Saved model and evaluation classification thresholds do not match")
        risk_thresholds = self.metadata["risk_thresholds"]
        self.low_max = float(risk_thresholds["low_max"])
        self.medium_max = float(risk_thresholds["medium_max"])
        self.risk_thresholds_note = str(self.metadata["risk_thresholds_note"])
        if set(self.feature_columns) != self.numeric_columns | self.categorical_columns:
            raise ValueError("Model metadata input feature declarations do not match")
        if self.metadata.get("prediction_cutoff") != 100:
            raise ValueError("Model artifact does not use the documented step-100 cutoff")
        if not 0 <= self.low_max < self.medium_max <= 1:
            raise ValueError("Model risk-band thresholds are invalid")

        assignments = pd.read_csv(split_assignments_path)
        if not {"vehicle_id", "split"}.issubset(assignments.columns):
            raise ValueError("Split assignments must contain vehicle_id and split")
        if assignments["vehicle_id"].duplicated().any() or not assignments["split"].isin(
            ["train", "validation", "test"]
        ).all():
            raise ValueError("Split assignments contain duplicate IDs or an invalid split")
        self._split_by_vehicle = {
            int(vehicle_id): str(split)
            for vehicle_id, split in assignments[["vehicle_id", "split"]].itertuples(index=False, name=None)
        }
        self.eligible_vehicle_ids = sorted(self._split_by_vehicle)
        self._predictions = self._score_eligible_cohort()
        self._fleet_risk_summary = self._build_fleet_risk_summary()

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])

    def _validate_payload(self, payload: dict[str, Any]) -> pd.DataFrame:
        unknown = set(payload) - set(self.feature_columns)
        if unknown:
            raise ValueError(f"Unknown model feature(s): {', '.join(sorted(unknown))}")
        values: dict[str, Any] = {}
        for name in self.feature_columns:
            value = payload.get(name)
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                value = None
            if name in self.numeric_columns and value is not None:
                if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)) or not np.isfinite(value):
                    raise ValueError(f"{name} must be a finite number or null")
                value = float(value)
            elif name in self.categorical_columns and value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or null")
            values[name] = value
        return pd.DataFrame([values], columns=self.feature_columns)

    def predict_vehicle(self, vehicle_id: int) -> PredictionResponse:
        cached_prediction = self._predictions.get(vehicle_id)
        if cached_prediction is not None:
            return cached_prediction
        payload = self.vehicle_service.get_model_record(vehicle_id, self.feature_columns)
        return self._predict_frame(self._validate_payload(payload), vehicle_id)

    def cached_prediction(self, vehicle_id: int) -> PredictionResponse:
        return self._predictions[vehicle_id]

    def fleet_risk_summary(self) -> FleetRiskResponse:
        return self._fleet_risk_summary

    def predict_payload(self, payload: dict[str, Any]) -> PredictionResponse:
        return self._predict_frame(self._validate_payload(payload), None)

    def _score_eligible_cohort(self) -> dict[int, PredictionResponse]:
        frame = self.vehicle_service.model_frame(self.eligible_vehicle_ids, self.feature_columns)
        probabilities = self.model.predict_proba(frame)[:, 1]
        if len(probabilities) != len(self.eligible_vehicle_ids):
            raise RuntimeError("The model returned an incomplete fleet prediction batch")
        timestamp = datetime.now(timezone.utc).isoformat()
        predictions: dict[int, PredictionResponse] = {}
        for vehicle_id, probability in zip(self.eligible_vehicle_ids, probabilities, strict=True):
            value = float(probability)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise RuntimeError(f"The model returned an invalid probability for vehicle {vehicle_id}")
            predictions[vehicle_id] = self._prediction_response(value, vehicle_id, timestamp)
        return predictions

    def _summary(self, predictions: list[PredictionResponse]) -> dict[str, Any]:
        probabilities = np.asarray([item.repair_probability for item in predictions], dtype=float)
        risk_counts = {
            level: sum(item.risk_level == level for item in predictions)
            for level in ("Low", "Medium", "High")
        }
        return {
            "evaluated_vehicle_count": len(predictions),
            "minimum_probability": float(probabilities.min()),
            "maximum_probability": float(probabilities.max()),
            "mean_probability": float(probabilities.mean()),
            "median_probability": float(np.median(probabilities)),
            "percentiles": {
                f"p{percentile}": float(np.percentile(probabilities, percentile))
                for percentile in (1, 5, 10, 25, 50, 75, 90, 95, 99)
            },
            "at_or_above_classification_threshold": sum(
                item.repair_probability >= self.classification_threshold for item in predictions
            ),
            "risk_counts": risk_counts,
        }

    def _build_fleet_risk_summary(self) -> FleetRiskResponse:
        ordered = list(self._predictions.values())
        split_summaries = {
            split: self._summary(
                [
                    self._predictions[vehicle_id]
                    for vehicle_id, vehicle_split in self._split_by_vehicle.items()
                    if vehicle_split == split
                ]
            )
            for split in ("train", "validation", "test")
        }
        return FleetRiskResponse(
            population_scope=(
                f"Vehicles in data/split_assignments.csv: the eligible, labeled {len(self.eligible_vehicle_ids):,}-vehicle "
                "modeling cohort scored from step-100 cutoff-safe features; includes train, "
                "validation, and test splits. Training predictions are in-sample, not unbiased evaluation."
            ),
            classification_threshold=self.classification_threshold,
            risk_band_thresholds={"low_max": self.low_max, "medium_max": self.medium_max},
            risk_thresholds_note=self.risk_thresholds_note,
            summary=self._summary(ordered),
            split_summaries=split_summaries,
            top_vehicles=sorted(ordered, key=lambda item: item.repair_probability, reverse=True)[:10],
        )

    def _predict_frame(self, frame: pd.DataFrame, vehicle_id: int | None) -> PredictionResponse:
        probability = float(self.model.predict_proba(frame)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("The model returned an invalid probability")
        return self._prediction_response(probability, vehicle_id, datetime.now(timezone.utc).isoformat())

    def _prediction_response(
        self,
        probability: float,
        vehicle_id: int | None,
        prediction_timestamp: str,
    ) -> PredictionResponse:
        if probability < self.low_max:
            risk_level = "Low"
        elif probability < self.medium_max:
            risk_level = "Medium"
        else:
            risk_level = "High"
        return PredictionResponse(
            vehicle_id=vehicle_id,
            repair_probability=probability,
            risk_level=risk_level,
            predicted_class="repair" if probability >= self.classification_threshold else "no_repair",
            classification_threshold=self.classification_threshold,
            risk_band_thresholds={"low_max": self.low_max, "medium_max": self.medium_max},
            prediction_window=self.metadata["prediction_window"],
            model_version=self.model_version,
            prediction_timestamp=prediction_timestamp,
        )

    def explain_vehicle(self, vehicle_id: int, top_n: int = 8) -> list[ImportantFeature]:
        payload = self.vehicle_service.get_model_record(vehicle_id, self.feature_columns)
        return self.explain_payload(payload, top_n)

    def explain_payload(self, payload: dict[str, Any], top_n: int = 8) -> list[ImportantFeature]:
        frame = self._validate_payload(payload)
        transformed = self.model.named_steps["preprocessor"].transform(frame)
        contributions = self.model.named_steps["classifier"].get_booster().predict(
            xgb.DMatrix(transformed), pred_contribs=True
        )[0]
        pairs = list(zip(self.feature_names, contributions[:-1], strict=True))
        pairs.sort(key=lambda item: abs(float(item[1])), reverse=True)
        return [
            ImportantFeature(feature_name=name, contribution=float(value))
            for name, value in pairs[:top_n]
        ]