"""Read cutoff-safe feature and vehicle specification records."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class VehicleService:
    def __init__(self, features_path: Path, specifications_path: Path) -> None:
        features = pd.read_csv(features_path)
        specifications = pd.read_csv(specifications_path)
        if features["vehicle_id"].duplicated().any() or specifications["vehicle_id"].duplicated().any():
            raise ValueError("Vehicle feature and specification IDs must be unique")
        self._records = features.merge(
            specifications,
            on="vehicle_id",
            how="left",
            validate="one_to_one",
        ).sort_values("vehicle_id").reset_index(drop=True)
        if self._records[[column for column in specifications.columns if column != "vehicle_id"]].isna().all(axis=None):
            raise ValueError("Vehicle specification data is empty")
        self._positions = {int(vehicle_id): row for row, vehicle_id in enumerate(self._records["vehicle_id"])}
        self.specification_columns = [column for column in specifications.columns if column != "vehicle_id"]
        self.feature_columns = [
            column
            for column in features.columns
            if column not in {"vehicle_id", *self.specification_columns}
        ]

    def _record(self, vehicle_id: int) -> pd.Series:
        position = self._positions.get(vehicle_id)
        if position is None:
            raise KeyError(vehicle_id)
        return self._records.iloc[position]

    def get_model_record(self, vehicle_id: int, model_features: list[str]) -> dict[str, Any]:
        record = self._record(vehicle_id)
        return {name: record[name] for name in model_features}

    def get_vehicle(self, vehicle_id: int) -> dict[str, Any]:
        record = self._record(vehicle_id)
        feature_values: dict[str, float | None] = {}
        for name in self.feature_columns:
            value = record[name]
            feature_values[name] = None if pd.isna(value) else float(value)
        specifications = {
            name: None if pd.isna(record[name]) else str(record[name])
            for name in self.specification_columns
        }
        return {
            "vehicle_id": vehicle_id,
            "specifications": specifications,
            "feature_values": feature_values,
        }

    def list_vehicles(
        self,
        offset: int,
        limit: int,
        vehicle_ids: list[int] | None = None,
    ) -> tuple[int, list[dict[str, int | None]]]:
        records = (
            self._records
            if vehicle_ids is None
            else self._records.loc[self._records["vehicle_id"].isin(vehicle_ids)]
        )
        page = records.iloc[offset : offset + limit]
        items = [
            {
                "vehicle_id": int(record.vehicle_id),
                "latest_time_step": int(record.latest_time_step) if pd.notna(record.latest_time_step) else None,
                "reading_count": int(record.reading_count) if pd.notna(record.reading_count) else None,
            }
            for record in page.itertuples(index=False)
        ]
        return len(records), items

    def model_frame(self, vehicle_ids: list[int], model_features: list[str]) -> pd.DataFrame:
        positions = [self._positions[vehicle_id] for vehicle_id in vehicle_ids]
        return self._records.iloc[positions][model_features].reset_index(drop=True)

    def feature_frame(self, vehicle_id: int, model_features: list[str]) -> pd.DataFrame:
        return self.model_frame([vehicle_id], model_features)