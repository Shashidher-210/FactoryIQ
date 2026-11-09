"""Stable request and response schemas for frontend consumers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded"]
    model_ready: bool
    model_version: str | None = None
    detail: str | None = None


class VehicleListItem(BaseModel):
    vehicle_id: int
    latest_time_step: int | None = None
    reading_count: int | None = None
    repair_probability: float = Field(ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High"]
    predicted_class: Literal["no_repair", "repair"]
    prediction_window: str


class VehiclesResponse(BaseModel):
    total: int
    offset: int
    limit: int
    classification_threshold: float = Field(ge=0, le=1)
    risk_band_thresholds: dict[str, float]
    risk_thresholds_note: str
    items: list[VehicleListItem]


class PredictionRequest(BaseModel):
    vehicle_id: int | None = Field(default=None, ge=0)
    features: dict[str, Any] | None = None

    @model_validator(mode="after")
    def require_one_input(self) -> "PredictionRequest":
        if (self.vehicle_id is None) == (self.features is None):
            raise ValueError("Provide exactly one of vehicle_id or features")
        if self.features is not None and not self.features:
            raise ValueError("features must contain at least one model input")
        return self


class RiskBandThresholds(BaseModel):
    low_max: float = Field(ge=0, le=1)
    medium_max: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    vehicle_id: int | None = None
    repair_probability: float = Field(ge=0, le=1)
    risk_level: Literal["Low", "Medium", "High"]
    predicted_class: Literal["no_repair", "repair"]
    classification_threshold: float = Field(ge=0, le=1)
    risk_band_thresholds: RiskBandThresholds
    prediction_window: str
    model_version: str
    prediction_timestamp: str


class VehicleDetailResponse(BaseModel):
    vehicle_id: int
    specifications: dict[str, str | None]
    feature_values: dict[str, float | None]
    prediction: PredictionResponse


class RiskCounts(BaseModel):
    Low: int
    Medium: int
    High: int


class PredictionSummary(BaseModel):
    evaluated_vehicle_count: int
    minimum_probability: float
    maximum_probability: float
    mean_probability: float
    median_probability: float
    percentiles: dict[str, float]
    at_or_above_classification_threshold: int
    risk_counts: RiskCounts


class FleetRiskResponse(BaseModel):
    population_scope: str
    classification_threshold: float = Field(ge=0, le=1)
    risk_band_thresholds: RiskBandThresholds
    risk_thresholds_note: str
    summary: PredictionSummary
    split_summaries: dict[str, PredictionSummary]
    top_vehicles: list[PredictionResponse]


class ImportantFeature(BaseModel):
    feature_name: str
    contribution: float
    contribution_scale: Literal["xgboost_raw_margin"] = "xgboost_raw_margin"


class InvestigationDetails(BaseModel):
    summary: str
    summary_source: Literal["rule_based", "llm"]
    important_features: list[ImportantFeature]
    recommended_actions: list[str]


class InvestigationResponse(BaseModel):
    vehicle_id: int
    prediction: PredictionResponse
    investigation: InvestigationDetails
    disclaimer: str = "Decision-support prototype; not a validated maintenance or safety system."


class InvestigationRequest(BaseModel):
    vehicle_id: int = Field(ge=0)