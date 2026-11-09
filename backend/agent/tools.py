"""Small explicit tools used by the investigation graph."""

from __future__ import annotations

from backend.services.prediction_service import PredictionService
from backend.services.vehicle_service import VehicleService


def retrieve_vehicle_tool(vehicle_service: VehicleService, vehicle_id: int) -> dict:
    return vehicle_service.get_vehicle(vehicle_id)


def predict_repair_risk_tool(prediction_service: PredictionService, vehicle_id: int) -> dict:
    return prediction_service.predict_vehicle(vehicle_id).model_dump(mode="json")


def explain_prediction_tool(prediction_service: PredictionService, vehicle_id: int) -> list[dict]:
    return [item.model_dump(mode="json") for item in prediction_service.explain_vehicle(vehicle_id)]