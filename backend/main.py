"""FactoryIQ FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.agent.graph import build_investigation_graph
from backend.config import (
    CORS_ORIGINS,
    FEATURES_PATH,
    MODEL_DIR,
    SPECIFICATIONS_PATH,
    SPLIT_ASSIGNMENTS_PATH,
)
from backend.schemas import (
    FleetRiskResponse,
    HealthResponse,
    InvestigationRequest,
    InvestigationResponse,
    PredictionRequest,
    PredictionResponse,
    VehicleDetailResponse,
    VehiclesResponse,
)
from backend.services.prediction_service import PredictionService
from backend.services.vehicle_service import VehicleService


LOGGER = logging.getLogger("factoryiq.api")


def create_app(
    model_dir: Path = MODEL_DIR,
    features_path: Path = FEATURES_PATH,
    specifications_path: Path = SPECIFICATIONS_PATH,
    cors_origins: list[str] | None = None,
    split_assignments_path: Path = SPLIT_ASSIGNMENTS_PATH,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        try:
            vehicle_service = VehicleService(features_path, specifications_path)
            prediction_service = PredictionService(model_dir, vehicle_service, split_assignments_path)
            application.state.vehicle_service = vehicle_service
            application.state.prediction_service = prediction_service
            application.state.investigation_graph = build_investigation_graph(vehicle_service, prediction_service)
            application.state.model_error = None
        except Exception as error:
            LOGGER.exception("Backend model initialization failed")
            application.state.vehicle_service = None
            application.state.prediction_service = None
            application.state.investigation_graph = None
            application.state.model_error = str(error)
        yield

    application = FastAPI(
        title="FactoryIQ Predictive Maintenance API",
        version="1.0.0",
        description="Decision-support prototype for SCANIA Component X repair risk.",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins if cors_origins is not None else CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

    def require_service(request: Request) -> tuple[VehicleService, PredictionService]:
        vehicle_service = getattr(request.app.state, "vehicle_service", None)
        prediction_service = getattr(request.app.state, "prediction_service", None)
        if vehicle_service is None or prediction_service is None:
            raise HTTPException(status_code=503, detail="Model or vehicle data is not ready")
        return vehicle_service, prediction_service

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        prediction_service = getattr(request.app.state, "prediction_service", None)
        error = getattr(request.app.state, "model_error", None)
        ready = prediction_service is not None
        return HealthResponse(
            status="ready" if ready else "degraded",
            model_ready=ready,
            model_version=prediction_service.model_version if ready else None,
            detail=None if ready else error or "Model is not loaded",
        )

    @application.get("/vehicles", response_model=VehiclesResponse)
    def list_vehicles(
        request: Request,
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=500),
    ) -> VehiclesResponse:
        vehicle_service, prediction_service = require_service(request)
        total, records = vehicle_service.list_vehicles(
            offset,
            limit,
            prediction_service.eligible_vehicle_ids,
        )
        items = []
        for record in records:
            prediction = prediction_service.cached_prediction(record["vehicle_id"])
            items.append(
                {
                    **record,
                    "repair_probability": prediction.repair_probability,
                    "risk_level": prediction.risk_level,
                    "predicted_class": prediction.predicted_class,
                    "prediction_window": prediction.prediction_window,
                }
            )
        return VehiclesResponse(
            total=total,
            offset=offset,
            limit=limit,
            classification_threshold=prediction_service.classification_threshold,
            risk_band_thresholds={
                "low_max": prediction_service.low_max,
                "medium_max": prediction_service.medium_max,
            },
            risk_thresholds_note=prediction_service.risk_thresholds_note,
            items=items,
        )

    @application.get("/fleet/risk", response_model=FleetRiskResponse)
    def fleet_risk(request: Request) -> FleetRiskResponse:
        _, prediction_service = require_service(request)
        return prediction_service.fleet_risk_summary()

    @application.get("/vehicles/{vehicle_id}", response_model=VehicleDetailResponse)
    def get_vehicle(vehicle_id: int, request: Request) -> VehicleDetailResponse:
        vehicle_service, _ = require_service(request)
        try:
            prediction_service = request.app.state.prediction_service
            return VehicleDetailResponse(
                **vehicle_service.get_vehicle(vehicle_id),
                prediction=prediction_service.predict_vehicle(vehicle_id),
            )
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Vehicle not found") from error

    @application.post("/predict", response_model=PredictionResponse)
    def predict(body: PredictionRequest, request: Request) -> PredictionResponse:
        _, prediction_service = require_service(request)
        try:
            if body.vehicle_id is not None:
                return prediction_service.predict_vehicle(body.vehicle_id)
            return prediction_service.predict_payload(body.features or {})
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Vehicle not found") from error
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @application.post("/agent/investigate", response_model=InvestigationResponse)
    def investigate(body: InvestigationRequest, request: Request) -> InvestigationResponse:
        _, prediction_service = require_service(request)
        graph = getattr(request.app.state, "investigation_graph", None)
        try:
            result = graph.invoke({"vehicle_id": body.vehicle_id})
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Vehicle not found") from error
        except Exception as error:
            LOGGER.exception("Vehicle investigation failed")
            raise HTTPException(status_code=500, detail="Vehicle investigation failed") from error
        return InvestigationResponse(
            vehicle_id=body.vehicle_id,
            prediction=PredictionResponse(**result["prediction"]),
            investigation={
                "summary": result["summary"],
                "summary_source": result["summary_source"],
                "important_features": result["important_features"],
                "recommended_actions": result["recommended_actions"],
            },
            disclaimer=result["disclaimer"],
        )

    return application


app = create_app()