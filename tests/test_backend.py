import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import create_app


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def get_vehicle_payload(client: TestClient) -> tuple[int, dict]:
    item = client.get("/vehicles?offset=0&limit=1").json()["items"][0]
    vehicle_id = item["vehicle_id"]
    detail = client.get(f"/vehicles/{vehicle_id}").json()
    features = {**detail["feature_values"], **detail["specifications"]}
    return vehicle_id, features


def test_health_and_swagger(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_ready"] is True
    assert client.get("/docs").status_code == 200


def test_paginated_vehicle_retrieval(client):
    response = client.get("/vehicles?offset=0&limit=3")
    assert response.status_code == 200
    assert response.json()["total"] == 14809
    assert len(response.json()["items"]) == 3
    assert all(0 <= item["repair_probability"] <= 1 for item in response.json()["items"])


def test_vehicle_detail_has_features_without_outcomes(client):
    vehicle_id, _ = get_vehicle_payload(client)
    response = client.get(f"/vehicles/{vehicle_id}")
    assert response.status_code == 200
    body = response.json()
    assert len(body["feature_values"]) == 737
    assert "target" not in body and "in_study_repair" not in body
    assert 0 <= body["prediction"]["repair_probability"] <= 1


def test_prediction_by_id_and_unseen_category_payload(client):
    vehicle_id, features = get_vehicle_payload(client)
    by_id = client.post("/predict", json={"vehicle_id": vehicle_id})
    assert by_id.status_code == 200
    probability = by_id.json()["repair_probability"]
    assert 0 <= probability <= 1
    features["Spec_0"] = "__unseen_category__"
    by_payload = client.post("/predict", json={"features": features})
    assert by_payload.status_code == 200
    assert 0 <= by_payload.json()["repair_probability"] <= 1


def test_fleet_risk_summary_covers_eligible_cohort_and_split_assignments(client):
    response = client.get("/fleet/risk")
    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["evaluated_vehicle_count"] == 14809
    assert body["summary"]["at_or_above_classification_threshold"] == 1009
    assert body["summary"]["risk_counts"] == {"Low": 13831, "Medium": 734, "High": 244}
    assert body["summary"]["minimum_probability"] == pytest.approx(0.0007315915)
    assert body["summary"]["maximum_probability"] == pytest.approx(0.9056275487)
    assert body["summary"]["mean_probability"] == pytest.approx(0.0592100288)
    assert body["summary"]["median_probability"] == pytest.approx(0.0238878354)
    assert body["summary"]["percentiles"]["p95"] == pytest.approx(0.2559412181)
    assert {
        split: summary["evaluated_vehicle_count"]
        for split, summary in body["split_summaries"].items()
    } == {"train": 10366, "validation": 2221, "test": 2222}
    assert body["split_summaries"]["train"]["at_or_above_classification_threshold"] == 743
    assert body["split_summaries"]["validation"]["at_or_above_classification_threshold"] == 131
    assert body["split_summaries"]["test"]["at_or_above_classification_threshold"] == 135
    assert body["classification_threshold"] == pytest.approx(0.19462941586971283)
    assert body["top_vehicles"][0]["vehicle_id"] == 10218


def test_api_vehicle_predictions_match_direct_saved_model_predictions(client):
    vehicle_service = client.app.state.vehicle_service
    prediction_service = client.app.state.prediction_service
    summary = client.get("/fleet/risk").json()
    for vehicle_id in (10218, 13758, summary["top_vehicles"][2]["vehicle_id"]):
        frame = vehicle_service.model_frame([vehicle_id], prediction_service.feature_columns)
        direct_probability = float(prediction_service.model.predict_proba(frame)[0, 1])
        response = client.post("/predict", json={"vehicle_id": vehicle_id})
        assert response.status_code == 200
        assert response.json()["repair_probability"] == pytest.approx(direct_probability, abs=1e-12)


def test_invalid_vehicle_and_payloads(client):
    assert client.get("/vehicles/99999999").status_code == 404
    assert client.post("/predict", json={"vehicle_id": -1}).status_code == 422
    assert client.post("/predict", json={"vehicle_id": 0, "features": {}}).status_code == 422
    assert client.post("/predict", json={"features": {"not_a_model_feature": 1}}).status_code == 422


def test_investigation_uses_rule_based_summary_without_credentials(client, monkeypatch):
    import backend.agent.graph as graph_module

    monkeypatch.setattr(graph_module, "LLM_API_KEY", None)
    vehicle_id, _ = get_vehicle_payload(client)
    response = client.post("/agent/investigate", json={"vehicle_id": vehicle_id})
    assert response.status_code == 200
    body = response.json()
    assert body["investigation"]["summary_source"] == "rule_based"
    assert len(body["investigation"]["important_features"]) > 0
    assert body["disclaimer"].startswith("Decision-support prototype")


def test_missing_model_artifacts_report_degraded_health(tmp_path):
    features_path = tmp_path / "features.csv"
    specifications_path = tmp_path / "specifications.csv"
    pd.DataFrame(
        [{"vehicle_id": 1, "reading_count": 1, "latest_time_step": 100, "measure__mean": 1.0}]
    ).to_csv(features_path, index=False)
    pd.DataFrame(
        [{"vehicle_id": 1, **{f"Spec_{index}": "category" for index in range(8)}}]
    ).to_csv(specifications_path, index=False)

    app = create_app(
        model_dir=tmp_path / "missing_models",
        features_path=features_path,
        specifications_path=specifications_path,
    )
    with TestClient(app) as test_client:
        health = test_client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is False
        assert test_client.post("/predict", json={"vehicle_id": 1}).status_code == 503