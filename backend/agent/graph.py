"""LangGraph workflow for a single-vehicle risk investigation."""

from __future__ import annotations

import logging
from typing import TypedDict

import httpx
from langgraph.graph import END, START, StateGraph

from backend.agent.prompts import SYSTEM_PROMPT, build_summary_prompt
from backend.agent.tools import explain_prediction_tool, predict_repair_risk_tool, retrieve_vehicle_tool
from backend.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from backend.services.prediction_service import PredictionService
from backend.services.vehicle_service import VehicleService


LOGGER = logging.getLogger("factoryiq.agent")
DISCLAIMER = "Decision-support prototype; not a validated maintenance or safety system."


class InvestigationState(TypedDict, total=False):
    vehicle_id: int
    vehicle: dict
    prediction: dict
    important_features: list[dict]
    summary: str
    summary_source: str
    recommended_actions: list[str]
    disclaimer: str


def _rule_based_summary(prediction: dict) -> str:
    risk = prediction["risk_level"].lower()
    probability = prediction["repair_probability"]
    return (
        f"The model estimates {risk} risk ({probability:.1%} probability) of an observed Component X repair "
        f"within {prediction['prediction_window']}. This is a risk ranking, not a failure-date estimate."
    )


def _optional_llm_summary(vehicle_id: int, prediction: dict, vehicle: dict, features: list[dict]) -> str | None:
    if not LLM_API_KEY:
        return None
    try:
        response = httpx.post(
            f"{LLM_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_summary_prompt(vehicle_id, prediction, vehicle, features)},
                ],
                "temperature": 0.2,
            },
            timeout=15,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"]).strip()
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
        LOGGER.warning("Optional LLM summary failed; using deterministic summary: %s", error)
        return None


def build_investigation_graph(
    vehicle_service: VehicleService,
    prediction_service: PredictionService,
):
    def retrieve_vehicle(state: InvestigationState) -> dict:
        return {"vehicle": retrieve_vehicle_tool(vehicle_service, state["vehicle_id"])}

    def predict_risk(state: InvestigationState) -> dict:
        return {"prediction": predict_repair_risk_tool(prediction_service, state["vehicle_id"])}

    def explain_risk(state: InvestigationState) -> dict:
        return {"important_features": explain_prediction_tool(prediction_service, state["vehicle_id"])}

    def compose_assessment(state: InvestigationState) -> dict:
        prediction = state["prediction"]
        actions = [
            "Review the vehicle's recent operational history using the anonymized feature identifiers.",
            "Compare its measurements with similar vehicles before drawing conclusions.",
            "Consider further diagnostic inspection according to established maintenance procedures.",
        ]
        llm_summary = _optional_llm_summary(
            state["vehicle_id"], prediction, state["vehicle"], state["important_features"]
        )
        return {
            "summary": llm_summary or _rule_based_summary(prediction),
            "summary_source": "llm" if llm_summary else "rule_based",
            "recommended_actions": actions,
            "disclaimer": DISCLAIMER,
        }

    workflow = StateGraph(InvestigationState)
    workflow.add_node("retrieve_vehicle", retrieve_vehicle)
    workflow.add_node("predict_risk", predict_risk)
    workflow.add_node("explain_risk", explain_risk)
    workflow.add_node("compose_assessment", compose_assessment)
    workflow.add_edge(START, "retrieve_vehicle")
    workflow.add_edge("retrieve_vehicle", "predict_risk")
    workflow.add_edge("predict_risk", "explain_risk")
    workflow.add_edge("explain_risk", "compose_assessment")
    workflow.add_edge("compose_assessment", END)
    return workflow.compile()