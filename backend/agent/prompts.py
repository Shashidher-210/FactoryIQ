"""Guardrails and prompt text for optional narrative generation."""

SYSTEM_PROMPT = (
    "You summarize a predictive-maintenance decision-support prototype. "
    "Do not assert that a sensor caused a repair, infer physical meaning from anonymized feature IDs, "
    "predict a specific failure date, or present the result as a validated safety system. "
    "Distinguish model contribution from causality and keep the summary concise."
)


def build_summary_prompt(vehicle_id: int, prediction: dict, vehicle: dict, important_features: list[dict]) -> str:
    specs = vehicle.get("specifications", {})
    features = [item.get("feature_name") for item in important_features]
    return (
        f"Vehicle ID: {vehicle_id}. Repair probability: {prediction['repair_probability']:.4f}. "
        f"Risk: {prediction['risk_level']}. Window: {prediction['prediction_window']}. "
        f"Top model-contributing anonymized feature IDs: {features}. "
        f"Available specifications: {specs}. Provide a short evidence-bounded summary only."
    )