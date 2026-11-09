# FactoryIQ

FactoryIQ is a decision-support prototype that estimates whether a SCANIA vehicle will have an observed Component X repair in the 100-step window after a step-100 prediction cutoff. It combines historical operational summaries, vehicle specifications, an XGBoost classifier, and a FastAPI investigation endpoint.

## Data And Target

Operational identifiers such as `171_0` and `397_14` are anonymized. The project keeps those identifiers unchanged and does not assign physical meanings to them. Features are aggregated only from readings with `time_step <= 100`; the event-time and study-end columns are used only to construct labels and are never exposed to the inference service.

The target is 1 when a repair is observed after step 100 and through step 200. It is 0 only when the complete window is observed without a repair in that interval. Vehicles repaired by step 100 and unrepaired vehicles censored before step 200 are excluded. The saved eligible cohort contains 14,809 vehicles: 910 positive and 13,899 negative.

## Environment Setup

Use the existing project environment where possible. These pinned dependencies retain the working pandas 2.3.3 version.

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional settings can be copied from `.env.example` into `.env`. No LLM key is needed for API or agent operation. When no key is configured, investigations use deterministic rule-based summaries and make no external LLM request.

## Train The Model

Run the pipeline from the project root:

```powershell
python build_features.py
python prepare_training_data.py
python train_model.py
python evaluate_model.py
pytest -q
```

`build_features.py` reads the operational file in chunks and writes `data/train_features_100step.csv`. It generates seven summaries per operational measurement (mean, sample standard deviation, min, max, latest reading at the vehicle's latest available time step, valid count, and missing fraction), plus reading count and latest time step. Statistics combine across chunks without reading post-cutoff measurements.

`prepare_training_data.py` performs one-to-one joins and drops unknown labels. `train_model.py` makes reproducible, stratified, vehicle-level 70/15/15 train/validation/test splits. Imputation and one-hot encoding are fit only on training data. Weighted and unweighted XGBoost candidates are compared on validation average precision; the classification threshold is selected on validation F1. The test set is evaluated only by `evaluate_model.py` after those choices are fixed.

Artifacts are written to `models/`: `xgb_pipeline.joblib`, `model_metadata.json`, `feature_names.json`, and `evaluation_metrics.json`. The split membership is saved to `data/split_assignments.csv` for reproducibility. The evaluation JSON also includes a majority-class baseline.

## Recorded Evaluation

The current saved run used 2,222 held-out test vehicles with 6.17% positive prevalence. At the validation-selected threshold of 0.1946, the XGBoost model achieved:

| Metric | Held-out result |
| --- | ---: |
| Precision | 0.326 |
| Recall | 0.321 |
| F1 | 0.324 |
| ROC-AUC | 0.793 |
| Average precision (PR-AUC) | 0.261 |
| Confusion matrix (TN, FP, FN, TP) | 1,994; 91; 93; 44 |

The majority baseline has 0.938 accuracy because the cohort is imbalanced, but it predicts no positive repairs and has 0 recall. These results are not evidence of production readiness. Thresholds for Low, Medium, and High risk (below 0.20, 0.20 to below 0.50, and 0.50 or higher) are demonstration thresholds, not calibrated operational cutoffs.

## Start The API

```powershell
uvicorn backend.main:app --reload
```

The interactive API reference is available at `http://127.0.0.1:8000/docs`. At startup, the backend loads the trained model and cutoff-safe feature/specification files once. It does not load repair outcomes or train the model. If artifacts are missing, `/health` reports `model_ready: false`, and model-dependent endpoints return HTTP 503.

## API Endpoints

- `GET /health` reports readiness and model version.
- `GET /fleet/risk` returns cached probability statistics, demonstration-band counts, split summaries, and the ten highest model probabilities for the eligible cohort.
- `GET /vehicles?offset=0&limit=50` returns eligible cohort vehicles with cached probabilities, risk bands, binary-threshold classes, and the prediction window.
- `GET /vehicles/{vehicle_id}` returns specifications, cutoff-safe historical feature summaries, and a model prediction.
- `POST /predict` accepts exactly one of a vehicle ID or a feature payload.
- `POST /agent/investigate` runs a LangGraph flow that retrieves the vehicle, predicts risk, obtains XGBoost contributions, and returns a structured assessment.

Vehicle-ID prediction request:

```json
{"vehicle_id": 0}
```

Feature-payload request uses the model input names from `models/feature_names.json`; omitted values are treated as missing and imputed by the saved pipeline. Unknown names and invalid numeric values are rejected.

```json
{"features": {"171_0__mean": 12.5, "Spec_0": "example-category"}}
```

Predictions contain a probability, demonstration risk category, model version, defined prediction window, and timestamp. Investigations identify contribution values on XGBoost's raw-margin scale. A contribution indicates model influence, not causation. Suggested actions are general investigation steps, not physical interpretations of anonymized readings.

## Agent And Optional LLM

LangGraph orchestrates four explicit steps: retrieve vehicle history, call prediction, obtain feature contributions, and compose an assessment. With no `FACTORYIQ_LLM_API_KEY` or `OPENAI_API_KEY`, the summary source is `rule_based` and no external request is made. An optional OpenAI-compatible endpoint can be configured through `FACTORYIQ_LLM_BASE_URL`, `FACTORYIQ_LLM_MODEL`, and the key environment variable. The API reports `summary_source` so the response does not imply a deterministic summary came from an LLM.

## Limitations

- The target concerns observed Component X repair events and the dataset's observation window; censoring is not treated as a healthy outcome.
- Historical records and anonymized columns do not establish physical cause, component state, or failure date.
- The test metrics show modest positive-class precision and recall and are not calibrated deployment metrics.
- Demonstration risk thresholds are not validated maintenance or safety thresholds.
- The API is intended for local development and frontend integration, not real fleet control or safety decisions.

## Day 3 Frontend Dashboard

The React dashboard lives in `frontend/` and consumes the FastAPI prediction and fleet-summary contracts without changing model artifacts.

### Architecture and technologies

- React 19, TypeScript, and Vite provide the SPA and production build.
- React Router handles Overview, Vehicle Explorer, Vehicle Details, Repair-Risk Prediction, AI Investigation, and Model Insights.
- TanStack Query manages API request lifecycle, cache, loading, cancellation, and retry behavior.
- Lucide React provides interface icons. CSS provides responsive layouts and the dark industrial theme.
- Vitest, jsdom, and Testing Library cover navigation, API rendering, selection, prediction, investigation, and errors.

Key structure:

```text
frontend/
  .env.example
  src/
    App.tsx             # Layout, routes, pages, session-only activity
    App.test.tsx        # UI and API integration behavior tests
    index.css           # Responsive dashboard styling
    lib/api.ts          # Typed FastAPI client and error handling
    main.tsx            # Router and TanStack Query setup
```

### Run locally

In one PowerShell terminal from the project root:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

In another PowerShell terminal:

```powershell
Set-Location frontend
Copy-Item .env.example .env
npm install
npm run dev
```

The dashboard is served at `http://localhost:5173`; FastAPI and its Swagger UI are at `http://127.0.0.1:8000` and `http://127.0.0.1:8000/docs`.

`frontend/.env.example` sets `VITE_API_BASE_URL=http://127.0.0.1:8000`. Set this environment variable to the API origin before starting Vite when using a different backend host. The backend's existing local CORS allowlist includes the Vite origins above.

### Pages and backend integration

- **Fleet scoring:** on backend startup, the saved pipeline scores the eligible IDs in `data/split_assignments.csv` as one batch from the existing step-100 feature/specification files. Predictions and summary statistics are cached in memory; dashboard reads do not trigger fleet-wide inference. No labels or post-cutoff outcome fields are loaded by the inference service.
- **Overview:** reports all eligible scored vehicles, the cached risk-band distribution, and highest-probability vehicles. Browser-session investigation activity remains separate.
- **Vehicle Explorer:** fetches `GET /vehicles` in pages of 25 with actual cached probabilities and both the risk band and binary-threshold class. Since the API has no search parameter, ID search applies only to the currently fetched page.
- **Vehicle Details:** fetches `GET /vehicles/{vehicle_id}` and displays the model estimate, plain-language interpretation, prediction window, suggested investigation step, and limitations. Raw operational feature summaries are inside an expandable Advanced Details section.
- **Repair-Risk Prediction:** submits `POST /predict` with `{"vehicle_id": n}` and displays the returned probability, risk category, window, model version, and timestamp.
- **AI Investigation:** submits `POST /agent/investigate` and displays the returned summary source (`rule_based` or `llm`), feature contributions, actions, and disclaimer. Deterministic output is labeled as such.
- **Model Insights:** displays the held-out results verified against `models/evaluation_metrics.json` (2,222 vehicles; ROC-AUC 0.7933; average precision 0.2615; precision 0.3259; recall 0.3212; F1 0.3235; TN 1,994; FP 91; FN 93; TP 44; validation-selected threshold 0.1946). Values are a fixed presentation of the saved evaluation artifact, not a live API response.

The validation-selected binary classification threshold (0.1946) is distinct from the dashboard's Low/Medium/High demonstration bands (<0.20, 0.20 to <0.50, and ≥0.50). The bands are not calibrated operational cutoffs. The frontend does not send feature payloads or repair labels. Model risk categories and proposed actions are for investigation only; no thresholds or metrics are represented as production-validated.

The current eligible cohort contains 14,809 vehicles. Saved-model probabilities range from 0.0007 to 0.9056 (mean 0.0592, median 0.0239); 1,009 are at or above the 0.1946 binary threshold. The demonstration-band counts are 13,831 Low, 734 Medium, and 244 High. These describe model scores across train, validation, and test records, not an unbiased evaluation; the training scores are in-sample.

### Frontend checks

Run from `frontend/`:

```powershell
npm run typecheck
npm run lint
npm test
npm run build
```

### Known limitations

- Search is limited to one paginated API response; server-side search is not part of the current backend contract.
- Recent investigation activity resets when the page/browser session is reloaded; fleet risk statistics and eligible vehicle predictions are backend startup caches, not browser-session counts.
- Evaluation metrics are presented from the saved local JSON file and are not exposed through a new backend endpoint.
- Anonymized features and raw-margin contributions do not identify physical measurements or causes. Predictions do not guarantee repairs or estimate a failure date.
- The UI is not a production maintenance, safety, or fleet-control system.

### Day 4 deployment/demo preparation

- Select and document a deployment topology, configure exact production CORS origins, and use HTTPS.
- Add deployment-specific API configuration and a repeatable frontend/backend startup or container workflow.
- Review data access, operational security, observability, and retention requirements before exposing vehicle information.
- Prepare an evaluation/demo script covering cohort scope, class imbalance, threshold limitations, and deterministic-vs-LLM investigation behavior.
- Validate the deployed UI against the intended environment and user workflows; do not imply production readiness without domain validation.