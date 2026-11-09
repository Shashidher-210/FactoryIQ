"""Environment-backed backend configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

MODEL_DIR = Path(os.getenv("FACTORYIQ_MODEL_DIR", PROJECT_ROOT / "models"))
FEATURES_PATH = Path(os.getenv("FACTORYIQ_FEATURES_PATH", PROJECT_ROOT / "data/train_features_100step.csv"))
SPECIFICATIONS_PATH = Path(os.getenv("FACTORYIQ_SPECIFICATIONS_PATH", PROJECT_ROOT / "data/train_specifications.csv"))
SPLIT_ASSIGNMENTS_PATH = Path(
    os.getenv("FACTORYIQ_SPLIT_ASSIGNMENTS_PATH", PROJECT_ROOT / "data/split_assignments.csv")
)
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "FACTORYIQ_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
LLM_API_KEY = os.getenv("FACTORYIQ_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("FACTORYIQ_LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("FACTORYIQ_LLM_BASE_URL", "https://api.openai.com/v1")