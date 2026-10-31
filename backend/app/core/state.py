"""Shared application state for PlantPath AI."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.database import Database
    from app.ml.torch_classifier import TorchClassifier
    from app.services.disease_service import DiseaseService
    from app.services.prediction_service import PredictionService


class AppState:
    settings = None
    classifier: TorchClassifier | None = None
    prediction_service: PredictionService | None = None
    disease_service: DiseaseService | None = None
    database: Database | None = None