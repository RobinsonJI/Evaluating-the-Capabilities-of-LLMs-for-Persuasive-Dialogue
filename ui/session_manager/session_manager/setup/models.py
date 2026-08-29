import uuid
import random
import string
import argparse
import math
from pathlib import Path
from typing import List, Tuple, Set, Dict
from itertools import product

import yaml
from pydantic import BaseModel
from typing import Optional

from session_manager.models.entities import Participant, Session, SessionParameters
from session_manager.models.enums import SpeakerType, ModelName
from session_manager.data import SQLParams

class ModelConfig(BaseModel):
    """Configuration for a single AI model and its enabled variants."""
    model: ModelName
    variants: List[SpeakerType]


class HumansConfig(BaseModel):
    """Configuration for human participants."""
    n: int
    c: Optional[int] = None  # debates per human, computed if not provided
    emails: Optional[List[str]] = None
    human_model_name: ModelName


class SQLConfig(BaseModel):
    """SQL connection configuration for all databases."""
    session_manager: SQLParams
    persuasio: SQLParams
    logs: SQLParams


class ExperimentConfig(BaseModel):
    """Configuration for a batched experiment.

    Defines the parameters for generating a complete experiment with
    balanced H-H and H-M debate assignments scheduled into batches.
    """
    models: List[ModelConfig]
    humans: HumansConfig
    sql: SQLConfig
    repeats: int = 4
    left_position: str = "20:40"
    right_position: str = "60:80"
    debate_topic: str = (
        "Debate the trade-offs of government intervention across a range of "
        "issues, including healthcare, immigration, and welfare."
    )

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))

class Batch(BaseModel):
    """A batch of debate sessions to run concurrently."""
    batch_id: int
    session_ids: List[str]