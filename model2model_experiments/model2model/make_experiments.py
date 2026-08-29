from sympy import re
import yaml
import argparse
import json
import pandas as pd
import sys
from pathlib import Path
from itertools import permutations
from typing import List, Dict, Any, Optional
import uuid
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator

# # Add the persuasio directory to Python path
# sys.path.insert(0, (str(Path(__file__).parent.parent)+'/persuasio'))

from persuasio.datatypes.api import SessionParameters
from persuasio.datatypes.enums import SpeakerType, ModelName
from persuasio.utils.api_dependencies import session_parameters

READERS  = {
    ".csv": pd.read_csv,
    ".json": pd.read_json,
    ".jsonl": lambda p: pd.read_json(p, lines=True),
}

SAVERS = {
    "csv": lambda df, p: df.to_csv(p, index=False),
    "json": lambda df, p: df.to_json(p, orient='records', indent=2),
    "jsonl": lambda df, p: df.to_json(p, orient='records', lines=True),
}

class Speaker(BaseModel):
    name: str
    type: SpeakerType
    model: ModelName

class MatchupGenerator(BaseModel):
    """Generates experiment matchups from models, model_types, and humans.
    
    model_config (ConfigDict): Allow extra fields to be passed in for session parameters.
    models (List[ModelName]): List of model names to include.
    model_types (List[SpeakerType]): List of model types corresponding to models.
    humans (List[str]): List of human participant names.
    participants (List[Speaker]): Internal list of all participants generated.
    repetitions (int): Number of times to repeat each matchup.
    left_political_position_range (str): Political position range for left-leaning speakers.
    right_political_position_range (str): Political position range for right-leaning speakers.
    include_uuid (bool): Whether to append a UUID to each session ID for uniqueness.
    
    """

    model_config = ConfigDict(extra='allow')
    models: List[ModelName] = Field(default_factory=list)
    model_types: List[SpeakerType] = Field(default_factory=list)
    humans: List[str] = Field(default_factory=list)
    participants: List['Speaker'] = Field(default_factory=list)
    repetitions: int = 1
    left_political_position_range: str = "20:40"
    right_political_position_range: str = "60:80"
    include_uuid: bool = False

    @field_validator('models', mode='before')
    @classmethod
    def validate_models(cls, v):
        if isinstance(v, list):
            return [ModelName(item) if isinstance(item, str) else item for item in v]
        return v

    @field_validator('model_types', mode='before')
    @classmethod
    def validate_model_types(cls, v):
        if isinstance(v, list):
            return [SpeakerType(item) if isinstance(item, str) else item for item in v]
        return v

    @model_validator(mode='after')
    def validate_models_and_types(self):
        if self.models and not self.model_types:
            raise ValueError("model_types cannot be empty when models are specified")
        return self

    def generate_all_matchups(self) -> List[SessionParameters]:
        """Generate all possible matchups"""
        participants = []

        # Add models with their types
        if self.models and self.model_types:
            for model in self.models:
                for model_type in self.model_types:
                    participants.append(Speaker(
                        name=f"{model.value}_{model_type.value}",
                        type=model_type,
                        model=model,
                    ))

        # Add humans
        if self.humans:
            for human in self.humans:
                participants.append(Speaker(
                    name=human,
                    type=SpeakerType.HUMAN,
                    model=ModelName.NO_MODEL,
                ))

        self.participants = participants

        # Generate all combinations
        matchups = []
        for i in range(self.repetitions):
            for l, r in permutations(participants, 2):
                # l,r
                matchups.append(
                    self._create_session_params(
                        session_id= self._make_id(l, "l", r, "r"),
                        first_speaker=l,
                        second_speaker=r,
                        first_polpos = self.left_political_position_range,
                        second_polpos = self.right_political_position_range,
                    )
                )
                # r,l
                matchups.append(
                    self._create_session_params(
                        session_id= self._make_id(r, "r", l, "l"),
                        first_speaker=r,
                        second_speaker=l,
                        first_polpos = self.right_political_position_range,
                        second_polpos = self.left_political_position_range,
                    )
                )

        return matchups
    
    def _make_id(self, first: Speaker, first_stance: str, second: Speaker, second_stance: str) -> str:
        """Make id from two speaker names, plus model type if not human, plus l or r, plus uuid"""  
        id = f"{first.name}_{first_stance}__vs__{second.name}_{second_stance}"

        if self.include_uuid:
            id += "_session:" + str(uuid.uuid4())
        return id

    def _create_session_params(self, session_id: str, first_speaker: Speaker, first_polpos: str, second_speaker: Speaker, second_polpos: str) -> SessionParameters:
        """Create SessionParameters from participant info."""
        # Required params for this specific matchup
        params = {
            "session_id": session_id,
            "first_speaker": first_speaker.name,
            "first_speaker_type": first_speaker.type,
            "first_speaker_model_name": first_speaker.model,
            "first_speaker_political_political_position_range": first_polpos,
            "second_speaker": second_speaker.name,
            "second_speaker_type": second_speaker.type,
            "second_speaker_model_name": second_speaker.model,
            "second_speaker_political_position_range": second_polpos,
        }

        # Pull extras from Pydantic v2's storage instead of __dict__
        extra_fields: Dict[str, Any] = {}
        model_extra = getattr(self, 'model_extra', None)
        if model_extra is not None:
            extra_fields.update(model_extra)
        else:
            # Fallback for internal attribute name (older/internal)
            internal_extra = getattr(self, '__pydantic_extra__', None)
            if internal_extra is not None:
                extra_fields.update(internal_extra)

        # Exclude internal/runtime attributes we don't want to forward
        extra_fields.pop('participants', None)

        # Don't override any of the required params above
        extra_fields = {k: v for k, v in extra_fields.items() if k not in params}

        params.update(extra_fields)

        session = SessionParameters(**params)
        
        return session

    def save_experiments(
        self, 
        matchups: List[SessionParameters], 
        output_dir: str,
        output_name: str, 
        output_format: str = "csv") -> str:
        """Save experiment configurations to file."""
        # Prepare output
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(exist_ok=True)

        # Ensure the output name has the correct extension
        output_path_obj = Path(output_name)
        if output_path_obj.suffix != f".{output_format}":
            output_name_final = f"{output_path_obj.stem}.{output_format}"
        else:
            output_name_final = output_name


        output_path = output_dir_path / output_name_final

        # Convert to DataFrame with enum values as strings
        experiment_dicts = [matchup.model_dump(mode='json') for matchup in matchups]
        df = pd.DataFrame(experiment_dicts)

        # Save based on format using pandas methods
        SAVERS[output_format](df, output_path)

        return str(output_path)

    def __call__(
        self, 
        output_dir: str = "data", 
        output_name: str = None,
        output_format: str = "csv") -> tuple[List[SessionParameters], str]:
        """Generate all matchups and save them to file in one step."""
        matchups = self.generate_all_matchups()
        print(f"Generated {len(matchups)} experiment configurations for {len(self.participants)} participants")

        output_path = self.save_experiments(
            matchups=matchups,
            output_dir=output_dir,
            output_name=output_name,
            output_format=output_format,
        )

        return matchups, output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate experiment configurations from YAML config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--output-name", type=str, default=None, help="Name of file to save the generated experiment configurations.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save the output file.")
    parser.add_argument("--output-format", type=str, choices=["csv", "json", "jsonl"], default="csv", help="Output format.")
    args = parser.parse_args()

    if args.output_name is None:
        args.output_name = Path(args.config).stem
    
    if args.output_dir is None:
        args.output_dir = str(Path(args.config).parent.parent / "experiments")

    return args


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def count_experiments():
    parser = argparse.ArgumentParser(description="Count experiment configurations from config or experiments file.")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--experiments-file", type=str, help="Path to existing experiments file (CSV, JSON, or JSONL)")
    args = parser.parse_args()  

    if args.config:
        config = load_config(args.config)
        generator = MatchupGenerator(**config)
        matchups = generator.generate_all_matchups()
        print(f"{len(matchups)} experiments from {args.config}")

    elif args.experiments_file:
        experiments_path = Path(args.experiments_file)
        df = READERS.get(experiments_path.suffix.lower())(experiments_path)
        print(f"{len(df)} experiments from {args.experiments_file}")

    else:
        print("Please provide either --config or --experiments-file")
        sys.exit(1)
    
    return


def main():
    args = parse_args()
    config = load_config(args.config)

    generator = MatchupGenerator(**config)

    # Generate and save experiments in one step
    matchups, output_path = generator(
        output_dir=args.output_dir,
        output_name=args.output_name,
        output_format=args.output_format,
    )

    print(f"Saved {len(matchups)} experiments to {output_path}")


if __name__ == "__main__":
    main()