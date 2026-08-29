from pydantic import BaseModel
from typing import Optional, List, Tuple, Dict
from enum import Enum

class UtteranceType(str, Enum):
    CLAIM = "claim"
    SINCE = "since"
    WHY = "why"
    QUESTION = "question"
    CONCEDE = "concede"
    RETRACT = "retract"
    DISAGREEING = "disagreeing"

class Utterance(BaseModel):
    id: str
    text: str
    utterance_type: UtteranceType

    def str(self) -> str:
        """ String representation of the utterance """
        return f"[{self.utterance_type.value}] {self.text}"

    def flatten(self) -> dict:
        """ Flatten utterance data for CSV export """
        return {
            'utterance_id': self.id,
            'utterance_text': self.text,
            'utterance_type': self.utterance_type.value
        }

class Dialogue(BaseModel):
    id: str
    data: list[Utterance] # list of utterances, most recent last

    def add_utterance(self, utterance: Utterance):
        self.data.append(utterance)

    def type(self, cutoff: Optional[int] = None) -> str:
        if cutoff is None:
            cdata = self.data
        else:
            cdata = self.data[:-cutoff]
        return "_".join([u.utterance_type.value for u in cdata])

    def flatten(self) -> dict:
        data = {'id': self.id, "type": self.type(), "type-1": self.type(1), "length": len(self.data)}
        for i, u in enumerate(self.data):
            data[f'u{i}_id'] = u.id
            data[f'u{i}_text'] = u.text
            data[f'u{i}_type'] = u.utterance_type.value

        return data

    def str(self, cutoff: Optional[int] = None) -> str:
        if cutoff is None:
            cdata = self.data
        else:
            cdata = self.data[:-cutoff]
        return "\n".join([f"{i+1}. {u.str()}" for i, u in enumerate(cdata)])
    
    def prompt(self) -> List[Dict[str, str]]:
        from utils import SYSTEM_MESSAGE

        content = f"""Classify the following sentence according to the defined utterance types:
Previous dialogue turns:
{chr(10).join([f"{i+1}. {u.str()}" for i, u in enumerate(self.data[:-1])])}

Current sentence:
'{self.data[-1].text}'"""

        return [SYSTEM_MESSAGE, {"role": "user", "content": content}]

    @classmethod
    def from_dict(cls, data: dict) -> 'Dialogue':
        """Reconstruct Dialogue from flattened CSV data"""
        dialogue_utterances = []

        # Extract all utterances from the dialogue
        i = 0
        while f'u{i}_text' in data:
            if data.get(f'u{i}_text') is not None:
                u_utterance = Utterance(
                    id=str(data.get(f'u{i}_id', '')),
                    text=str(data[f'u{i}_text']),
                    utterance_type=UtteranceType(data[f'u{i}_type'])
                )
                dialogue_utterances.append(u_utterance)
            i += 1

        return cls(
            id=str(data.get('id', '')),
            data=dialogue_utterances
        )


class ModelParameters(BaseModel):
    temperature: float = 0.0
    seed: Optional[int] = None


class Prediction(BaseModel):
    utterance: Utterance
    dialogue: Dialogue
    label: UtteranceType  # majority vote or single prediction
    model_name: str
    model_params: ModelParameters

    # Optional for repetitions
    r_labels: Optional[List[UtteranceType]] = None
    individual_seeds: Optional[List[int]] = None