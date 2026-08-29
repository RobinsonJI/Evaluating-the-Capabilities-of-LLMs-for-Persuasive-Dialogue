from pathlib import Path
import json

from persuasio.states.state import ParentState
from persuasio.datatypes.enums import ModelName, Mode
from persuasio.utils.logs import log_function
from persuasio.utils.parsers import enum_encoder

@log_function
def save_dialogue_outputs(state: ParentState) -> ParentState:

    if state["mode"] == Mode.PROD or (state["mode"] == Mode.PROD.value):
        # Return state without saving locally because the final state will be saved in persuasio/api.py
        return state

    # Local saving.
    root_path = Path(__file__).parent.parent

    save_directory = root_path / 'outputs' / 'dialogues' / f'{state["first_speaker_type"].value}-to-{state["second_speaker_type"].value}'

    if not save_directory.exists():
        save_directory.resolve().mkdir(parents=True, exist_ok=False)

    file_name = f'{state["session_id"]}.json'.replace(":", "_")
    file_path = save_directory / file_name

    state_as_dict = state.copy()

    state_as_dict["dialogue_history"] = [x.model_dump() for x in state_as_dict["dialogue_history"]]

    with file_path.open('w', encoding='utf-8') as file:
        json.dump(state_as_dict, file, indent=4, default=enum_encoder, ensure_ascii=False)

    return state