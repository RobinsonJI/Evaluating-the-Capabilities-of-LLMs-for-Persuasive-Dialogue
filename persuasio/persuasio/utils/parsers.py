from typing import List, Dict
import argparse
from enum import Enum
from pydantic import BaseModel

from persuasio.datatypes.enums import SpeakerType,SessionStatus, ModelName, SpeakerOrder, UtteranceClassificationApproach, PoliticalPositionEnsembleOrModelName, Mode
from persuasio.datatypes.api import PersuasioResponse
from persuasio.datatypes.pydantic_basemodels import DataForOneDialogueTurn

def parse_political_position_elements(string : str, sep=":") -> List[int]:
    """
    Parse a string containing two number numbers separated by a colon and return a list of integers values (of length == 2)
    """
    elements = string.split(sep)

    assert len(elements) == 2

    integer_elements = []

    for element in elements:
        element = element.strip() # remove whitespace 
        try:
            integer_elements.append(int(element))
        except ValueError:
            print("You must specify integer values (from 'parse_elements' function in /utils/parsers.py)")
            
    return sorted(integer_elements)

def parse_state_to_persuasio_basemodel_object(session_state : Dict, session_status : SessionStatus) -> PersuasioResponse:
    """
    Takes the state returned by LangGraph, parses it, and returns a PersuasioReponse object that will be sent back to session manager.
    """

    if (session_status == SessionStatus.FINISHED) or (session_status == SessionStatus.TERMINATED):
        response = PersuasioResponse(
                dialogue_history=[x.model_dump(mode="json") if isinstance(x,BaseModel) else x for x in session_state["dialogue_history"]],
                first_speaker=session_state["first_speaker"],
                first_speaker_utterance=session_state["first_speaker_utterance"],
                first_speaker_commitments=session_state["first_speaker_commitments"],
                first_speaker_original_claim=session_state["first_speaker_original_claim"],
                first_speaker_typical_replies_for_second_speakers_response=[],

                second_speaker= session_state["second_speaker"],
                second_speaker_utterance= session_state["second_speaker_utterance"],
                second_speaker_commitments= session_state["second_speaker_commitments"],
                second_speaker_original_claim= session_state["second_speaker_original_claim"],
                second_speaker_typical_replies_for_first_speakers_response= [],

                winner= session_state["winner"],
                loser=session_state["loser"],
                reason_for_dialogue_termination=session_state["reason_for_dialogue_termination"],

                session_status=session_status
        )
        return response

    if session_state["current_speaker"] == SpeakerOrder.FIRST_SPEAKER:
        response = PersuasioResponse(
                dialogue_history=[x.model_dump(mode="json") if isinstance(x,BaseModel) else x for x in session_state["dialogue_history"]],
                first_speaker=session_state["first_speaker"],
                first_speaker_utterance=session_state["first_speaker_utterance"],
                first_speaker_commitments=session_state["first_speaker_commitments"],
                first_speaker_original_claim=session_state["first_speaker_original_claim"],
                first_speaker_typical_replies_for_second_speakers_response=session_state["__interrupt__"][0].value["typical_replies_for_last_speakers_response"],

                second_speaker= session_state["second_speaker"],
                second_speaker_utterance= session_state["second_speaker_utterance"],
                second_speaker_commitments= session_state["second_speaker_commitments"],
                second_speaker_original_claim= session_state["second_speaker_original_claim"],
                second_speaker_typical_replies_for_first_speakers_response= [],

                winner= session_state["winner"],
                loser=session_state["loser"],
                reason_for_dialogue_termination=session_state["reason_for_dialogue_termination"],

                session_status=SessionStatus.RUNNING
        )
    elif session_state["current_speaker"] == SpeakerOrder.SECOND_SPEAKER:
        response = PersuasioResponse(
                dialogue_history=[x.model_dump(mode="json") if isinstance(x,BaseModel) else x for x in session_state["dialogue_history"]],
                first_speaker=session_state["first_speaker"],
                first_speaker_utterance=session_state["first_speaker_utterance"],
                first_speaker_commitments=session_state["first_speaker_commitments"],
                first_speaker_original_claim=session_state["first_speaker_original_claim"],
                first_speaker_typical_replies_for_second_speakers_response=[],

                second_speaker= session_state["second_speaker"],
                second_speaker_utterance= session_state["second_speaker_utterance"],
                second_speaker_commitments= session_state["second_speaker_commitments"],
                second_speaker_original_claim= session_state["second_speaker_original_claim"],
                second_speaker_typical_replies_for_first_speakers_response= session_state["__interrupt__"][0].value["typical_replies_for_last_speakers_response"],

                winner= session_state["winner"],
                loser=session_state["loser"],
                reason_for_dialogue_termination=session_state["reason_for_dialogue_termination"],

                session_status=SessionStatus.RUNNING
        )
    

    return response


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
    

# custom encoder
def enum_encoder(obj):
    if isinstance(obj, Enum):
        return obj.value
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


