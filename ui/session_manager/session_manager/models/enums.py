from enum import Enum

class SessionStatus(Enum):
    """Possible states of a debate session."""
    STARTED = "Started."
    FINISHED = "Finished."
    RUNNING = "Running."

class SpeakerOrder(Enum):
    FIRST_SPEAKER = "first_speaker"
    SECOND_SPEAKER = "second_speaker"

class SpeakerType(Enum):
    HUMAN = "human"
    BASE = "base"
    MAS = "mas"
    MAS_RAG = "mas_rag"

class ModelName(Enum):
    NO_MODEL = ""
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_5 = "gpt-5"
    GPT_4O = "gpt-4o"
    GROK_3 = "grok-3"
    GROK_4 = "grok-4"
    MISTRAL_MEDIUM = "mistral-medium"

class UtteranceClassificationApproach(Enum):
    ALL_CLASSIFICATIONS = "all-classifications"
    MOST_CLASSIFIED = "most-classified"
    SINGLE_CLASSIFICATION = "single-classification"

class PoliticalPositionEnsembleOrModelName(Enum):
    NO_USE_OF_KNOWLEDGE_BASE = ""
    ENSEMBLE_1_ALL_MODELS = "ensemble1"
    ENSEMBLE_2_REASONING_MODELS = "ensemble2"
    ENSEMBLE_3_LESS_POL_SCORES_THAN_NA = "ensemble3"
    CLAUDE_35_HAIKU = "claude_3_5_haiku_2024102"
    CLAUDE_37_SONNET = "claude_3_7_sonnet_20250219"
    DEEPSEEK_R1 = "deepseek_r1"
    DEEPSEEK_V3 = "deepseek_v3"
    GEMINI_15_PRO = "gemini_1_5_pro_002"
    GEMINI_25_FLASH = "gemini_2_5_flash_preview"
    GPT_35_TURBO = "gpt_3_5_turbo_0125"
    GPT_45 = "gpt_4_5_preview_2025_02_27"
    GPT_4_TURBO = "gpt_4_turbo_2024_04_09"
    GPT_4O = "gpt_4o_2024_08_06"
    GPT_4O_MINI = "gpt_4o_mini_2024_07_18"
    GPT_O3_MINI = "o3_mini_2025_01_31"
    GROK_2 = "grok_2_1212"
    LLAMA_31_8B = "llama3_1_8b"
    LLAMA_32_3B = "llama3_2_3b"
    LLAMA_31_405B = "llama_3_1_405b"
    LLAMA_33_70B = "llama_3_3_70b_instruct"
    LLAMA_4_MAVERICK = "llama_4_maverick"
    MISTRAL_7B = "mistral_7b"
    PHI_4 = "phi_4"
    QWEN_3_235B = "qwen3_235b_a22b"
    QWEN_QWQ_32B = "qwen_qwq_32b"

class AuthStatus(Enum):
    """Authentication result statuses"""
    SUCCESS = "success"
    INVALID_CODE = "invalid_code"

class JoinSessionStatus(Enum):
    """Join session result statuses"""
    JOINED = "joined"
    NOT_AUTHENTICATED = "not_authenticated"
    INVALID_SESSION = "invalid_session"
    NOT_ASSIGNED = "not_assigned"
    INVALID_PARTICIPANT = "invalid_participant"
    ERROR = "error"

class SendMessageStatus(Enum):
    """Send message result statuses"""
    SENT = "sent"
    NOT_IN_SESSION = "not_in_session"
    INVALID_SESSION = "invalid_session"
    NOT_YOUR_TURN = "not_your_turn"
    BACKEND_ERROR = "backend_error"

class EndSessionStatus(Enum):
    """End session result statuses"""
    SUCCESS = "success"
    INVALID_SESSION = "invalid_session"
    PERSUASIO_ERROR = "persuasio_error"
    ERROR = "error"
