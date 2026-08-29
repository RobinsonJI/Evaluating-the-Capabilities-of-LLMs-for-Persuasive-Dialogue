from pydantic import BaseModel, Field, field_validator
from typing import List, Tuple, Dict
from datetime import datetime
import json
from typing import get_origin, get_args

class DataForOneDialogueTurn(BaseModel):
    """
    Stores the data for a single dialogue turn
    """
    speaker: str
    sentences_with_utterance_types: List[Tuple[str, str]]
    sentences_no_utterance_types: str
    timestamp: str = Field(default_factory=lambda : datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

class SafeBaseModel(BaseModel):
    """
    A universal base class that can intelligently coerce various LLM response formats
    (raw strings, dicts, lists, or JSON thereof) into the expected schema for the model.
    """

    @classmethod
    def from_llm(cls, raw_output):
        """
        Coerce raw LLM output into a shape that matches this model’s fields.
        Handles:
        - Plain string → {first_field: string}
        - JSON string (object, array, or string)
        - List → {first_field: list}
        - Dict → used directly if compatible
        """

        first_field = next(iter(cls.model_fields.keys()))
        field_info = cls.model_fields[first_field]
        field_type = field_info.annotation

        # Helper to detect if a field expects a list or list of lists
        def is_list_of_lists(tp):
            origin = get_origin(tp)
            args = get_args(tp)
            return origin in (list, List) and len(args) > 0 and get_origin(args[0]) in (list, List)


        # Step 1: Try to parse JSON if it's a string
        parsed = raw_output
        if isinstance(raw_output, str):
            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                parsed = raw_output  # keep as string if not JSON

        # Step 2: Determine model's first expected field
        first_field = next(iter(cls.model_fields.keys()))

        # Step 3: Handle dicts (try direct load)
        if isinstance(parsed, dict):
            try:
                return cls(**parsed)
            except Exception:
                # fallback to wrapping entire dict under first field
                return cls(**{first_field: parsed})

        # Step 4: Handle lists (wrap as list under first field)
        elif isinstance(parsed, list):
            if is_list_of_lists(field_type):
                # Field expects list of lists
                if all(isinstance(x, list) for x in parsed):
                    return cls(**{first_field: parsed})
                else:
                    return cls(**{first_field: [parsed]})
            elif get_origin(field_type) in (list, List):
                # Field expects a flat list
                if all(not isinstance(x, list) for x in parsed):
                    return cls(**{first_field: parsed})
                else:
                    # Flatten nested structure if needed
                    flat = [item for sub in parsed for item in (sub if isinstance(sub, list) else [sub])]
                    return cls(**{first_field: flat})
            else:
                # Unexpected type — fallback
                return cls(**{first_field: parsed})

        # Step 5: Handle strings (wrap as string under first field)
        elif isinstance(parsed, str):
            return cls(**{first_field: parsed})

        # Step 6: Fallback — unknown structure, stringify it
        return cls(**{first_field: str(parsed)})

class UtteranceClass(SafeBaseModel):
    """
    Store the utterance classification agent results.
    """
    Classification : str = Field(..., description="A string containing the utterance type classification for a given sentence.")

    @field_validator("Classification")
    def ensuring_correct_outputs(cls, string):

        if "claim" in string.lower():
            return "___Claim___"
        elif "since" in string.lower():
            return "___Since___"
        elif "why" in string.lower():
            return "___Why___"
        elif "question" in string.lower():
            return "___Question___"
        elif "concede" in string.lower():
            return "___Concede___"
        elif "retract" in string.lower():
            return "___Retract___"
        else:
            raise ValueError(f"UtteranceClassificationAgent did not return the right UtteranceClass(BaseModel). Model return '{string}' which is wrong.")
        
class ClaimNegationResponse(SafeBaseModel):
    Claim: List[str]

class WhyClaimResponses(SafeBaseModel):
    Why: List[str]

class QuestionClaimResponses(SafeBaseModel):
    Question: List[str]

class ConcedeClaimResponses(SafeBaseModel):
    Concede: List[str]

class SinceClaimResponses(SafeBaseModel):
    Since: List[str]

class ClaimResponses(SafeBaseModel):
    Claim: List[str]

class RetractClaimResponses(SafeBaseModel):
    Retract: List[str]

class PersuasivenessChoice(SafeBaseModel):
    Choice: str

class PersuasivenessChoicesList(SafeBaseModel):
    Choices: List[str]

class SimilarSentencesResponse(SafeBaseModel):
    ListOfSimilarSentences: List[List[str]] = Field(
        ..., description="Groups of sentences with the same or highly similar meaning."
    )

    @field_validator("ListOfSimilarSentences")
    def no_singletons(cls, groups):
        """
        This method is for error handling to ensure that model has returned lists of sentences where each list contains 2 or more sentences.
        """
        list_of_sentences = []
        for group in groups:
            if len(group) > 1:
                list_of_sentences.append(group)
        return list_of_sentences
    

class KeyPolicyPositions(BaseModel):
    taxation: str
    healthcare: str
    education: str
    immigration: str
    environment: str
    defense: str

class InformationSources(BaseModel):
    preferred_media: List[str]
    trusted_figures: List[str]
    information_processing: List[str]

class IdeologicalFramework(BaseModel):
    political_values: List[str]      
    role_of_government: List[str]     
    economic_philosophy: List[str]       
    social_issues: List[str]             
    change_approach: List[str]  

class CommunicationStyle(BaseModel):
    discourse_style: List[str]          
    argumentation_approach: List[str]   
    rhetoric_patterns: List[str]      
    emotional_triggers: List[str]     

    
class PersonaSchema(BaseModel):
    political_stance: str = Field(
        ..., 
        description="Precise political label using standard terminology. Examples: 'centre', 'left', 'right', 'far-right', 'far-left', 'extremely far-right', 'extremely far-left'. Use hyphenated compound terms when appropriate (e.g., 'centre-right', 'centre-left')."
    )
    stance_range: List[int] = Field(
        ...,
        description="Two integers [min, max] representing the numerical range of political positioning on a 0-100 scale where 0=extremely far-left, 50=centre, 100=extremely far-right."
    )
    core_values: List[str] = Field(
        ..., 
        description="3-5 fundamental political principles that drive this persona's worldview. Be specific and actionable. Examples: 'individual economic freedom', 'social safety net expansion', 'traditional family structures', 'environmental sustainability', 'limited government intervention', 'racial equity', 'fiscal responsibility'. Avoid vague terms like 'fairness' or 'justice' without qualification."
    )

    political_positions_on_main_topic : List[str] = Field(
        ..., 
        description="3-5 specific policy positions or viewpoints this persona holds regarding the main topic being discussed. Each should be a complete stance, not just a keyword. Example: 'Supports universal healthcare as a fundamental right' rather than just 'healthcare'. Include nuanced positions that reflect real political complexity."
    )
    main_topic : str = Field(
        ..., 
        description="Clear, concise summary of the primary subject matter being discussed. Should be specific enough to guide relevant policy positions. Examples: 'Healthcare reform and universal coverage', 'Immigration policy and border security', 'Climate change and environmental regulation', 'Economic inequality and taxation'."                
    )
    key_policy_positions: KeyPolicyPositions = Field(
        ..., 
        description="Dictionary mapping 4-6 major policy areas to this persona's specific stances. Keys should be policy domains (e.g., 'taxation', 'healthcare', 'education', 'immigration', 'environment', 'defense'). Values should be detailed position statements explaining both the stance and reasoning. Example: {'taxation': 'Supports progressive taxation with higher rates on wealthy to fund social programs, believing this reduces inequality while maintaining economic growth'}"
    )
    information_sources : InformationSources = Field(
        ..., 
        description="Dictionary with three keys: 'preferred_media' (news sources they trust, e.g., 'The Guardian', 'Wall Street Journal', 'NPR'), 'trusted_figures' (specific politicians, commentators, academics they follow, e.g., 'Elizabeth Warren', 'Ben Shapiro', 'Thomas Sowell'), and 'information_processing' (how they assess political information, e.g., 'peer-reviewed research', 'grassroots testimonials', 'historical precedent', 'economic data'). Each list should contain 3-5 specific, realistic entries."
    )
    ideological_framework : IdeologicalFramework = Field(
        ..., 
        description="Dictionary with four keys describing core philosophical positions: 'political_values' (views on key political issues/topics), 'role_of_government' (views on government scope and intervention, e.g., 'minimal regulation except for market failures', 'strong social programs to ensure equality'), 'economic_philosophy' (preferred economic approach, e.g., 'free market capitalism with safety net', 'democratic socialism', 'regulated capitalism'), 'social_issues' (positions on cultural matters, e.g., 'traditional values with personal freedom', 'progressive social change through legislation'), and 'change_approach' (how they prefer societal change, e.g., 'gradual reform through institutions', 'grassroots activism and protest', 'market-driven solutions'). Each list should contain 2-3 specific philosophical positions."
    )
    communication_style: CommunicationStyle = Field(
    ..., 
    description="Dictionary with four keys describing how this persona communicates politically: 'discourse_style' (overall approach like 'academic and measured', 'populist and direct', 'pragmatic and solution-focused', 'idealistic and principled'), 'argumentation_approach' (how they build arguments, e.g., 'cites data and research', 'uses personal anecdotes', 'appeals to moral principles', 'references historical precedents'), 'rhetoric_patterns' (specific language habits, e.g., 'frames issues as moral imperatives', 'uses business metaphors', 'emphasizes grassroots voices', 'quotes founding documents'), and 'emotional_triggers' (topics that provoke passionate responses, e.g., 'attacks on civil liberties', 'economic inequality', 'threats to traditional values', 'environmental destruction'). Each list should contain 2-4 specific, realistic communication characteristics."
    )


    @field_validator("stance_range")
    def validate_stance_range(cls, v):
        if len(v) != 2:
            raise ValueError("stance_range must contain exactly two integers")
        if not all(isinstance(i, int) for i in v):
            raise ValueError("stance_range values must be integers")
        return v
    
class BaseModelCompletion(SafeBaseModel):
    response : str

    @field_validator("response", mode="before")
    def validate_response(cls, v):
        if isinstance(v, dict):
            if "response" not in v:
                raise ValueError("Missing 'response' key in dictionary.")
            v = v["response"]

        if not isinstance(v, str):
            raise TypeError(f"Expected string or dict, got {type(v).__name__}")

        # Case 3: Empty string
        if len(v.strip()) == 0:
            raise ValueError("Model did not respond with any text.")

        return v
        

class IsSimilar(SafeBaseModel):
    similar : bool

class Concedes(SafeBaseModel):
    concede : bool

class Retracts(SafeBaseModel):
    retract : bool

class RetractedSentences(SafeBaseModel):
    retracted_sentences : List[str]