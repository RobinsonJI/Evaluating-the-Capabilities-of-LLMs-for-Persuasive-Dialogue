import re
from typing import List, Tuple, TypeVar


from persuasio.models.models import GenerateLLMResponses
from persuasio.states.state import HumanState, GenerationAgentsState, BaseModelState
# from persuasio.tools.check_state import is_state
from persuasio.tools.check_state import is_instance_of_typed_dict
from persuasio.prompts.generators.utterance_classification import create_utterance_classification_prompt
from persuasio.datatypes.pydantic_basemodels import UtteranceClass
from persuasio.datatypes.enums import UtteranceClassificationApproach, ClassifyingUtteranceOf, LogLevels
from persuasio.utils.logs import log_class, log


# TypeVar allows this class to be initialised with multiple possible state types
T = TypeVar("T", HumanState, GenerationAgentsState, BaseModelState)


@log_class
class UtteranceClassificationAgent:
    """
    An agent that classifies utterances into different categories using LLM responses.

    The agent supports classification of:
        - The last speaker's utterance
        - The human user's response
    """

    def __init__(self, state: T, which_utterances : ClassifyingUtteranceOf):
        """
        Initialise the UtteranceClassificationAgent.

        Args:
            state (T): The state of the dialogue. Can be `HumanState`, `GenerationAgentsState`, or `BaseModelState`.
            which_utterances (ClassifyingUtteranceOf): Specifies whether to classify the last speaker's utterance
                or the human's response.
        """
        

        self.state = state
        self.session_id = state["session_id"]

        # check whether state is HumanState
        if is_instance_of_typed_dict(self.state, HumanState):
            self.model = state["human_model_name"]
            self.temp = state["human_model_temp"]
            self.top_p = state["human_model_top_p"]
            self.seed = state["human_model_seed"]

        elif (is_instance_of_typed_dict(self.state, GenerationAgentsState)) or (is_instance_of_typed_dict(self.state, BaseModelState)):
            # State will be either of type GenerationAgentsState or BaseModelState
            self.model = state["speaker_model_name"]
            self.temp = state["model_temp"]
            self.top_p = state["model_top_p"]
            self.seed = state["model_seed"]

        self.which_utterances = which_utterances

        # Parameters for classification strategy
        self.approach = state["utterance_classification_approach"]
        self.k = state["utterance_classification_number_of_classifications"]

        # Splits user's response into sentences.
        if self.which_utterances == ClassifyingUtteranceOf.LAST_SPEAKER:
            self.sentences = re.split('(?<=[.!?]) +',self.state["opponents_utterance"])
        elif self.which_utterances == ClassifyingUtteranceOf.HUMAN_RESPONSE:
            self.sentences = re.split('(?<=[.!?]) +',self.state["utterance"])

        # Perform classification on the extracted sentences
        self.classified_sentences = self._classify_sentence_utterance_types()


    def _classify_sentence_utterance_types(self) -> List[Tuple[str, str]]:
        """
        Classifies each sentence into one or more utterance types.

        Returns:
            List[Tuple[str, str]]: A list of tuples where each tuple is (classification, sentence).
        """

        classifications = []
        for sent in self.sentences:
            # Build the classification prompt for the current sentence
            prompt = create_utterance_classification_prompt(current_sentence=sent, 
                                                            state=self.state, 
                                                            which_utterances = self.which_utterances)

            sentence_classifications = []
            # Generate multiple classifications (depending on k)
            for i in range(self.k):
                try:
                    result = GenerateLLMResponses(model_choice=self.model,
                                                prompt = prompt,
                                                temperature=self.temp,
                                                top_p= self.top_p,
                                                seed= self.seed,
                                                datatype_schema=UtteranceClass).return_completion()
                    
                    sentence_classifications.append(result["Classification"])
                    log(
                        session_id=self.session_id,
                        level=LogLevels.INFO,
                        service=self._classify_sentence_utterance_types.__name__,
                        message=f"'UtteranceClassificationAgent' completion returned and validated from MODEL = '{self.model}'.",
                        mode=self.state["mode"]
                    )
                except ValueError as e:
                    log(
                        session_id=self.session_id,
                        level=LogLevels.ERROR,
                        service=self._classify_sentence_utterance_types.__name__,
                        message=f"'UtteranceClassificationAgent' completion not validated; MODEL = '{self.model}'; \n REASON: \n\n {e}",
                        mode=self.state["mode"],
                        context={"prompt" : prompt, "state" : self.state, "exception" : e}
                    )


            # Determine how to aggregate classifications based on the chosen approach
            if self.approach == UtteranceClassificationApproach.ALL_CLASSIFICATIONS:
                # Keep all unique classifications
                sentence_with_utterance_types = [(utterance_type, sent) for utterance_type in list(set(sentence_classifications))]
            elif self.approach == UtteranceClassificationApproach.MOST_CLASSIFIED:
                # Use the most frequently occurring classification
                sentence_with_utterance_types = [(max(sentence_classifications), sent)]
            elif self.approach == UtteranceClassificationApproach.SINGLE_CLASSIFICATION:
                # Use one classification (only one completion will have been generated for this case, i.e. k = 1)
                sentence_with_utterance_types = [(sentence_classifications[0], sent)] 

            classifications.extend(sentence_with_utterance_types)

        return classifications
    
    def return_classified_sentences(self) -> T:
        """
        Returns the classified sentences annotated with their corresponding utterance types.

        Returns:
            T: A dictionary mapping the utterances to their classifications, 
               depending on whether the target was the last speaker or the human response.
        """
        
        if self.which_utterances == ClassifyingUtteranceOf.LAST_SPEAKER:
            return {
                "opponents_utterance_with_corresponding_types" : self.classified_sentences
            }
        elif self.which_utterances == ClassifyingUtteranceOf.HUMAN_RESPONSE:
            return {
                "utterance_with_corresponding_types" : self.classified_sentences
            }