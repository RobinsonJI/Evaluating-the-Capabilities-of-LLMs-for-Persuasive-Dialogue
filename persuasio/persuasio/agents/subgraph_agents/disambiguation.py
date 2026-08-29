from typing import List, Tuple

from persuasio.states.state import GenerationAgentsState
from persuasio.models.models import GenerateLLMResponses
from persuasio.datatypes.pydantic_basemodels import (
    SimilarSentencesResponse,
    PersuasivenessChoice,
    PersuasivenessChoicesList
)
from persuasio.datatypes.enums import LogLevels
from persuasio.prompts.generators.disambiguation import (
    are_sentences_in_list_similar,
    choose_one_sentence,
    choose_most_persuasive_sents_to_return
)
from persuasio.utils.logs import log_class, log

@log_class
class DisambiguationAgent:
    """
    The DisambiguationAgent is responsible for detecting and removing semantically 
    redundant sentences in the LLM's latest response, while preserving at least one 
    representative sentence for each group of similar dialogue move types.

    This is useful when the LLM produces multiple sentences that essentially convey 
    the same meaning but are classified under dialogue move categories such as 
    ___Claim___, ___Since___, ___Question___, ___Why___, ___Concede___, or ___Retract___.

    Attributes:
        state (LLMResponseState):
            The dialogue state containing the latest LLM output, with each sentence paired 
            with its corresponding dialogue move type.
        sentences_to_remove (set):
            A set of sentences marked for removal because they are semantically redundant.
        sentences_with_utterance_types (list[list[str, str]]):
            The filtered list of sentences and their dialogue move types after disambiguation.
    """

    def __init__(self, state: GenerationAgentsState):
        """
        Initialises the DisambiguationAgent, identifies semantically similar sentences
        within the LLM output, and removes redundant ones.

        Args:
            state (LLMResponseState): The dialogue state containing the most recent 
            LLM utterances and their associated types.
        """

        self.state = state

        # If more than one sentence is in the latest LLM output, 
        # check for redundancy among similar dialogue move types
        if len(self.state["utterance_with_corresponding_types"]) > 1:

            # Tracks sentences the model decides to remove
            self.sentences_to_remove = set()

            # Group sentences by similar dialogue move types
            claims, questions_and_challenges, retract_and_concede = self._grouping_similar_utterance_types()

            # Run disambiguation checks for each group that contains >1 sentence
            if len(claims) > 1:
                self.sentences_to_remove.update(self._identifying_similar_sentences_and_choosing_one_to_return_to_user(sentences=claims))
            
            if len(questions_and_challenges) > 1:
                self.sentences_to_remove.update(self._identifying_similar_sentences_and_choosing_one_to_return_to_user(sentences=questions_and_challenges))

            if len(retract_and_concede) > 1:
                self.sentences_to_remove.update(self._identifying_similar_sentences_and_choosing_one_to_return_to_user(sentences=retract_and_concede))

            # Remove repetitive sentences from the state
            self.sentences_with_utterance_types = self._removing_repetitive_sentences()

        else:
            # If only one sentence exists, no disambiguation needed
            self.sentences_with_utterance_types = self.state["utterance_with_corresponding_types"]

        if len(self.sentences_with_utterance_types) > state["max_sentences"]:
            self.sentences_with_utterance_types = self._choose_the_most_persuasive_sentences()

    def _grouping_similar_utterance_types(self):
        """
        Groups sentences from the last LLM utterance into categories of similar dialogue move types.

        Returns:
            tuple: (claims, questions_and_challenges, retract_and_concede)
                - claims: List of sentences classified as ___Claim___ or ___Since___
                - questions_and_challenges: List of sentences classified as ___Question___ or ___Why___
                - retract_and_concede: List of sentences classified as ___Concede___ or ___Retract___
        """

        claims = []                     
        questions_and_challenges = []
        retract_and_concede = []
        for (utterance_type_key, sentence) in self.state["utterance_with_corresponding_types"]:
            # Group claim-like moves
            # ___Claim___ and ___Since___ are assumed to be similar
            if utterance_type_key in {"___Claim___","___Since___"}:
                claims.append(sentence)
            # Group question-like moves
            # ___Question___ and ___Why___ are assumed to be similar
            elif utterance_type_key in {"___Question___","___Why___"}:
                questions_and_challenges.append(sentence)
            # Group concession/retraction moves
            # ___Concede___ and ___Retract___ moves are assumed to be similar
            elif utterance_type_key in {"___Concede___","___Retract___"}:
                retract_and_concede.append(sentence)

        claims = list(set(claims))
        questions_and_challenges = list(set(questions_and_challenges))
        retract_and_concede = list(set(retract_and_concede))

        return claims, questions_and_challenges, retract_and_concede
    

    def _identifying_similar_sentences_and_choosing_one_to_return_to_user(self, sentences):
        """
        Identifies semantically similar sentences within a given list and keeps only one.

        The process involves:
        1. Asking the LLM to group similar sentences.
        2. For each group, asking the LLM to pick the single best sentence to keep.
        3. Marking the rest for removal.

        Args:
            sentences (list[str]): A list of sentences belonging to the same dialogue move category.

        Returns:
            set: A set of sentences to be removed.
        """

        # Prompt the LLM to group similar sentences
        prompt = are_sentences_in_list_similar(sentences=sentences)
        # Returns a 2D array of sentences grouped by same semantic meaning.
        result = None
        try:
            result = GenerateLLMResponses(model_choice=self.state["speaker_model_name"],
                                        prompt = prompt,
                                        temperature=self.state["model_temp"],
                                        top_p= self.state["model_top_p"],
                                        seed= self.state["model_seed"],
                                        datatype_schema=SimilarSentencesResponse).return_completion()
            log(
                    session_id=self.state["session_id"],
                    level=LogLevels.INFO,
                    service=self._identifying_similar_sentences_and_choosing_one_to_return_to_user.__name__,
                    message=f"'DisambiguationAgent' grouped similar sentences, completion returned and validated; MODEL = '{self.state['speaker_model_name'].value}'.",
                    mode=self.state["mode"]
                )
        except ValueError as e:
            log(
                    session_id=self.state["session_id"],
                    level=LogLevels.ERROR,
                    service=self._identifying_similar_sentences_and_choosing_one_to_return_to_user.__name__,
                    message=f"'DisambiguationAgent' could not group similar sentences, completion not validated; MODEL = '{self.state['speaker_model_name'].value}'; \n REASON: \n\n {e}",
                    mode=self.state["mode"],
                    context={"prompt" : prompt, "state" : self.state, "exception" : e}
                )

        sentences_to_remove = set()
        # Iterate over each similarity group
        for sents in result["ListOfSimilarSentences"]:
            # Ask LLM to pick the best representative sentence
            prompt = choose_one_sentence(sentences=list(set(sents)))
            result = None
            try:
                result = GenerateLLMResponses(model_choice=self.state["speaker_model_name"],
                                                prompt = prompt,
                                                temperature=self.state["model_temp"],
                                                top_p= self.state["model_top_p"],
                                                seed= self.state["model_seed"],
                                                datatype_schema=PersuasivenessChoice).return_completion()
                # Determine which sentences to remove
                # Create an empty list
                result_list = []
                # Append the completion (i.e. the string) output by the model to the list
                result_list.append(result["Choice"])

                # Convert list to string to conduct set difference
                result_set = set(result_list)
                # Compute the set difference
                sents_to_remove = set(sents) - result_set
                # Add sentences which the model has chosen to remove to the set tracking the sentences to remove
                sentences_to_remove.update(sents_to_remove)

                log(
                    session_id=self.state["session_id"],
                    level=LogLevels.INFO,
                    service=self._identifying_similar_sentences_and_choosing_one_to_return_to_user.__name__,
                    message=f"'DisambiguationAgent' chose sentences to remove, completion returned and validated from MODEL = '{self.state['speaker_model_name'].value}'.",
                    mode=self.state["mode"]
                )
            except ValueError as e:
                log(
                    session_id=self.state["session_id"],
                    level=LogLevels.ERROR,
                    service=self._identifying_similar_sentences_and_choosing_one_to_return_to_user.__name__,
                    message=f"'DisambiguationAgent' could not choose sentences to remove, completion not validated; MODEL = '{self.state['speaker_model_name'].value}'; \n REASON: \n\n {e}",
                    mode=self.state["mode"],
                    context={"prompt" : prompt, "state" : self.state, "exception" : e}
                )


        return sentences_to_remove


    def _removing_repetitive_sentences(self):
        """
        Removes sentences that have been marked as redundant from the latest LLM utterance.

        Returns:
            list[list[str, str]]: The updated list of (utterance_type, sentence) pairs 
                                   with duplicates removed.
        """

        # Find the indexes to delete
        indexes_to_del = []
        for index, (utterance_type, sentence) in enumerate(self.state["utterance_with_corresponding_types"]):
            if sentence in list(self.sentences_to_remove):
                indexes_to_del.append(index)

        # Copy the original list
        disambiguated_sentences = self.state["utterance_with_corresponding_types"].copy()

        # Delete the sentences that the model believed to be repetitive / says the same thing (reversed order to avoid index shifting issues)
        for index in sorted(indexes_to_del, reverse=True):
            del disambiguated_sentences[index]

        return list(set(disambiguated_sentences))
    
    def _choose_the_most_persuasive_sentences(self) -> List[Tuple[str, str]]:
        """
        The maximum allowable sentences have been reached. Model now needs to choose which sentences to return to user.
        """

        sentences = [sent for utt_type, sent in self.sentences_with_utterance_types.copy()]

        prompt = choose_most_persuasive_sents_to_return(sentences=sentences, max_sentences=self.state["max_sentences"])
        most_persuasive_disambiguated_sentences = []
        result = None
        try:
            result = GenerateLLMResponses(model_choice=self.state["speaker_model_name"],
                                        prompt = prompt,
                                        temperature=self.state["model_temp"],
                                        top_p= self.state["model_top_p"],
                                        seed= self.state["model_seed"],
                                        datatype_schema=PersuasivenessChoicesList).return_completion()
            
            
            for (utterance_type, sentence) in self.state["utterance_with_corresponding_types"]:
                # Check if sentence is a substring of model choices
                for model_sent in result["Choices"]:
                    if model_sent in sentence:
                        most_persuasive_disambiguated_sentences.append((utterance_type, sentence))

            log(
                    session_id=self.state["session_id"],
                    level=LogLevels.INFO,
                    service=self._choose_the_most_persuasive_sentences.__name__,
                    message=f"'DisambiguationAgent': reached maximum allowed sentences, succussfully chose most persuasive sentences, completion returned and validated from MODEL = '{self.state['speaker_model_name'].value}'.",
                    mode=self.state["mode"]
                )
        except ValueError as e:
            log(
                    session_id=self.state["session_id"],
                    level=LogLevels.ERROR,
                    service=self._choose_the_most_persuasive_sentences.__name__,
                    message=f"'DisambiguationAgent': reached maximum allowed sentences, could not choose most persuasive sentences, completion not validated; MODEL = '{self.state['speaker_model_name'].value}'; \n REASON: \n\n {e}",
                    mode=self.state["mode"],
                    context={"prompt" : prompt, "state" : self.state, "exception" : e}
                )

        return list(set(most_persuasive_disambiguated_sentences))

    def return_disambiguated_sentences(self) -> GenerationAgentsState:
        """
        Returns the filtered list of sentences after disambiguation.

        Returns:
            dict: {"utterance_with_corresponding_types": filtered_sentences}
        """
        return {
            "utterance_with_corresponding_types" : list(set(self.sentences_with_utterance_types))
        }