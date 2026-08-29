from typing import List, Dict

from persuasio.states.state import BaseModelState
from persuasio.models.models import GenerateLLMResponses
from persuasio.datatypes.pydantic_basemodels import BaseModelCompletion
from persuasio.datatypes.enums import LogLevels
from persuasio.prompts.generators.base_model import create_base_model_prompt
from persuasio.utils.logs import log_class, log

@log_class
class BaseModelAgent:
    """
    Agent responsible for generating responses from a base LLM model.

    The BaseModelAgent class takes a BaseModelState state instance, constructs a prompt, 
    queries a language model, and stores the generated response.
    It acts as a wrapper that encapsulates the prompt construction 
    and model interaction, returning an utterance which will then be classified by another agent for utterance types.

    Attributes
    ----------
    state : BaseModelState
        The current state of the agent, containing configuration and context.
    prompt : str
        The generated prompt based on the agent state.
    base_model_utterance : str
        The response returned by the base language model.
    """
    def __init__(self, state : BaseModelState):

        """
        Initialise the state, prompt, and generate utterance.

        Parameters
        ----------
        state : BaseModelState
            The state object containing model configuration, prompt context, 
            and other metadata needed for response generation.
        """
        
        self.state = state
        # Construct the prompt based on the given state
        self.prompt = self._create_prompt()

        # Generate a response from the base model
        self.base_model_utterance = self._generate_response()


    def _create_prompt(self) -> List[Dict[str, str]]:
        """
        Construct a prompt for the basemodel using the current state.

        Returns
        -------
        List[Dict[str, str]]
            The prompt.
        """
        
        prompt = create_base_model_prompt(state=self.state)

        return prompt

    def _generate_response(self):
        """
        Generate a response from the base language model using the constructed prompt.

        Returns
        -------
        str
            The utterance produced by the base model.
        """
        base_model_utterance = None

        result = None
        
        try:
            # Initialise the model response generator with state configuration
            result = GenerateLLMResponses(
                    model_choice=self.state["speaker_model_name"],  # which model to use
                    prompt = self.prompt,                           # input prompt
                    temperature=self.state["model_temp"],           # randomness control
                    top_p= self.state["model_top_p"],               # nucleus sampling parameter
                    seed= self.state["model_seed"],                 # reproducibility
                    datatype_schema=BaseModelCompletion             # enforce response schema
                ).return_completion()
            
            # Extract the textual response from the model output
            base_model_utterance = result["response"]
            log(
                session_id=self.state["session_id"],
                level=LogLevels.INFO,
                service=self._generate_response.__name__,
                message=f"'BaseModelAgent' generated a response; MODEL = '{self.state['speaker_model_name'].value}'.",
                mode=self.state["mode"]
            )
        except ValueError as e:
            log(
                session_id=self.state["session_id"],
                level=LogLevels.ERROR,
                service=self._generate_response.__name__,
                message=f"'BaseModelAgent' could not generate a response, completion not validated; MODEL = '{self.state['speaker_model_name'].value}'; \n REASON: \n\n {e}",
                mode=self.state["mode"],
                context={"prompt" : self.prompt, "state":self.state, "exception":e, "model" : self.state['speaker_model_name'].value}
            )

        return base_model_utterance

    def return_base_model_response(self) -> Dict[str, str]:
        """
        Return the base model's generated utterance to the BaseModelState.

        Returns
        -------
        dict
            A dictionary containing the base model utterance, structured as:
            {
                "utterance": <model response string>
            }
        """
        return {
            "utterance" : self.base_model_utterance
        }