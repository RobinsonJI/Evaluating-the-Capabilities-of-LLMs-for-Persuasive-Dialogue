from persuasio.models.models import GenerateLLMResponses
from persuasio.states.state import GenerationAgentsState
from persuasio.prompts.generators.dialogue_moves import MASPromptGenerator
from persuasio.datatypes.enums import LogLevels
from persuasio.utils.logs import log_class, log

@log_class
class DialogueMoveGenerator:
    """
    Generates model responses for various dialogue moves such as Claim, Why, Concede, etc.
    """

    def __init__(self, state: GenerationAgentsState):
        self.state = state

    def generate(self, prompt_method_name : str, schema, output_key : str) -> GenerationAgentsState:
        """
        Generic generator for dialogue moves.
        """
        results = []

        # Essentially runs 'PromptGenerator(self.state).prompt_method_name()
        prompts = getattr(MASPromptGenerator(self.state), prompt_method_name)()

        for prompt in prompts:

            completions = {}
            try:
                result = GenerateLLMResponses(
                    model_choice=self.state["speaker_model_name"],
                    prompt = prompt[:2],
                    temperature=self.state["model_temp"],
                    top_p= self.state["model_top_p"],
                    seed= self.state["model_seed"],
                    datatype_schema=schema
                ).return_completion()
                completions[output_key] = result[output_key.strip("_")]

                results.append([completions,prompt[2],prompt[3]]) 
                log(
                    session_id=self.state["session_id"],
                    level=LogLevels.INFO,
                    service=self.generate.__name__,
                    message=f"Completion returned and validated from MODEL = '{self.state['speaker_model_name'].value}'.",
                    mode=self.state["mode"]
                )

            except ValueError as e:
                log(
                    session_id=self.state["session_id"],
                    level=LogLevels.ERROR,
                    service=self.generate.__name__,
                    message=f"Completion not validated from generation agent; MODEL = '{self.state['speaker_model_name'].value}'; \n REASON: \n\n {e}",
                    mode=self.state["mode"],
                    context={"exception" : e, "state" : self.state, "prompt" : prompt[:2]}
                )

        return {
            "intermediate_generations" : results,
            }