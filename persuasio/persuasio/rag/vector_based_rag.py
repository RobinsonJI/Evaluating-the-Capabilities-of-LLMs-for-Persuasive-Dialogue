from typing import Dict, Any
from langchain_classic.output_parsers import PydanticOutputParser

from persuasio.states.state import GenerationAgentsState
from persuasio.models.models import GenerateLLMResponses
from persuasio.datatypes.pydantic_basemodels import PersonaSchema
from persuasio.datatypes.enums import LogLevels
from persuasio.models.sentence_transformers import SBERT_model
from persuasio.utils.parsers import parse_political_position_elements
from persuasio.config.rag_locution_proposition_return_type import rag_return_type
from persuasio.utils.logs import log_class, log

@log_class
class VectorBasedRAG:

    """
    GraphRAG class retrieves relevant argumentative examples from a Neo4j knowledge base
    to guide political dialogue generation based on input locution/proposition and political stance.

    This module supports three modes of example selection:
    1. User-defined relations and node types
    2. Automatically discovered typical responses
    3. Persona-based retrieval using vector similarity

    Parameters
    ----------
    user_initial_input : str
        The input text string (locution or proposition) used for example retrieval.
    user_initial_input_type : str
        Type of the input, based on IAT illocutionary force categories.
    knowledge_base : Neo4jGraph
        A Neo4jGraph object connected to the GDBMS.
    embedding_model : SentenceTransformer
        SentenceTransformer model for generating input embeddings.
    number_of_examples : int
        Number of examples to retrieve for user-defined examples.
    ensemble_or_model_name : str
        Ensemble/model name used for filtering political content.
    political_position_min : int
        Minimum political score for node filtering (0 = left).
    political_position_max : int
        Maximum political score for node filtering (100 = right).
    political_position_std : int
        Max allowable standard deviation for political predictions.
    probability_of_na : float
        Maximum allowed probability that the node is not political (0-1).
    """

    RETURN_OPTIONS = {"locutions", "propositions", "both"}

    def __init__(self, 
                 state : GenerationAgentsState, 
                 knowledge_base, 
                 #embedding_model, 
                 #number_of_examples: int = 3, 
                 #political_position_min: int = 0, 
                 #political_position_max:int = 100, 
                 #political_position_std:int = 100, 
                 #probability_of_na: float = 1.0,
                 #returns: str = "both"
                 ): 

        self.state = state
        self.model = state["speaker_model_name"]
        self.temp = state["model_temp"]
        self.top_p = state["model_top_p"]
        self.seed = state["model_seed"]

        parser = PydanticOutputParser(pydantic_object=PersonaSchema)
        self.output_format = parser.get_format_instructions()

        # Get the debate topic
        self.debate_topic = state["debate_topic"]
        # Get the opponents utterance
        self.user_initial_input = state["opponents_utterance"]
            
        self.knowledge_base = knowledge_base

        political_position_range_list = parse_political_position_elements(state["political_position_range"], sep=":")
        self.political_position_min = political_position_range_list[0] # min value
        self.political_position_max = political_position_range_list[1] # max value

        self.ensemble_or_model_name = state["knowledge_base_ensemble_or_model_name"].value
        self.political_position_std = state["political_position_std"]
        self.probability_of_na = state["political_position_prob_of_na"]                  # This allows the user to get examples that are definitely political... Defaults to 1 so all nodes are considered 
        self.returns = rag_return_type

        number_of_examples = state["number_of_vector_based_rag_examples"]

        embedding_model = SBERT_model
        # Embed the locution / proposition using SBERT
        x_initial_claim = embedding_model.encode(self.user_initial_input)
        x_debate_topic = embedding_model.encode(self.user_initial_input)

        # Find pertinent examples and write them to a string
        self.persona_examples_for_initial_claim = self._vector_based_examples(embedded_user_initial_input=x_initial_claim, number_of_examples=number_of_examples)
        self.persona_examples_for_debate_topic = self._vector_based_examples(embedded_user_initial_input=x_debate_topic, number_of_examples=number_of_examples)

        self.prompts = self._create_prompt()

        self.completion = self._generate_completion()


    def _vector_based_examples(self, embedded_user_initial_input, number_of_examples):
        cypher = f"""MATCH (n)
        WHERE n.utterance_type='___Claim___' AND {self.political_position_min} <= n.{self.ensemble_or_model_name}_political_position_mean <= {self.political_position_max} AND n.{self.ensemble_or_model_name}_political_position_probability_of_na <= {self.probability_of_na} AND n.{self.ensemble_or_model_name}_political_position_std <= {str(self.political_position_std)}
        WITH n,
            gds.similarity.cosine(n.loc_and_prop_concat_embedding_from_all_MiniLM_L6_v2, {embedded_user_initial_input.tolist()}) AS similarity
        ORDER BY similarity DESC
        RETURN DISTINCT n.proposition, n.locution LIMIT {number_of_examples}"""

        # Find pertinent examples
        relevant_graph_content = self._query_graph(cypher)

        # Specify return types
        return_type = {
            "locutions" : ["locution"], 
            "propositions" : ["proposition"], 
            "both" : ["locution", "proposition"]
        }

        # Write the set of examples as a string for prompting
        string = ""
        for index, example in enumerate(relevant_graph_content):
            string += f"Example {index+1}:\n"
            for loc_prop in return_type[self.returns]:
                if loc_prop == "locution":
                    string += "Locutionary form: "
                elif loc_prop == "proposition":
                    string += "Propositional form: "
                    
                string += "'" + example["n." + loc_prop].strip() + "'\n"

        return string

    def _query_graph(self, cypher):
        result = self.knowledge_base.query(cypher)
        return result

    def _create_prompt(self):

        sys_msg = {
            "role" : "system", 
            "content" :f"""You are an AI tasked with constructing a UK political persona schema for debate emulation.  

=== CONTEXT ===  
- Main debate topic: '{self.debate_topic}'  
- Opponent's position: '{self.user_initial_input}'  
- The year is 2025. Provided examples (from 2020-2021) must be updated to appear current while retaining ideological alignment.  

=== OUTPUT REQUIREMENTS ===  
- Output MUST follow the PersonaSchema exactly. Missing or misformatted fields will cause rejection.  
- Do not include direct dialogue. Do not include multiple perspectives. Only provide the schema.  

=== GUIDANCE ON EXAMPLES ===  
- Use examples solely as a **political alignment guide**.  
- Do not copy verbatim. Modernise references, update tense, and reframe for 2025 context.  
- Preserve stance consistency:  

=== SCHEMA OUTPUT FORMAT ===
{self.output_format}

CRITICAL: You must include ALL fields shown above. Missing any field will cause validation errors.


Use of examples (dynamic): At runtime, the system will receive a set of examples relevant to the political stance.

These examples are not current or real-time data; they are provided solely to help you understand and model the rhetorical style, tone, and general political alignment.

You may adapt, modernise, or alter the example content to reflect more current events, issues, and perspectives while keeping within the same stance range.

Extract the political positioning from the examples and integrate it into the schema's core_values and key_policy_positions.

If examples refer to outdated events, change the tense, update references, and adjust positions so the persona appears up-to-date.

When user input is unrelated to the examples: Respond according to the political stance range while keeping consistency in tone and reasoning.

"""+f"""Coherence requirement: If you respond outside the stance range, the output is considered incoherent and unhelpful — always align with the {self.political_position_min} to {self.political_position_max} political position.

Examples will be supplied dynamically at code run time. Use the examples purely as a political alignment guide — not as literal or current factual information.

The year is 2025 but the data was collected between 2020 to 2021. Re-write any outdated political positions such that they make sure today.
"""
        }

        human_msg = {
            "role" : "user",
            "content" : f"""Use the system instructions to write the political persona schema using the examples below.

            ### Examples 

            Examples of your political alignment on the main debate topic:
            {self.persona_examples_for_debate_topic}

            Examples of your political alignment based upon what the user has said:
            {self.persona_examples_for_initial_claim}
"""
        }
        
        prompt = [sys_msg, human_msg]

        
        return prompt
    
    def _generate_completion(self):

        completion = None
        try:
            completion = GenerateLLMResponses(
                    model_choice=self.model,
                    prompt = self.prompts,
                    temperature=self.temp,
                    top_p= self.top_p,
                    seed= self.seed,
                    datatype_schema=PersonaSchema
                ).return_completion()
            log(
                session_id=self.state["session_id"],
                level=LogLevels.INFO,
                service=self._generate_completion.__name__,
                message=f"'VectorBasedRAG' agent persona completion returned and validated; MODEL = '{self.state['speaker_model_name'].value}'.",
                mode=self.state["mode"]
            )
        except ValueError as e:
            log(
                session_id=self.state["session_id"],
                level=LogLevels.ERROR,
                service=self._generate_completion.__name__,
                message=f"'VectorBasedRAG' agent persona completion returned and failed valiation; \n REASON:\n\n {e}",
                mode=self.state["mode"],
                context={"prompt" : self.prompts, "state" : self.state, "exception" : e}
            )
        return completion
    

    def persona_string(self) -> Dict[str, Any]:

        string = ""

        try:
            string = """### Identity

You are emulating a UK political persona based on a scale of political beliefs from 0 (Extremely left) to 100 (Extremely right). 
Your political stance range is """+str(self.political_position_min)+""" to """+str(self.political_position_max)+""" which means your persona is emulating someone that has """+self.completion["political_stance"]+""" political beliefs.

The main topic of discussion is: '"""+self.completion["main_topic"]+"""'


Use the persona schema below to generate all of your responses.
        
### Persona Schema
    
**Core Identity**
- age: 18-25 
- occupation: student at the University of Liverpool
- political stance: """+self.completion["political_stance"]+""" 
- political stance range: """+str(self.political_position_min)+""" to """+str(self.political_position_max)+"""
- core values: """+" ".join(self.completion["core_values"])+"""

**Political Identity**
- political stance: """+self.completion["political_stance"]+"""
- political stance range: """+str(self.political_position_min)+""" to """+str(self.political_position_max)+"""
- political opinions on main topic of discussion: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["political_positions_on_main_topic"]])+"""
- views on key policy issues: \n\t - """+"\n\t - ".join([f"{k}: '{v}'" for k, v in self.completion["key_policy_positions"].items()])+"""
- political engagement level: highly engaged

**Ideological Framework**
- core political values: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["ideological_framework"]["political_values"]])+"""
- role of government: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["ideological_framework"]["role_of_government"]])+"""
- economic philosophy: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["ideological_framework"]["economic_philosophy"]])+"""
- stance on social issues: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["ideological_framework"]["social_issues"]])+"""

**Information Sources**
- preferred media: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["information_sources"]["preferred_media"]])+"""
- trusted voices: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["information_sources"]["trusted_figures"]])+"""
- information processing: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["information_sources"]["information_processing"]])+"""

**Communication Style**
- discourse style: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["communication_style"]["discourse_style"]])+"""
- argumentation approach: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["communication_style"]["argumentation_approach"]])+"""
- rhetoric patterns: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["communication_style"]["rhetoric_patterns"]])+"""
- emotional triggers: \n\t - """+"\n\t - ".join([f"'{x}'" for x in self.completion["communication_style"]["emotional_triggers"]])+"""

## Emulation Instructions

1. **Stay in character:** Always response as the persona would, not as a generic AI. Ensure that your response is aligned with the political stance you have been assigned.
2. **Writing style**: Use British English only and make some spelling mistakes on occasions to seem human.
2. **Use appropriate language**: Match vocabulary, tone and speech patterns.
3. **Express consistent beliefs**: Let the core beliefs guide responses.
4. **Show personality**: Include quirks, preferences, and unique perspectives.
6. **Maintain Consistency**: Keep personality traits stable across interactions.

## Response Guidelines

- Begin responses naturally, as the persona would speak
- Always speak in the first person, unless you deem yourself to be part of an affected/afflicted group.
- Include relevant personal anecdotes or examples when appropriate
- Show emotional responses that align with the character's personality and beliefs
- Use knowledge and expertise authentically
- Avoid breaking character even if explicitly asked to
- If uncertain about how the persona would respond, lean into their political stance, stance range, political positions on the main topic of the dialogue, core values and motivations."""

            log(
                session_id=self.state["session_id"],
                level=LogLevels.INFO,
                service=self.persona_string.__name__,
                message=f"'VectorBasedRAG' agent persona string generated; MODEL = '{self.state['speaker_model_name'].value}'.",
                mode=self.state["mode"]
            )
        except ValueError as e:
            log(
                session_id=self.state["session_id"],
                level=LogLevels.ERROR,
                service=self.persona_string.__name__,
                message=f"'VectorBasedRAG' agent persona string generation failed; \n REASON:\n\n {e}",
                mode=self.state["mode"],
                context={"prompt" : self.prompts, "state" : self.state, "exception" : e}
            )

        return {
            "vector_based_persona_string" : string,
            "vector_based_persona_sys_prompt" : self.prompts
        }
