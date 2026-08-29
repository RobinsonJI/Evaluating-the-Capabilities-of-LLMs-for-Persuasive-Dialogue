from persuasio.states.state import GenerationAgentsState
from persuasio.datatypes.enums import SpeakerType

from persuasio.prompts.system.generation_agents import (
    create_claim_negation_system_prompt,
    create_why_claim_system_prompt,
    create_question_claim_system_prompt,
    create_concede_claim_system_prompt,
    create_since_claim_system_prompt,
    create_claim_system_prompt,
    create_retract_claim_system_prompt
)

from persuasio.prompts.system.persuasiveness_choice import (
    create_persuasiveness_choice_system_prompt_for_initial_claim,
    create_persuasiveness_choice_system_prompt
)

from persuasio.prompts.system.generic_rules import rules_for_all_models
from persuasio.utils.logs import log_class

@log_class
class MASPromptGenerator:
    """
    """

    def __init__(self, state : GenerationAgentsState):
        """
        """

        self.state = state

        current_speaker_name = self.state["speaker"]
        opponent_speaker_name = self.state["opponent_speaker_name"]

        self.current_speaker_commitments = self.state.get("commitments", [])
        self.current_speaker_commitments = "\n".join(self.current_speaker_commitments)

        self.current_speaker_original_claim = self.state.get("original_claim", "")

        self.opponents_commitments = self.state.get("opponents_commitments", [])
        self.opponents_commitments = "\n".join(self.opponents_commitments)

        self.opponents_original_claim = self.state.get("opponents_original_claim", "")

        # Getting the conversation history
        combined = []
        for index, dialogue_data in enumerate(self.state.get("dialogue_history", [])):
            _role = dialogue_data.speaker
            claims = dialogue_data.sentences_with_utterance_types
            if current_speaker_name == _role:
                role = "model"
            else:
                role = _role
            flattened = " ".join([item for sublist in claims for item in sublist])
            combined.append(f"Dialogue turn {index+1}, {role}: {flattened}")
        
        self.conversation_string = "\n".join(combined)


    def claim_negation(self):
        """
        Creates a list of prompts which will be used to task a LLM to generate utterances that conflict with a sentence the user has previously said. 
        """

        prompts = []

        sys_msg = create_claim_negation_system_prompt(state=self.state)

        

        # Check to see if rag was chosen as an option
        if self.state["speaker_type"] == SpeakerType.MAS_RAG:
            # Create an empty list to store the system message and the Graph RAG examples.
            sys_msg_and_graph_rag_examples = []
            # If a claim negation example is in the dictionary of Graph RAG examples, then append the example for a given sentence to the system prompt
            if "___NotClaim___" in self.state["graph_rag_examples"][-1]:
                # Iterate through the Graph RAG examples
                for example in self.state["graph_rag_examples"][-1]["___NotClaim___"]:
                    # Create a new system prompt with Graph RAG examples and append the list
                    sys_msg_and_graph_rag_examples.append(
                        {"role" : "system",
                         "content" : sys_msg["content"] + example}
                        )

        index = 0
        # Iterate through the dictionary of typical replies to each sentence that the user has previously uttered.
        for sentence in self.state.get("typical_replies_for_last_speakers_response", []):
            if "___NotClaim___" in sentence[0]:
                # if claim negation is a typical response, then create a claim negation prompt
                if sentence[0]["___NotClaim___"]:
                    
                    human_msg = {"role" : "user",
                                "content" : """Generate 3 sentences that are opposed to: '""" +sentence[0]["___NotClaim___"] +"""'
                                
                                Do not write an introduction or summary. Output in JSON format using the following template: {'Claim' : ['sentence1', 'sentence2', 'sentence3']}
                                
                                Respond only with valid JSON. Do not write an introduction or summary."""#)
                    }

         
                    if self.state["speaker_type"] == SpeakerType.MAS_RAG:
                        prompts.append([sys_msg_and_graph_rag_examples[index], human_msg,sentence[1],sentence[2]])
                        index += 1
                    else:
                        prompts.append([sys_msg, human_msg,sentence[1],sentence[2]]) 

        
        return prompts
    

    def why_claim(self):
        """
        Generates a list of prompts that will be used to challenge a sentence that the user has said, by asking why, in the last dialogue turn.
        """

        prompts = []

        sys_msg = create_why_claim_system_prompt(state=self.state)

        # Check to see if rag was chosen as an option
        if self.state["speaker_type"] == SpeakerType.MAS_RAG:
            # Create an empty list to store the system message and the Graph RAG examples.
            sys_msg_and_graph_rag_examples = []
            # If a Why example is in the dictionary of Graph RAG examples, then append the example for a given sentence to the system prompt
            if "___Why___" in self.state["graph_rag_examples"][-1]:
                # Iterate through the Graph RAG examples
                for example in self.state["graph_rag_examples"][-1]["___Why___"]:
                    # Create a new system prompt with Graph RAG examples and append the list
                    sys_msg_and_graph_rag_examples.append(
                        {"role" : "system",
                         "content" : sys_msg["content"] + example}
                        )
                    
        index = 0
        # Iterate through the dictionary of typical replies to each sentence that the user has previously uttered.
        for sentence in self.state.get("typical_replies_for_last_speakers_response", []):
            # Check to see if a ___Why___ is in the dictionary of response for a given sentence
            if "___Why___" in sentence[0]:
                if sentence[0]["___Why___"]:
                    
                    human_msg = {"role" : "user",
                                "content" : """Generate 3 'why' questions for the sentence: """ +sentence[0]["___Why___"] +"""
                                
                                Do not write an introduction or summary. Output in JSON format using the following template: {'Why' : ['Why' + 'response1', 'Why' + 'response2', 'Why' + 'response3']}
                                
                                Respond only with valid JSON. Do not write an introduction or summary."""#)
                    }

     
                    if self.state["speaker_type"] == SpeakerType.MAS_RAG:
                        prompts.append([sys_msg_and_graph_rag_examples[index], human_msg,sentence[1],sentence[2]])
                        index += 1
                    else:
                        prompts.append([sys_msg, human_msg,sentence[1],sentence[2]]) 

        return prompts
    

    def question_claim(self):
        """
        Generates a list of prompts which will be used to task an LLM with asking the user journalistic questions -- i.e. 'Who', 'What', 'When', 'Where', 'Why', or 'How' -- about something they have previously said. 
        """

        prompts = []

        sys_msg = create_question_claim_system_prompt(state=self.state)

        # Check to see if rag was chosen as an option
        if self.state["speaker_type"] == SpeakerType.MAS_RAG:
            # Create an empty list to store the system message and the Graph RAG examples.
            sys_msg_and_graph_rag_examples = []
            # If a ___Question___ example is in the dictionary of Graph RAG examples, then append the example for a given sentence to the system prompt
            if "___Question___" in self.state["graph_rag_examples"][-1]:
                # Iterate through the Graph RAG examples
                for example in self.state["graph_rag_examples"][-1]["___Question___"]:
                    # Create a new system prompt with Graph RAG examples and append the list
                    sys_msg_and_graph_rag_examples.append(
                        {"role" : "system",
                         "content" : sys_msg["content"] + example}
                        )
                    
        index = 0
        # Iterate through the sentences and typical response types 
        for sentence in self.state.get("typical_replies_for_last_speakers_response", []):
            # Check to see if ___Question___ is in the set of responses
            if "___Question___" in sentence[0]:
                if sentence[0]["___Question___"]:
                    

                    human_msg = {"role" : "user",
                                "content" : """Generate 3 journalistic questions for the sentence: """ +sentence[0]["___Question___"] +"""
                                
                                Do not write an introduction or summary. Output in JSON format using the following template: {'Question' : ['response1', 'response2', 'response3']}
                                
                                Respond only with valid JSON. Do not write an introduction or summary."""
                    }

         
                    if self.state["speaker_type"] == SpeakerType.MAS_RAG:
                        prompts.append([sys_msg_and_graph_rag_examples[index], human_msg,sentence[1],sentence[2]])
                        index += 1
                    else:
                        prompts.append([sys_msg, human_msg,sentence[1],sentence[2]]) 

        return prompts
    

    def concede_claim(self):
        """
        Generates a list of prompts that will be used to concede to a Claim or Since move that the user has previously asserted.
        """
        prompts = []

        sys_msg = create_concede_claim_system_prompt(state=self.state)

        # Check to see if rag was chosen as an option
        if self.state["speaker_type"] == SpeakerType.MAS_RAG:
            # Create an empty list to store the system message and the Graph RAG examples.
            sys_msg_and_graph_rag_examples = []
            # If a ___Concede___ example is in the dictionary of Graph RAG examples, then append the example for a given sentence to the system prompt
            if "___Concede___" in self.state["graph_rag_examples"][-1]:
                # Iterate through the Graph RAG examples
                for example in self.state["graph_rag_examples"][-1]["___Concede___"]:
                    # Create a new system prompt with Graph RAG examples and append the list
                    sys_msg_and_graph_rag_examples.append(
                        {"role" : "system",
                         "content" : sys_msg["content"] + example}
                        )
                    
        index = 0
        for sentence in self.state.get("typical_replies_for_last_speakers_response", []):
            if "___Concede___" in sentence[0]:
                if sentence[0]["___Concede___"]:

                    human_msg = {"role" : "user",
                                "content" : """Generate 3 sentences that concede to the human user's sentence: """ +sentence[0]["___Concede___"] +"""

Do not write an introduction or summary. Output in JSON format using the following template: {'Concede' : ['sentence1', 'sentence2', 'sentence3']}

Respond only with valid JSON. Do not write an introduction or summary."""
                    }

         
                    if self.state["speaker_type"] == SpeakerType.MAS_RAG:
                        prompts.append([sys_msg_and_graph_rag_examples[index], human_msg,sentence[1],sentence[2]])
                        index += 1
                    else:
                        prompts.append([sys_msg, human_msg,sentence[1],sentence[2]]) 

        return prompts
    

    def since_claim(self):
        """
        Generates a list of prompts which forces the model to generate premises that support an utterance that it previously stated.
        """

        prompts = []

        sys_msg = create_since_claim_system_prompt(state=self.state)            

        # Check to see if rag was chosen as an option 
        if self.state["speaker_type"] == SpeakerType.MAS_RAG:
            # Create an empty list to store the system message and the Graph RAG examples.
            sys_msg_and_graph_rag_examples = []
            # If a ___Since___ example is in the dictionary of Graph RAG examples, then append the example for a given sentence to the system prompt
            if "___Since___" in self.state["graph_rag_examples"][-1]:
                # Iterate through the Graph RAG examples
                for example in self.state["graph_rag_examples"][-1]["___Since___"]:
                    # Create a new system prompt with Graph RAG examples and append the list
                    sys_msg_and_graph_rag_examples.append(
                        {"role" : "system",
                         "content" : sys_msg["content"] + example}
                        )

        index = 0
        for sentence in self.state.get("typical_replies_for_last_speakers_response", []):
            if "___Since___" in sentence[0]:
                if sentence[0]["___Since___"]:
                    human_msg = {"role" : "user",
                                "content" : """Generate 3 sentences that answer the following question: """ + sentence[0]["___Since___"] + """
                                
                                Do not write an introduction or summary. Output in JSON format using the following template: {'Since' : ['sentence1', 'sentence2', 'sentence3']}
                                
                                Respond only with valid JSON. Do not write an introduction or summary."""
                    }
                    
         
                    if self.state["speaker_type"] == SpeakerType.MAS_RAG:
                        prompts.append([sys_msg_and_graph_rag_examples[index], human_msg,sentence[1],sentence[2]])
                        index += 1
                    else:
                        prompts.append([sys_msg, human_msg,sentence[1],sentence[2]]) 

        return prompts
    

    def claim(self):
        """
        Generates a list of prompts (str) which forces models to state a claim in response to either a ___Why___ or ___Question___ move.
        """

        prompts = []

        sys_msg = create_claim_system_prompt(state=self.state)            

        # Check to see if rag was chosen as an option 
        if self.state["speaker_type"] == SpeakerType.MAS_RAG:
            # Create an empty list to store the system message and the Graph RAG examples.
            sys_msg_and_graph_rag_examples = []
            # If a ___Claim___ example is in the dictionary of Graph RAG examples, then append the example for a given sentence to the system prompt
            if "___Claim___" in self.state["graph_rag_examples"][-1]:
                # Iterate through the Graph RAG examples
                for example in self.state["graph_rag_examples"][-1]["___Claim___"]:
                    # Create a new system prompt with Graph RAG examples and append the list
                    sys_msg_and_graph_rag_examples.append(
                        {"role" : "system",
                         "content" : sys_msg["content"] + example}
                        )
                                        
        index = 0
        for sentence in self.state.get("typical_replies_for_last_speakers_response", []):
            if "___Claim___" in sentence[0]:
                if sentence[0]["___Claim___"]:

                    human_msg = {"role" : "user",
                                "content" : """Generate 3 claims or sentences that answer the following question: '""" + sentence[0]["___Claim___"] + """'
                                            
                                Do not write an introduction or summary. Output in JSON format using the following template: {'Claim' : ['sentence1', 'sentence2', 'sentence3']}
                                
                                Respond only with valid JSON. Do not write an introduction or summary."""
                    }
                    
         
                    if self.state["speaker_type"] == SpeakerType.MAS_RAG:
                        prompts.append([sys_msg_and_graph_rag_examples[index], human_msg,sentence[1],sentence[2]])
                        index += 1
                    else:
                        prompts.append([sys_msg, human_msg,sentence[1],sentence[2]]) 

        return prompts
    

    def retract_claim(self):
        """
        Generates a list of prompts that force an agent to retract an utterance they previously made.
        """
        prompts = []

        # Getting the conversation history
        if len(self.state["dialogue_history"]) > 1:
            model_last_dialogue_move = " ".join([item[1] for item in self.state["dialogue_history"][-2].sentences_with_utterance_types])
        else:
            model_last_dialogue_move = ""

        sys_msg = create_retract_claim_system_prompt(state=self.state,
                                                     model_last_dialogue_move=model_last_dialogue_move)            

        

        """
        Retract moves don't currently have any examples of retraction through Graph RAG because QT30 doesn't contain this... perhaps we could use some simple vector-embedding RAG for this??
        """

        # Iterate through the list of typical replies to the last utterance.
        for sentence in self.state.get("typical_replies_for_last_speakers_response", []):
            if "___Retract___" in sentence[0]:
                if sentence[0]["___Retract___"]:
                    
                    human_msg = {"role" : "user",
                                "content" : """Generate 3 sentences that retract what you said or communicate that you are not committed to the claim implied in the question.
                                
                                The human user has asserted: """ + sentence[0]["___Retract___"] + """

                                You said: 
                                """+model_last_dialogue_move+"""

                                Your response should be one sentence in length. Do not write an introduction or summary. Output in JSON format using the following template: {'Retract' : ['sentence1', 'sentence2', 'sentence3']}
                                
                                Respond only with valid JSON. Do not write an introduction or summary."""
                    }
                    
                    prompts.append([sys_msg, human_msg,sentence[1],sentence[2]])

        return prompts
    

    def persuasiveness_choice(self):

        prompts = []

        # This conditional executes if the LLM has generated an initial claim and the human user has not made any utterances yet.
        if len(self.state["dialogue_history"]) == 0:

            sys_msg = create_persuasiveness_choice_system_prompt_for_initial_claim(state=self.state)

            sentences = "\n".join(self.state["intermediate_generations"][0][0]["___Claim___"])
            
            human_msg = {"role" : "user",
                        "content" : """Choose the most persuasive sentence out of the following sentences.                            
                        Sentences:
                        """ + sentences + """

                        Do not write an introduction or summary. Output in JSON format using the following template: {'Choice' : string}
                        """
            }
            
            prompts.append([sys_msg, human_msg, True, True])  # True to monitor when first turn is internal
        
            return prompts

        for typical_responses in self.state.get("typical_replies_for_last_speakers_response", []):
            # Monitors which turn of the dialogue we are on
            dialogue_turn_index = typical_responses[1]
            # Indexes of the sentence(s) that were provided by the user.
            sentence_index = typical_responses[2]
            # Create an empty list which will stored previous generation for each sentence and be converted to a string for the prompt
            generations_as_string = []
            generations_with_utterance_types = []
            # Iterate through the generations
            for intermediate_generations in self.state["intermediate_generations"]:
                # If the dialogue turn index matches the 'intermediate_generations' dialogue turn index 
                # and the sentence index matches the 'intermediate_generations' sentence index, 
                # then append sentences to the list of 'generations' and relevant generations
                if (dialogue_turn_index == intermediate_generations[1]) and (sentence_index == intermediate_generations[2]):
                    key = list(intermediate_generations[0].keys())[0]
                    generations_as_string.extend([key + " " + sent for sent in list(*intermediate_generations[0].values())]) # This will be added to the prompt as a string
                    generations_with_utterance_types.extend([[key, sent] for sent in list(*intermediate_generations[0].values())])

            sys_msg = create_persuasiveness_choice_system_prompt(state=self.state)

            # Convert the generations list to a string of sentences separated by newlines
            sentences = "\n".join(generations_as_string)
            
            human_msg = {"role" : "user",
                        "content" : """Choose 1 to 5 sentences that you think are the most persuasive.                            
                        Sentences:
                        """ + sentences
            }
            
            prompts.append([sys_msg, human_msg, generations_as_string, generations_with_utterance_types])
        
        return prompts
    

    def initial_llm_claim(self):

        self.vec_rag_persona_examples = ""
        if self.state["speaker_type"] == SpeakerType.MAS_RAG:
            self.vec_rag_persona_examples += self.state["vector_based_persona_string"]

        sys_msg ={
            "role" : "system",
            "content" : f"""# Task 
            
Your task is to generate 3 sentences that make a SNAPPY (not waffly) claim based on the topic and your political beliefs to start the debate.

{self.vec_rag_persona_examples}

# Rules

- Each response must be exactly one sentence.
- The sentence should be opinionated.
{rules_for_all_models}


"""

        }

        human_msg = {
            "role" : "user",
            "content" : f"""Generate three claims that are ONE sentence in length using the following topic: '{self.state["debate_topic"]}'."""
        }

        # Set these to zero because LLM is making the inital claim which should be one sentence in length
        dialogue_length_index = 0
        sentence_length_index = 0

        prompts = [[sys_msg, human_msg, dialogue_length_index, sentence_length_index]]

        return prompts