from persuasio.states.state import GenerationAgentsState
from persuasio.models.sentence_transformers import SBERT_model
from persuasio.utils.parsers import parse_political_position_elements
from persuasio.config.rag_locution_proposition_return_type import rag_return_type
from persuasio.utils.logs import log_class

@log_class
class GraphRAG:

    '''
    The Graph RAG module for the persuasion dialogue system. 

    ...

    Parameters
    ----------
    knowledge_base : Neo4jGraph()
        An API connection to the neo4j graph database management system.
    state : dict
        The state dictionary from the LangGraph multi-agent system prior to the LLM's next set of generations.
    embedding_model : SentenceTransformer()
        The embedding model which will be used to convert the user's last set of utterances into SBERT vector embedding. We chose to use the SBERT model 'all-MiniLM-L6-v2' for the embeddings in the GDBMS and also user queries.
    number_of_examples : int
        The number of examples that the GraphRAG module will search for from the GDBMS. Default is 3 examples.
    ensemble_or_model_name : str
        Name of the ensemble/model from which the mean political position will be used. Ensembles are:
            - ensemble1 -> mean, standard deviation and probability of na computed using results from all models
            - ensemble2 -> mean, standard deviation and probability of na computed using results from reasoning models
            - ensemble3 -> mean, standard deviation and probability of na computed using results from models that could distinguish between nodes with political content and those nodes that were NA
    political_position_min : int 
        The minimum mean political position of nodes which will be used to select politically biased examples from the GDBMS.
    political_position_max : int
        The max mean political position of nodes which will be used to select politically biased examples from the GDBMS.
    political_position_std : int
        The amount of standard deviation allowed across predictions for each node. Default is set to 100 but this number can be lowered to choose nodes with a small amount of variance.
    probability_of_na : float
        The probability of a node's locution and proposition not being political. This variable is used to select nodes that have a probability less than the probability of being not applicable. Default is 1.0 but can be assigned a value closer to unity to ensure that the nodes selected by the RAG system have political value.


    Methods
    -------
    '''

    RETURN_OPTIONS = {"locutions", "propositions", "both"}

    def __init__(self, 
                 state : GenerationAgentsState, 
                 knowledge_base, 
                 ):
        
        self.state = state
        self.knowledge_base = knowledge_base
        embedding_model = SBERT_model

        political_position_range_list = parse_political_position_elements(state["political_position_range"], sep=":")
        self.political_position_min = political_position_range_list[0]
        self.political_position_max = political_position_range_list[1]

        self.ensemble_or_model_name = state["knowledge_base_ensemble_or_model_name"].value
        self.political_position_std = state["political_position_std"]
        self.probability_of_na = state["political_position_prob_of_na"]                  # This allows the user to get examples that are definitely political... Defaults to 1 so all nodes are considered 

        number_of_examples = state["number_of_graph_rag_examples"]
        
        self.rag_examples = {
                "___Why___" : [],
                "___NotClaim___" : [],
                "___Concede___" : [],
                "___Question___" : [],
                "___Since___" : [],
                "___Claim___" : []
            }
        
        self.returns = rag_return_type
        self.return_type = {
            "locutions" : ["locution"], 
            "propositions" : ["proposition"], 
            "both" : ["locution", "proposition"]
        }

        # Iterate through the user's input sentence and typical response types
        for typical_responses in self.state["opponents_utterance_with_corresponding_types"]:           
            utterance_type = typical_responses[0]
            sentence = typical_responses[1]
            
            x = embedding_model.encode(sentence)

            # Use the utterance type of the user's sentence to find examples of what the LLMs might say using GraphRAG
            if utterance_type == "___Claim___":
                # Find example according to Prakken's table
                self.rag_examples["___Why___"].append(self.why(x, number_of_examples, from_node_type="Claim"))
                self.rag_examples["___NotClaim___"].append(self.claim_negation(x, number_of_examples, from_node_type = "Claim"))
                self.rag_examples["___Concede___"].append(self.concede(x, number_of_examples, from_node_type="Claim"))
                self.rag_examples["___Question___"].append(self.question(x, number_of_examples, from_node_type="Claim"))

            elif utterance_type == "___Why___":
                self.rag_examples["___Since___"].append(self.since(x, number_of_examples, from_node_type = "Why"))
                self.rag_examples["___Claim___"].append(self.claim(x, number_of_examples, from_node_type = "Why"))
                # Not sure if there are any retract moves in the QT30 dataset
                #self.retract(x, number_of_examples)

            elif utterance_type == "___Since___":
                self.rag_examples["___Why___"].append(self.why(x, number_of_examples, from_node_type="Since"))
                self.rag_examples["___Concede___"].append(self.concede(x, number_of_examples, from_node_type="Since"))
                self.rag_examples["___Question___"].append(self.question(x, number_of_examples, from_node_type="Since"))

            elif utterance_type == "___Question___":
                self.rag_examples["___Claim___"].append(self.claim(x, number_of_examples, from_node_type = "Question"))
                self.rag_examples["___NotClaim___"].append(self.claim_negation(x, number_of_examples, from_node_type = "Question"))
                # Not sure if there are any retract moves in the QT30 dataset
                # self.retract()

    def claim(self, x, number_of_examples, from_node_type):

        if from_node_type == "Why":
            # Claim response to a why/challenge
            cypher = """MATCH (m:Why)-[]->(n:Claim)
                WHERE """+str(self.political_position_min)+""" <= n."""+self.ensemble_or_model_name+"""_political_position_mean <= """+str(self.political_position_max)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_probability_of_na <= """+str(self.probability_of_na)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_std <= """+str(self.political_position_std)+"""
                WITH m, n,
                    gds.similarity.cosine(m.loc_and_prop_concat_embedding_from_all_MiniLM_L6_v2, """+str(x.tolist())+""") AS similarity
                ORDER BY similarity DESC
                RETURN DISTINCT m.proposition, m.locution, n.proposition, n.locution LIMIT """+str(number_of_examples)
            
            relevant_graph_content = self.query_graph(cypher)
            

            if not relevant_graph_content:  
                empty_string = ""
                return empty_string

            string = f"""### Pertinent Examples
            
A user has asked why (or challenged) something you previously said. We have provided {len(relevant_graph_content)} examples of user's claims and your set of typical responses which are in agreement with your political beliefs. Sentences have been annotated into locution and their propositions. Extract the political position from the examples and use them as a guide to help you generate your response. Examples:

"""
            for index, result in enumerate(relevant_graph_content):
                count = index + 1
                string += f"""Example {count}:"""
                for loc_prop in self.return_type[self.returns]:
                    string += f"""The user's challenge {loc_prop} (or why question): '{result["m."+loc_prop]}'
Your answer/{loc_prop}: '{result["n."+loc_prop]}'
"""
                #string +="""The user's challenge proposition (or why question): '{result["m.proposition"]}'
                #Your answer/proposition: '{result["n.locution"]}'

                #"""

            string += """The above examples are to be used as a rough guide. It is imperative that you extract the political position from the examples, even if you don't use the examples, and use that political leaning in your response. The examples provided might refer to past events in the present tense because the data is now a few years old. If so, you should change the tense and the perspective of the examples so that it makes sense in the current dialogue, ensuring that you extract the political leaning from the examples when including them in your response to the user.

"""  

            return string

        
        elif from_node_type == "Question":
            # Claim response to user's question
            cypher = """MATCH (n:Claim)-[:IS_A_REPHRASE_OF]->(m:Question)
                WHERE """+str(self.political_position_min)+""" <= n."""+self.ensemble_or_model_name+"""_political_position_mean <= """+str(self.political_position_max)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_probability_of_na <= """+str(self.probability_of_na)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_std <= """+str(self.political_position_std)+"""
                WITH m, n,
                    gds.similarity.cosine(m.loc_and_prop_concat_embedding_from_all_MiniLM_L6_v2, """+str(x.tolist())+""") AS similarity
                ORDER BY similarity DESC
                RETURN DISTINCT n.proposition, n.locution, m.proposition, m.locution LIMIT """+str(number_of_examples)
            
            relevant_graph_content = self.query_graph(cypher)
            

            if not relevant_graph_content:  
                empty_string = ""
                return empty_string

            string = f"""### Pertinent Examples
            
A user has asked a question about something you previously said. We have provided {len(relevant_graph_content)} examples of user's claims and your set of typical responses which are in agreement with your political beliefs. Sentences have been annotated into locution and their propositions. Extract the political position from the examples and use them as a guide to help you generate your response. Examples:

"""
            for index, result in enumerate(relevant_graph_content):
                count = index + 1
                string += f"""Example {count}:"""
                for loc_prop in self.return_type[self.returns]:
                    string += f"""The user's question as a {loc_prop}: '{result["m."+loc_prop]}'
Your answer/{loc_prop}: '{result["n."+loc_prop]}'
"""
                #The user's question in propositional form: '{result["m.proposition"]}'
                #Your answer/proposition: '{result["n.locution"]}'

                #"""

            string += """The above examples are to be used as a rough guide. It is imperative that you extract the political position from the examples, even if you don't use the examples, and use that political leaning in your response. The examples provided might refer to past events in the present tense because the data is now a few years old. If so, you should change the tense and the perspective of the examples so that it makes sense in the current dialogue, ensuring that you extract the political leaning from the examples when including them in your response to the user.

"""

            return string

    def concede(self, x, number_of_examples, from_node_type):
        
        if from_node_type == "Claim" or from_node_type == "Since":

            # Concede moves after user has made a claim
            cypher = """MATCH (n:Claim) 
            WITH n,
                gds.similarity.cosine(n.loc_and_prop_concat_embedding_from_all_MiniLM_L6_v2, """+str(x.tolist())+""") AS similarity
            ORDER BY similarity DESC
            UNWIND n.prediction_index as index  
            MATCH (m:Concede)-[:SUPPORTS|IS_A_REPHRASE_OF|TRANSITIONS_TO]->(n:Claim {prediction_index: index})
            WHERE """+str(self.political_position_min)+""" <= m."""+self.ensemble_or_model_name+"""_political_position_mean <= """+str(self.political_position_max)+""" AND m."""+self.ensemble_or_model_name+"""_political_position_probability_of_na <= """+str(self.probability_of_na)+""" AND m."""+self.ensemble_or_model_name+"""_political_position_std <= """+str(self.political_position_std)+"""
            RETURN DISTINCT n.proposition, n.locution, m.proposition, m.locution LIMIT """+str(number_of_examples)

            relevant_graph_content = self.query_graph(cypher)
            

            if not relevant_graph_content:  
                empty_string = ""
                return empty_string   

            string = f"""### Pertinent Examples
            
A user has made a claim which you must concede to. We have provided {len(relevant_graph_content)} examples of user's claims and thing you might say to agree with them. Sentences have been annotated into locution and their propositions. Extract the political position from the examples and use them as a guide to help you generate your response. Examples:

"""

            for index, result in enumerate(relevant_graph_content):
                count = index + 1
                string += f"""Example {count}:"""
                for loc_prop in self.return_type[self.returns]:
                    if loc_prop == "locution":
                        string+=f"""User's locution: '{result["n.locution"]}'
Your illocutionary concede response: '{result["m.locution"]}'
"""
                    if loc_prop == "proposition":
                        string+=f"""User's proposition: '{result["n.proposition"]}'
Your propositional concede response: '{result["m.proposition"]}'

"""
                
            string += """The above examples are to be used as a rough guide. It is imperative that you extract the political position from the examples, even if you don't use the examples, and use that political leaning in your response. The examples provided might refer to past events in the present tense because the data is now a few years old. If so, you should change the tense and the perspective of the examples so that it makes sense in the current dialogue, ensuring that you extract the political leaning from the examples when including them in your response to the user.

If an example does not contain agreement, then you should extract the political positions from the examples and include those political positions in your response.

"""    
                
            return string


    def claim_negation(self, x, number_of_examples, from_node_type):
        """
        A method for querying the knowledge base and providing examples of claim negation by using the set of attacks between nodes in the neo4j graph.
        """
        if from_node_type == "Claim":

            cypher = """MATCH (n:Claim) 
            WITH n,
                gds.similarity.cosine(n.loc_and_prop_concat_embedding_from_all_MiniLM_L6_v2, """+str(x.tolist())+""") AS similarity
            ORDER BY similarity DESC
            UNWIND n.prediction_index as index  
            MATCH (m:Claim)-[:ATTACKS]->(n:"""+str(from_node_type)+""" {prediction_index: index})
            WHERE """+str(self.political_position_min)+""" <= m."""+self.ensemble_or_model_name+"""_political_position_mean <= """+str(self.political_position_max)+""" AND m."""+self.ensemble_or_model_name+"""_political_position_probability_of_na <= """+str(self.probability_of_na)+""" AND m."""+self.ensemble_or_model_name+"""_political_position_std <= """+str(self.political_position_std)+"""
            RETURN DISTINCT m.proposition, m.locution, n.proposition, n.locution LIMIT """+str(number_of_examples)
            # The example response here is the proposition & locution from "m" because that is the node that attacks "n"
            relevant_graph_content = self.query_graph(cypher)
            

            if not relevant_graph_content:  
                empty_string = ""
                return empty_string

            string = f"""### Pertinent Examples
            
A user has made a claim and it is your job to negate that claim. We have provided {len(relevant_graph_content)} examples of claim negations pertaining to the topic that you are discussing. Sentences have been annotated into locution and their propositions. Extract the political position from the examples and use them as a guide to help you generate your response. Examples:

"""
            for index, result in enumerate(relevant_graph_content):
                count = index + 1
                string += f"""Example {count}:"""
                for loc_prop in self.return_type[self.returns]:
                    if loc_prop == "locution":
                        string+=f"""User's locution: '{result["n.locution"]}'
Your illocutionary response: '{result["m.locution"]}'
'{result["m.locution"]}' -[:ATTACKS]-> '{result["n.locution"]}'
"""
                    if loc_prop == "proposition":
                        string+=f"""User's proposition: '{result["n.proposition"]}'
Your propositional response: '{result["m.proposition"]}'
'{result["m.proposition"]}' -[:ATTACKS]-> '{result["n.proposition"]}'

"""

            string += """The above examples are to be used as a rough guide. It is imperative that you extract the political position from the examples, even if you don't use the examples, and use that political leaning in your response. The examples provided might refer to past events in the present tense because the data is now a few years old. If so, you should change the tense and the perspective of the examples so that it makes sense in the current dialogue, ensuring that you extract the political leaning from the examples when including them in your response to the user.

"""
            return string

        elif from_node_type == "Question":

            cypher = """MATCH (n:Claim)-[:ATTACKS]->(m:Question) 
            WITH n, m,
                gds.similarity.cosine(m.loc_and_prop_concat_embedding_from_all_MiniLM_L6_v2, """+str(x.tolist())+""") AS similarity
            ORDER BY similarity DESC 
            WHERE """+str(self.political_position_min)+""" <= n."""+self.ensemble_or_model_name+"""_political_position_mean <= """+str(self.political_position_max)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_probability_of_na <= """+str(self.probability_of_na)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_std <= """+str(self.political_position_std)+"""
            RETURN DISTINCT m.proposition, m.locution, n.proposition, n.locution LIMIT """+str(number_of_examples)

            relevant_graph_content = self.query_graph(cypher)
            

            if not relevant_graph_content:  
                empty_string = ""
                return empty_string

            string = f"""### Pertinent Examples
            
You have been asked a question and your task is to generate a conflict with the underlying claim implied by that question. We have provided {len(relevant_graph_content)} examples of claim negations pertaining to the topic that you are discussing. Sentences have been annotated into locution and their propositions. Extract the political position from the examples and use them as a guide to help you generate your response. Examples:

"""
            for index, result in enumerate(relevant_graph_content):
                count = index + 1
                string += f"""Example {count}:"""
                for loc_prop in self.return_type[self.returns]:
                    if loc_prop == "locution":
                        string+=f"""User's question as a locution: '{result["m.locution"]}'
Your locution negating the claim: '{result["n.locution"]}'
'{result["n.locution"]}' -[:ATTACKS]-> '{result["m.locution"]}'
"""
                    if loc_prop == "proposition":
                        string+=f"""User's question as a proposition: '{result["m.proposition"]}'
Your proposition negating the claim '{result["n.proposition"]}'
'{result["n.proposition"]}' -[:ATTACKS]-> '{result["m.proposition"]}'

"""

            string += """The above examples are to be used as a rough guide. It is imperative that you extract the political position from the examples, even if you don't use the examples, and use that political leaning in your response. The examples provided might refer to past events in the present tense because the data is now a few years old. If so, you should change the tense and the perspective of the examples so that it makes sense in the current dialogue, ensuring that you extract the political leaning from the examples when including them in your response to the user.

"""

            return string

    def since(self, x, number_of_examples, from_node_type):
        
        if from_node_type == "Why":
            cypher = """MATCH (m:Why)-[]->(o:Claim)
            WITH m, o,
                gds.similarity.cosine(m.loc_and_prop_concat_embedding_from_all_MiniLM_L6_v2, """+str(x.tolist())+""") AS similarity
            ORDER BY similarity DESC
            UNWIND o.prediction_index as index  
            MATCH (n:Claim)-[:SUPPORTS]->(o:Claim)
            WHERE """+str(self.political_position_min)+""" <= n."""+self.ensemble_or_model_name+"""_political_position_mean <= """+str(self.political_position_max)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_probability_of_na <= """+str(self.probability_of_na)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_std <= """+str(self.political_position_std)+"""
            RETURN DISTINCT m.proposition, m.locution, n.proposition, n.locution, o.proposition, o.locution LIMIT """+str(number_of_examples)

            relevant_graph_content = self.query_graph(cypher)
            

            if not relevant_graph_content:  
                empty_string = ""
                return empty_string
        
            string = f"""You made a claim, the user challenged that claim (i.e. asked a why question), and now you need to make another claim in response to the user's challenge in order to provide more reasons to believe your initial claim. 

We have provided {len(relevant_graph_content)} examples of claims supporting other claims for the topic that you are discussing. Sentences have been annotated into locution and their propositions. Extract the political position from the examples and use them as a guide to help you generate your response. Examples:

"""
            for index, result in enumerate(relevant_graph_content):
                count = index + 1
                string += f"""Example {count}:"""
                for loc_prop in self.return_type[self.returns]:
                    if loc_prop == "locution":
                        string+=f"""Your first locution: '{result["o.locution"]}'
The user's challenge locution (or why question): '{result["m.locution"]}'
Your answer/second supporting locution: '{result["n.locution"]}'
Your argument as two locutions: '{result["o.locution"]}'-[:SUPPORTS]->'{result["n.locution"]}'
"""
                    if loc_prop == "proposition":
                        string+=f"""Your first proposition: '{result["o.proposition"]}'
The user's challenge proposition (or why question): '{result["m.proposition"]}'
Your answer/second supporting proposition: '{result["n.locution"]}'
Your argument in propositional form: '{result["o.proposition"]}'-[:SUPPORTS]->'{result["n.proposition"]}'

"""

            string += """The above examples are to be used as a rough guide. It is imperative that you extract the political position from the examples, even if you don't use the examples, and use that political leaning in your response. The examples provided might refer to past events in the present tense because the data is now a few years old. If so, you should change the tense and the perspective of the examples so that it makes sense in the current dialogue, ensuring that you extract the political leaning from the examples when including them in your response to the user.
     
"""
            return string

    
    def why(self, x, number_of_examples, from_node_type):

        if from_node_type == "Claim":

            cypher = """MATCH (n:Claim) 
            WITH n,
            gds.similarity.cosine(n.loc_and_prop_concat_embedding_from_all_MiniLM_L6_v2, """+str(x.tolist())+""") AS similarity
            ORDER BY similarity DESC
            UNWIND n.prediction_index as index  
            MATCH (m:Why)-[]->(n:Claim {prediction_index: index})
            WHERE """+str(self.political_position_min)+""" <= m."""+self.ensemble_or_model_name+"""_political_position_mean <= """+str(self.political_position_max)+""" AND m."""+self.ensemble_or_model_name+"""_political_position_probability_of_na <= """+str(self.probability_of_na)+""" AND m."""+self.ensemble_or_model_name+"""_political_position_std <= """+str(self.political_position_std)+"""
            RETURN DISTINCT m.proposition, m.locution, n.proposition, n.locution LIMIT """+str(number_of_examples)+""""""

            # The example response here is the proposition & locution from "m" because that is the node that attacks "n"
            relevant_graph_content = self.query_graph(cypher)
            

            if not relevant_graph_content:  
                empty_string = ""
                return empty_string 

            string = f"""### Pertinent Examples
            
A user has made a claim and it is your job to challenge that claim through the generation of a question containing the word 'why'. We have provided {len(relevant_graph_content)} examples of challenges (i.e. why questions) pertaining to the topic that you are discussing. Sentences have been annotated into locution and their propositions. Extract the political position from the examples and use them as a guide to help you generate your response. Examples:

"""

            for index, result in enumerate(relevant_graph_content):
                count = index + 1
                string += f"""Example {count}:"""
                for loc_prop in self.return_type[self.returns]:
                    if loc_prop == "locution":
                        string+=f"""User's locution: '{result["n.locution"]}'
Your illocutionary WHY response: '{result["m.locution"]}'
"""
                    if loc_prop == "proposition":
                        string+=f"""User's proposition: '{result["n.proposition"]}'
Your propositional WHY response: '{result["m.proposition"]}'

"""

            string += """The above examples are to be used as a rough guide. It is imperative that you extract the political position from the examples, even if you don't use the examples, and use that political leaning in your response. The examples provided might refer to past events in the present tense because the data is now a few years old. If so, you should change the tense and the perspective of the examples so that it makes sense in the current dialogue, ensuring that you extract the political leaning from the examples when including them in your response to the user.

If some of the examples do not include the word 'why', then ignore it and only extract the political stance from the example. 
         
"""

            return string

        elif from_node_type == "Since":

            cypher = """MATCH (m:Claim)-[]->(o:Claim)
            WITH m, o,
            gds.similarity.cosine(m.loc_and_prop_concat_embedding_from_all_MiniLM_L6_v2, """+str(x.tolist())+""") AS similarity
            ORDER BY similarity DESC
            UNWIND m.prediction_index as index  
            MATCH (n:Why)-[]->(m:Claim {prediction_index: index})
            WHERE """+str(self.political_position_min)+""" <= n."""+self.ensemble_or_model_name+"""_political_position_mean <= """+str(self.political_position_max)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_probability_of_na <= """+str(self.probability_of_na)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_std <= """+str(self.political_position_std)+"""
            RETURN DISTINCT m.proposition, m.locution, n.proposition, n.locution, o.proposition, o.locution LIMIT """+str(number_of_examples)+""""""

            # The example response here is the proposition & locution from "m" because that is the node that attacks "n"
            relevant_graph_content = self.query_graph(cypher)
            

            if not relevant_graph_content:  
                empty_string = ""
                return empty_string 

            string = f"""### Pertinent Examples 
            
A user has made an argument which is comprised of a claim (or premise) that supports another claim (or conclusion). Your task is to generate a set of why responses that challenge the premise of the argument. We have provided {len(relevant_graph_content)} examples of challenges (i.e. why questions) pertaining to the topic that you are discussing. Sentences have been annotated into locution and their propositions. Extract the political position from the examples and use them as a guide to help you generate your response. Examples:

"""

            for index, result in enumerate(relevant_graph_content):
                count = index + 1
                string += f"""Example {count}:"""
                for loc_prop in self.return_type[self.returns]:
                    if loc_prop == "locution":
                        string+=f"""User's premise as a locution: '{result["m.locution"]}'
User's conclusion as a locution: '{result["o.locution"]}'
User's argument as a locution: '{result["m.locution"]}'-[:SUPPORTS]->'{result["o.locution"]}'
Your why locution to challenge the premise: '{result["n.locution"]}'
"""
                    if loc_prop == "proposition":
                        string+=f"""User's premise as a proposition: '{result["m.proposition"]}'
User's conclusion as a proposition: '{result["o.proposition"]}'
User's argument in propositional form: '{result["m.proposition"]}'-[:SUPPORTS]->'{result["o.proposition"]}'
Your why proposition to challenge the premise: '{result["n.proposition"]}'

"""

            string += """The above examples are to be used as a rough guide. It is imperative that you extract the political position from the examples, even if you don't use the examples, and use that political leaning in your response. The examples provided might refer to past events in the present tense because the data is now a few years old. If so, you should change the tense and the perspective of the examples so that it makes sense in the current dialogue, ensuring that you extract the political leaning from the examples when including them in your response to the user.

If some of the examples do not include the word 'why', then ignore it and only extract the political stance from the example. 

"""

            return string


    def question(self, x, number_of_examples, from_node_type):

        if from_node_type == "Claim":

            cypher = """MATCH (m:Claim)-[:TRANSITIONS_TO|IS_A_REPHRASE_OF|SUPPORTS]->(n:Question)
            WITH n, m,
                gds.similarity.cosine(m.loc_and_prop_concat_embedding_from_all_MiniLM_L6_v2, """+str(x.tolist())+""") AS similarity
            ORDER BY similarity DESC 
            WHERE """+str(self.political_position_min)+""" <= n."""+self.ensemble_or_model_name+"""_political_position_mean <= """+str(self.political_position_max)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_probability_of_na <= """+str(self.probability_of_na)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_std <= """+str(self.political_position_std)+"""
            RETURN DISTINCT m.proposition, m.locution, n.proposition, n.locution LIMIT """+str(number_of_examples)

            relevant_graph_content = self.query_graph(cypher)
            

            if not relevant_graph_content:  
                empty_string = ""
                return empty_string

            string = f"""### Pertinent Examples
            
A user has made a claim and you have been tasked with writing a sentence that questions the user's claim. We have provided {len(relevant_graph_content)} examples questions pertaining to the topic that you are discussing. Sentences have been annotated into locution and their propositions. Extract the political position from the examples and use them as a guide to help you generate your response. Examples:

"""

            for index, result in enumerate(relevant_graph_content):
                count = index + 1
                string += f"""Example {count}:"""
                for loc_prop in self.return_type[self.returns]:
                    if loc_prop == "locution":
                        string+=f"""User's claim as a locution: '{result["m.locution"]}'
Your illocutionary question response: '{result["n.locution"]}'
"""
                    if loc_prop == "proposition":
                        string+=f"""User's claim as a proposition: '{result["m.proposition"]}'
Your propositional question response: '{result["n.proposition"]}'

"""

            string += """The above examples are to be used as a rough guide. It is imperative that you extract the political position from the examples, even if you don't use the examples, and use that political leaning in your response. The examples provided might refer to past events in the present tense because the data is now a few years old. If so, you should change the tense and the perspective of the examples so that it makes sense in the current dialogue, ensuring that you extract the political leaning from the examples when including them in your response to the user.
            
If some of the examples do not include a question, then ignore it and only extract the political stance from the example. 
                
"""

            return string

        elif from_node_type == "Since":

            cypher = """MATCH (m:Claim)-[]->(o:Claim)
            WITH m, o,
            gds.similarity.cosine(m.loc_and_prop_concat_embedding_from_all_MiniLM_L6_v2, """+str(x.tolist())+""") AS similarity
            ORDER BY similarity DESC
            UNWIND m.prediction_index as index  
            MATCH (n:Question)-[]->(m:Claim {prediction_index: index})
            WHERE """+str(self.political_position_min)+""" <= n."""+self.ensemble_or_model_name+"""_political_position_mean <= """+str(self.political_position_max)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_probability_of_na <= """+str(self.probability_of_na)+""" AND n."""+self.ensemble_or_model_name+"""_political_position_std <= """+str(self.political_position_std)+"""
            RETURN DISTINCT m.proposition, m.locution, n.proposition, n.locution, o.proposition, o.locution LIMIT """+str(number_of_examples)+""""""

            relevant_graph_content = self.query_graph(cypher)
            

            if not relevant_graph_content:  
                empty_string = ""
                return empty_string 

            string = f"""### Pertinent Examples
            
A user has made an argument which is comprised of a claim (or premise) that supports another claim (or conclusion). Your task is to generate a set of questions for the user's premise. We have provided {len(relevant_graph_content)} examples of questions pertaining to the topic that you are discussing. Sentences have been annotated into locution and their propositions. Extract the political position from the examples and use them as a guide to help you generate your response. Examples:

"""

            for index, result in enumerate(relevant_graph_content):
                count = index + 1
                string += f"""Example {count}:"""
                for loc_prop in self.return_type[self.returns]:
                    if loc_prop == "locution":
                        string+=f"""User's premise as a locution: '{result["m.locution"]}'
User's conclusion as a locution: '{result["o.locution"]}'
User's argument as a locution: '{result["m.locution"]}'-[:SUPPORTS]->'{result["o.locution"]}'
Your question about the premise as a locution: '{result["n.locution"]}'
"""
                    if loc_prop == "proposition":
                        string+=f"""User's premise as a proposition: '{result["m.proposition"]}'
User's conclusion as a proposition: '{result["o.proposition"]}'
User's argument in propositional form: '{result["m.proposition"]}'-[:SUPPORTS]->'{result["o.proposition"]}'
Your question about the premise as a proposition: '{result["n.proposition"]}'

"""

            string += """The above examples are to be used as a rough guide. It is imperative that you extract the political position from the examples, even if you don't use the examples, and use that political leaning in your response. The examples provided might refer to past events in the present tense because the data is now a few years old. If so, you should change the tense and the perspective of the examples so that it makes sense in the current dialogue, ensuring that you extract the political leaning from the examples when including them in your response to the user.
            
If some of the examples do not include the word 'why', then ignore it and only extract the political stance from the example. 
                
"""

            return string
        

    #def retract(self):

    def query_graph(self, cypher):
        result = self.knowledge_base.query(cypher)
        return result

    def return_rag_examples(self):
        current_examples = self.state["graph_rag_examples"].copy()

        current_examples.append(self.rag_examples)

        return {"graph_rag_examples" : current_examples}