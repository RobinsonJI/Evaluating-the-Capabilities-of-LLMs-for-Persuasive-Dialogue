from langgraph.graph import StateGraph, START, END
from langchain_community.graphs import Neo4jGraph

from persuasio.rag.vector_based_rag import VectorBasedRAG
from persuasio.rag.graph_rag import GraphRAG
from persuasio.agents.subgraph_agents.utterance_classification import UtteranceClassificationAgent
from persuasio.agents.subgraph_agents.typical_responses import replies_for_last_utterance
from persuasio.agents.subgraph_agents.generation_agents import DialogueMoveGenerator
from persuasio.agents.subgraph_agents.persuasiveness_choice import llm_completion_choice
from persuasio.agents.subgraph_agents.disambiguation import DisambiguationAgent
from persuasio.agents.subgraph_agents.commitments import commitment_update
from persuasio.states.state import GenerationAgentsState
from persuasio.routers.mas_rag_graph_routers import (
    check_for_vector_based_persona_examples,
    check_for_model_start_dialogue_after_vector_based_personas,
    llm_response_router
)
from persuasio.utils.graph_compiler import compile_sub_graph
from persuasio.utils.draw_langgraphs import draw_graph

from persuasio.datatypes.enums import ClassifyingUtteranceOf
from persuasio.datatypes.pydantic_basemodels import (
    ClaimNegationResponse,
    WhyClaimResponses,
    QuestionClaimResponses,
    ConcedeClaimResponses,
    SinceClaimResponses,
    ClaimResponses,
    RetractClaimResponses
)


def create_mas_rag_workflow():
    import persuasio.app as app_module
    # Ensure that agents have access to the neo4j graph if they require it.
    graph = Neo4jGraph(url=app_module.NEO4J_URL, username=app_module.NEO4J_USERNAME, password=app_module.NEO4J_PASSWORD)

    workflow = StateGraph(GenerationAgentsState)

    '''
    ===================================================================================================================================================================
                                                                            NODES
    ===================================================================================================================================================================
    '''
    workflow.add_node("VectorBasedRAGPersonaGenerationAgent", lambda state : VectorBasedRAG(state=state, knowledge_base=graph).persona_string())

    workflow.add_node("InitialClaimGenerationAgent", lambda state : DialogueMoveGenerator(state=state).generate(prompt_method_name="initial_llm_claim",
                                                                                                            schema=ClaimResponses,
                                                                                                            output_key="___Claim___")
                                    )
    
    workflow.add_node("UtteranceClassificationAgent", lambda state: UtteranceClassificationAgent(
        state=state, 
        which_utterances=ClassifyingUtteranceOf.LAST_SPEAKER
        ).return_classified_sentences()
    )
    
    workflow.add_node("PertinentExamplesFromGraphRAG", lambda state : GraphRAG(state=state,knowledge_base=graph).return_rag_examples())

    workflow.add_node("IdentifyModelsSetOfTypicalResponses", replies_for_last_utterance)

    # The nodes below are agents which generate the typical responses identified above, for each sentence that the human user has uttered. 
    # Each node generates 3 sentences, which the persuasive choice agent will then go on to choose the most persuasive sentences to return to the user
    workflow.add_node("SinceClaimGenerationAgent", lambda state : DialogueMoveGenerator(state=state).generate(prompt_method_name="since_claim",
                                                                                                         schema=SinceClaimResponses,
                                                                                                         output_key="___Since___")
                      )
    workflow.add_node("ClaimGenerationAgent", lambda state : DialogueMoveGenerator(state=state).generate(prompt_method_name="claim",
                                                                                                         schema=ClaimResponses,
                                                                                                         output_key="___Claim___")
                      )
    workflow.add_node("ClaimNegationGenerationAgent", lambda state : DialogueMoveGenerator(state=state).generate(prompt_method_name="claim_negation",
                                                                                                         schema=ClaimNegationResponse,
                                                                                                         output_key="___Claim___")
                      )
    workflow.add_node("WhyClaimGenerationAgent", lambda state : DialogueMoveGenerator(state=state).generate(prompt_method_name="why_claim",
                                                                                                         schema=WhyClaimResponses,
                                                                                                         output_key="___Why___")
                      )
    workflow.add_node("QuestionClaimGenerationAgent", lambda state : DialogueMoveGenerator(state=state).generate(prompt_method_name="question_claim",
                                                                                                         schema=QuestionClaimResponses,
                                                                                                         output_key="___Question___")
                      )
    workflow.add_node("ConcedeClaimGenerationAgent", lambda state : DialogueMoveGenerator(state=state).generate(prompt_method_name="concede_claim",
                                                                                                         schema=ConcedeClaimResponses,
                                                                                                         output_key="___Concede___")
                      )
    workflow.add_node("RetractClaimGenerationAgent", lambda state : DialogueMoveGenerator(state=state).generate(prompt_method_name="retract_claim",
                                                                                                         schema=RetractClaimResponses,
                                                                                                         output_key="___Retract___")
                      )
    
    workflow.add_node("PersuasivenessChoiceAgent", llm_completion_choice)

    workflow.add_node("DisambiguationAgent", lambda state : DisambiguationAgent(state = state
                                                                  ).return_disambiguated_sentences()
                                                                  )
    
    workflow.add_node("CommitmentUpdateAgent", commitment_update)

    '''
    ===================================================================================================================================================================
                                                                            EDGES
    ===================================================================================================================================================================
    '''
    workflow.add_conditional_edges(START, check_for_vector_based_persona_examples)
    workflow.add_conditional_edges("VectorBasedRAGPersonaGenerationAgent", check_for_model_start_dialogue_after_vector_based_personas)
    workflow.add_edge("InitialClaimGenerationAgent", "PersuasivenessChoiceAgent")

    workflow.add_edge("UtteranceClassificationAgent", "PertinentExamplesFromGraphRAG")
    workflow.add_edge("PertinentExamplesFromGraphRAG", "IdentifyModelsSetOfTypicalResponses")
    workflow.add_conditional_edges("IdentifyModelsSetOfTypicalResponses", llm_response_router)
    generation_agents = [
        "SinceClaimGenerationAgent",
        "ClaimNegationGenerationAgent",
        "ClaimGenerationAgent",
        "WhyClaimGenerationAgent",
        "QuestionClaimGenerationAgent",
        "ConcedeClaimGenerationAgent",
        "RetractClaimGenerationAgent"]
    
    for node in generation_agents:
        workflow.add_edge(node, "PersuasivenessChoiceAgent")

    workflow.add_edge("PersuasivenessChoiceAgent", "DisambiguationAgent")

    workflow.add_edge("DisambiguationAgent", "CommitmentUpdateAgent")

    workflow.add_edge("CommitmentUpdateAgent", END)

    return workflow

mas_rag_workflow = create_mas_rag_workflow()
mas_rag_graph = compile_sub_graph(workflow=mas_rag_workflow)
draw_graph(graph=mas_rag_graph, graph_name="mas_rag_graph")