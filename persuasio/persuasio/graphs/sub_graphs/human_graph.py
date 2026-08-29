from langgraph.graph import StateGraph, START, END

from persuasio.agents.subgraph_agents.utterance_classification import UtteranceClassificationAgent
from persuasio.agents.subgraph_agents.commitments import commitment_update
from persuasio.datatypes.enums import ClassifyingUtteranceOf
from persuasio.states.state import HumanState
from persuasio.agents.subgraph_agents.human_in_the_loop import interrupt_and_resume
from persuasio.agents.subgraph_agents.typical_responses import replies_for_last_utterance
from persuasio.routers.human_graph_routers import check_if_user_starts_dialogue
from persuasio.utils.graph_compiler import compile_sub_graph
from persuasio.utils.draw_langgraphs import draw_graph

def create_human_workflow():

    workflow = StateGraph(HumanState)

    '''
    ===================================================================================================================================================================
                                                                            NODES
    ===================================================================================================================================================================
    '''
    
    workflow.add_node("UtteranceClassificationAgentBeforeInterrupt", lambda state: UtteranceClassificationAgent(
        state=state, 
        which_utterances=ClassifyingUtteranceOf.LAST_SPEAKER
        ).return_classified_sentences()
    )
    workflow.add_node("IdentifyHumansSetOfTypicalResponses", replies_for_last_utterance)
    workflow.add_node("InterruptAndGetHumanResponse", interrupt_and_resume)
    workflow.add_node("UtteranceClassificationAgentAfterInterrupt", lambda state: UtteranceClassificationAgent(
        state=state, 
        which_utterances=ClassifyingUtteranceOf.HUMAN_RESPONSE
        ).return_classified_sentences()
    )
    workflow.add_node("CommitmentUpdateAgent", commitment_update)
    

    '''
    ===================================================================================================================================================================
                                                                            EDGES
    ===================================================================================================================================================================
    '''
    workflow.add_conditional_edges(START, check_if_user_starts_dialogue)
    workflow.add_edge("UtteranceClassificationAgentBeforeInterrupt", "IdentifyHumansSetOfTypicalResponses")
    workflow.add_edge("IdentifyHumansSetOfTypicalResponses", "InterruptAndGetHumanResponse")
    workflow.add_edge("InterruptAndGetHumanResponse", "UtteranceClassificationAgentAfterInterrupt")
    workflow.add_edge("UtteranceClassificationAgentAfterInterrupt", "CommitmentUpdateAgent")
    workflow.add_edge("CommitmentUpdateAgent", END)

    return workflow

human_workflow = create_human_workflow()
human_graph = compile_sub_graph(workflow=human_workflow)
draw_graph(graph=human_graph, graph_name="human_graph")