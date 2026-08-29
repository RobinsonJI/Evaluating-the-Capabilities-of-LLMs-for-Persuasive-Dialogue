from langgraph.graph import StateGraph, START, END

from persuasio.agents.subgraph_agents.utterance_classification import UtteranceClassificationAgent
from persuasio.agents.subgraph_agents.base_model import BaseModelAgent
from persuasio.agents.subgraph_agents.commitments import commitment_update
from persuasio.datatypes.enums import ClassifyingUtteranceOf
from persuasio.states.state import BaseModelState
from persuasio.utils.graph_compiler import compile_sub_graph
from persuasio.utils.draw_langgraphs import draw_graph


def create_base_workflow():

    workflow = StateGraph(BaseModelState)

    '''
    ===================================================================================================================================================================
                                                                            NODES
    ===================================================================================================================================================================
    '''
    workflow.add_node("BaseModelAgent", lambda state : BaseModelAgent(state).return_base_model_response())

    workflow.add_node("UtteranceClassificationAgent", lambda state: UtteranceClassificationAgent(
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
    workflow.add_edge(START, "BaseModelAgent")
    workflow.add_edge("BaseModelAgent", "UtteranceClassificationAgent")
    workflow.add_edge("UtteranceClassificationAgent", "CommitmentUpdateAgent")
    workflow.add_edge("CommitmentUpdateAgent", END)

    return workflow

base_workflow = create_base_workflow()
base_graph = compile_sub_graph(workflow=base_workflow)
draw_graph(graph = base_graph, graph_name="base_graph")