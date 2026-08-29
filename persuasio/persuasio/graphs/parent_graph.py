from langgraph.graph import StateGraph, START, END

from persuasio.states.state import ParentState

from persuasio.agents.subgraph_calls import invoke_subgraphs

from persuasio.routers.parent_graph_routers import (
    check_for_end_of_dialogue_after_first_speaker,
    check_for_end_of_dialogue_after_second_speaker
)

from persuasio.tools.end_of_dialogue_outputs import save_dialogue_outputs
from persuasio.utils.graph_compiler import compile_parent_graph
from persuasio.utils.draw_langgraphs import draw_graph


def create_graph():

    workflow = StateGraph(ParentState)

    '''
    ===================================================================================================================================================================
                                                                            NODES
    ===================================================================================================================================================================
    '''
    workflow.add_node("FirstSpeaker", invoke_subgraphs)
    workflow.add_node("SecondSpeaker", invoke_subgraphs)

    workflow.add_node("EndOfDialogueOutputs", save_dialogue_outputs)

    '''
    ===================================================================================================================================================================
                                                                            EDGES
    ===================================================================================================================================================================
    '''
    workflow.add_edge(START, "FirstSpeaker")
    workflow.add_conditional_edges("FirstSpeaker", check_for_end_of_dialogue_after_first_speaker)
    workflow.add_conditional_edges("SecondSpeaker", check_for_end_of_dialogue_after_second_speaker)
    workflow.add_edge("EndOfDialogueOutputs", END)

    return workflow

workflow = create_graph()
#graph = compile_parent_graph(workflow=workflow)
