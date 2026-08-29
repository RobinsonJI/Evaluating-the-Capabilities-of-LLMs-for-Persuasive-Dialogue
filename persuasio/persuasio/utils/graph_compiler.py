def compile_parent_graph(workflow, checkpointer):
    # Compile the graph and ensure that state memory has the checkpoint defined above.
    graph = workflow.compile(checkpointer=checkpointer)

    return graph

def compile_sub_graph(workflow):
    
    graph = workflow.compile()

    return graph


# Helper function to get the compiled graph (now just returns the global instance)
def get_compiled_graph(compiled_graph):
    """
    Get the pre-compiled parent graph instance
    """
    if compiled_graph is None:
        raise RuntimeError("Graph not compiled. Application may not have started properly.")
    return compiled_graph
