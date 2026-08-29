import os
from pathlib import Path

def draw_graph(graph, graph_name : str) -> None:

    root_path = Path(__file__).parent.parent

    folder_path = root_path / "graphs" / "figures"

    file_path = folder_path / f"{graph_name}.png"

    # Create folder if missing
    if not folder_path.exists():
        folder_path.resolve().mkdir(parents=True, exist_ok=False)

    # Draw the compiled graph
    try:
        png_graph = graph.get_graph().draw_mermaid_png()
    

        # Save
        with open(file_path, "wb") as f:
            f.write(png_graph)
    except:
        png_graph = graph.get_graph().draw_mermaid()

        # Save
        with open(file_path, "w") as f:
            f.write(png_graph)
