#!/bin/bash

SESSION_NAME="model2model_tests"

# Check if session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then 
    echo "Session $SESSION_NAME already exists. Attaching to it."
    tmux attach-session -t $SESSION_NAME

else
    # Create a new session and name it
    tmux new-session -d -s $SESSION_NAME

    # Split the windows
    tmux split-window -h -t "$SESSION_NAME":0.0   
    tmux split-window -v -t "$SESSION_NAME":0.0   

    # Setup run_experiments window
    tmux send-keys -t "$SESSION_NAME":0.0  'source Evaluating-the-Capabilities-of-LLMs-for-Persuasive-Dialogue/Code/scripts/azure-neo4j-setup.sh' C-m

    # Setup persuasio window
    tmux send-keys -t "$SESSION_NAME":0.1 'source Evaluating-the-Capabilities-of-LLMs-for-Persuasive-Dialogue/Code/scripts/azure-persuasio-setup.sh' C-m

    # Setup neo4j server
    tmux send-keys -t "$SESSION_NAME":0.2 'source Evaluating-the-Capabilities-of-LLMs-for-Persuasive-Dialogue/Code/scripts/azure-run-experiments-setup.sh' C-m

    # Attach to the created session
    tmux attach-session -t $SESSION_NAME

fi