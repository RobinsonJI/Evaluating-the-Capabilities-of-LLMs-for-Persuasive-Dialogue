#!/bin/bash

# Set the number of sockets (open files) to a large number
ulimit -n 8192

# install dependencies for persuasio
cd Evaluating-the-Capabilities-of-LLMs-for-Persuasive-Dialogue/Code/persuasio && uv sync

# export api endpoints and keys
source tokens.sh # this file has been deleted for security reasons