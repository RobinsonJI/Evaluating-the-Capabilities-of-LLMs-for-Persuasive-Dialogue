#!/bin/bash

# Donwload uv
wget -qO- https://astral.sh/uv/install.sh | sh

# add uv's executebale to path
source $HOME/.local/bin/env

# Update apt repo 
sudo apt-get update

# Download tmux
sudo apt install tmux -y

sudo apt-get update

# Install C and C compiler 
sudo apt update
sudo apt install build-essential -y

# Install the PostgreSQL SDK
sudo apt-get install libpq-dev -y

# install Python-dev
sudo apt-get install python3-dev -y

sudo apt-get update