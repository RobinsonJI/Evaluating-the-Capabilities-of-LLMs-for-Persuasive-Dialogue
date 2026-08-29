#!/bin/bash

sudo apt-get update

# Install repo containing git LFS
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

sudo apt-get update

# Pull the dump files using git lfs
cd Evaluating-the-Capabilities-of-LLMs-for-Persuasive-Dialogue/ && git pull && git lfs pull && cd ..

sudo apt-get update

# Install OpenJDK 21
sudo apt install openjdk-21-jdk -y

sudo apt-get update

# Add the neo4j debian repo to apt keys
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/neotechnology.gpg

# Add the neo4j repo to apt list
echo 'deb [signed-by=/etc/apt/keyrings/neotechnology.gpg] https://debian.neo4j.com stable latest' | sudo tee -a /etc/apt/sources.list.d/neo4j.list

sudo apt-get update

# Install neo4j
sudo apt-get install neo4j=1:2025.08.0 -y

# Copy the neo4j dump file (format = aligned) to neo4j dumps directory
sudo cp Evaluating-the-Capabilities-of-LLMs-for-Persuasive-Dialogue/Code/persuasio/persuasio/rag/knowledge_base/neo4j.dump /var/lib/neo4j/data/dumps/neo4j.dump

# Change owner of the file to neo4j
sudo chown neo4j:neo4j /var/lib/neo4j/data/dumps/neo4j.dump

# Change permissions of dump file to read only
sudo chmod 666 /var/lib/neo4j/data/dumps/neo4j.dump

# neo4j user runs command to load dump file into db
sudo runuser -l neo4j -c 'neo4j-admin database load --from-path=/var/lib/neo4j/data/dumps neo4j --overwrite-destination=true'

# Download plugins
sudo wget https://github.com/neo4j/apoc/releases/download/2025.08.0/apoc-2025.08.0-core.jar -P /var/lib/neo4j/plugins # apoc
sudo wget https://github.com/neo4j/graph-data-science/releases/download/2.21.0/neo4j-graph-data-science-2.21.0.jar -P /var/lib/neo4j/plugins # graph data science

# Change parameters in the config file
sudo sed -i 's/^#*\s*dbms.security.procedures.unrestricted=.*/dbms.security.procedures.unrestricted=apoc.*,gds.*/' /etc/neo4j/neo4j.conf
sudo sed -i 's/^#*\s*dbms.security.procedures.allowlist=.*/dbms.security.procedures.allowlist=apoc.*,gds.*/' /etc/neo4j/neo4j.conf
sudo sed -i 's/^#*\s*server.memory.heap.initial_size=.*/server.memory.heap.initial_size=1g/' /etc/neo4j/neo4j.conf
sudo sed -i 's/^#*\s*server.memory.heap.max_size=.*/server.memory.heap.max_size=4g/' /etc/neo4j/neo4j.conf
sudo sed -i 's/^#*\s*server.memory.pagecache.size=.*/server.memory.pagecache.size=8g/' /etc/neo4j/neo4j.conf

# Make the neo4j server public facing (omitted for model2model tests)
#sudo sed -i 's/^#*\s*server.default_listen_address=.*/server.default_listen_address=0.0.0.0/' /etc/neo4j/neo4j.conf

# Set the password of the db before running
sudo runuser -l neo4j -c "neo4j-admin dbms set-initial-password 'neo4j-db<ATI&Liv&Lee>&<Persuasio>!4<dso>'"

# Load plugins and conf file parameters
sudo systemctl restart neo4j

# Start db
sudo systemctl start neo4j

# Check if db is running
sudo systemctl status neo4j