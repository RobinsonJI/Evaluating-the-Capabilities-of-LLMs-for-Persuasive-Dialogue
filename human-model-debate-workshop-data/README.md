# Human--Model Debate Workshop Data

Raw data collected from the human--model debate workshop, including questionnaire responses, server logs, and database exports.

## Overview

This directory contains all data artefacts produced during the workshop in which human participants debated against LLMs via the [Persuasio](../persuasio) backend, [Session Manager](../session-manager), and [User Interface](../persuasui) interface.

## Data

### Questionnaires

| File | Description |
|------|-------------|
| `human-participant-workshop-entry-questionnaire.csv` | Pre-workshop questionnaire capturing participant demographics and prior attitudes. |
| `human-participant-workshop-exit-survey.csv` | Post-workshop survey capturing participant experience and feedback. |
| `debate-sessions.xlsx` | The sessions that took place during the workshop in batches. |

### Logs

The `logs/` directory contains server logs from each component during the workshop:

| File | Description |
|------|-------------|
| `logs/client.json` | Client-side interaction logs. |
| `logs/persuasio.json` | Persuasio API server logs. |
| `logs/session_manager.json` | Session Manager server logs. |

### Persuasio Database Export

The `persuasio/` directory contains JSON exports from the Persuasio PostgreSQL database:

| File | Description |
|------|-------------|
| `checkpoint_blobs.json` | Serialised LangGraph checkpoint data. |
| `checkpoint_migrations.json` | Checkpoint schema migration records. |
| `checkpoint_writes.json` | Individual checkpoint write operations. |
| `checkpoints.json` | LangGraph checkpoint metadata. |
| `final_states.json` | Final dialogue graph states at session end. |
| `runtime_states.json` | Intermediate dialogue graph states during execution. |
| `session_data.json` | Persuasio session metadata (models, topics, configuration). |
| `terminated_states.json` | States of sessions that were terminated early (these were treated as draws). |

### Session Manager Database Export

The `session_manager/` directory contains JSON exports from the Session Manager database:

| File | Description |
|------|-------------|
| `dialogues.json` | Dialogue transcripts between human participants and LLMs. |
| `participants.json` | Participant records and assigned session mappings. |
| `sessions.json` | Workshop session metadata and configuration. |

## License

This project is licensed under the [MIT License](../LICENSE).