""" 
Pipeline to setup experiment database with participants and sessions.
"""
import argparse
from pathlib import Path
from typing import List, Set
import json
import sys
import yaml
import pandas as pd
import time
import runpy


from session_manager.models.entities import Participant, Session
from session_manager.models.enums import SpeakerType
from session_manager.setup.data import SetupDB
from session_manager.setup.models import ExperimentConfig, ModelConfig, HumansConfig
from session_manager.setup.pairings import make_all_sessions
from session_manager.setup.utility import generate_auth_code


def create_admin_participant(code: str = "<4DM1N>") -> Participant:
    """Create an admin participant with auth code."""
    return Participant(
        participant_id="ADMIN",
        participant_type=SpeakerType.HUMAN,
        auth_code=code,
        is_admin=True,
    )


def generate_human_participants(humans_config: HumansConfig, existing_codes: Set[str]) -> List[Participant]:
    """
    Generate human participants with random auth codes.

    Args:
        humans_config: HumansConfig with n (number of humans) and optional emails
        existing_codes: Set to track existing auth codes (modified in place)

    Returns:
        List of participants.
        Participant IDs: human_1, human_2, etc.
        Auth codes: Random 6-digit alphanumeric codes
    """
    participants = []

    for i in range(1, humans_config.n + 1):
        participant_id = f"human_{i}"
        auth_code = generate_auth_code(existing_codes)

        participant = Participant(
            participant_id=participant_id,
            participant_type=SpeakerType.HUMAN,
            auth_code=auth_code,
        )
        participants.append(participant)

    return participants


def generate_ai_participants(model_configs: List[ModelConfig]) -> List[Participant]:
    """
    Generate AI participants from model configurations.

    Args:
        model_configs: List of ModelConfig, each specifying a model and its variants

    Returns:
        List of AI Participant objects.
        Participant IDs: gpt-4o_base, grok-3_mas_rag, etc.
    """
    participants = []

    for mc in model_configs:
        for variant in mc.variants:
            participant_id = f"{mc.model.value}_{variant.value}"

            participant = Participant(
                participant_id=participant_id,
                participant_type=variant,
                auth_code=None,
            )
            participants.append(participant)

    return participants


def push_to_db(
    sql_session_manager: SetupDB,
    sql_persuasio: SetupDB,
    sql_logs: SetupDB,
    participants: List[Participant],
    sessions: List[Session],
    backup_path: Path):
    """Push participants and sessions to the experiment databases."""
    
    # get current time for backup subdir suffixes
    timestamp = time.strftime("%Y%m%d_%H%M%S") 
    
    backup_dir = backup_path / f"{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # session_manager:
    sm_backup_dir = backup_dir / "session_manager"
    sm_backup_dir.mkdir(parents=True, exist_ok=True)
    # download and empty "sessions", "participants", and "dialogues"
    sql_session_manager.download_and_empty_all_tables(
        tables=["sessions", "participants", "dialogues"],
        backup_dir=sm_backup_dir
    )
    # replace with new "participants" and "sessions"
    sql_session_manager.replace_table(
        table="participants",
        rows=[
            dict(participant_id=p.participant_id, participant_data=p.model_dump(mode="json"))
            for p in participants
        ]
    )
    sql_session_manager.replace_table(
        table="sessions",
        rows=[
            dict(session_id=s.session_id, session_data=s.model_dump(mode="json")) 
            for s in sessions
        ]
    )
    
    # persuasio:
    per_backup_dir = backup_dir / "persuasio"
    per_backup_dir.mkdir(parents=True, exist_ok=True)
    # list all tables -> download and empty each
    tables = sql_persuasio.list_tables()
    sql_persuasio.download_and_empty_all_tables(
        tables=tables,
        backup_dir=per_backup_dir
    )
    
    # logs:
    log_backup_dir = backup_dir / "logs"
    log_backup_dir.mkdir(parents=True, exist_ok=True)   
    # list all tables -> download and empty each
    tables = sql_logs.list_tables()
    sql_logs.download_and_empty_all_tables(
        tables=tables,
        backup_dir=log_backup_dir
    )
    
    
def make_mail_merge(batches: List[Session], sessions: List[Session], humans: List[Participant], output_dir: Path) -> Path:
    """ Generate mail merge Excel file for human participants in each batch.
    Args:
        batches: List of Batch objects
        sessions: List of Session objects
        humans: List of human Participant objects
        output_dir: Directory to save the Excel file
    Returns:
        Path to the generated Excel file
    """
    dfs = []
    for batch in batches:
        rows = []
        for session in sessions:
            if session.session_id in batch.session_ids:
                p1 = session.parameters.first_speaker
                p2 = session.parameters.second_speaker
                
                for p_self, p_opp in [(p1, p2), (p2, p1)]:
                    if p_self.startswith("human_"):
                        row = {
                            "user_id": p_self,
                            "email": h2email[p_self],
                            "auth_code": next(h.auth_code for h in humans if h.participant_id == p_self),
                            "session_code": session.session_id,
                            "opponent_id": p_opp,
                        }
                        rows.append(row)
        # add rows for any missing humans with empty codes / opponent
        human_ids_in_batch = {
            p1 for session in sessions 
            if session.session_id in batch.session_ids 
            for p1 in [
                session.parameters.first_speaker, 
                session.parameters.second_speaker
            ] 
            if p1.startswith("human_")
        }
        for h in humans:
            if h.participant_id not in human_ids_in_batch:
                row = {
                    "user_id": h.participant_id,
                    "email": h2email[h.participant_id],
                    "auth_code": h.auth_code,
                    "session_code": "No Session Assigned",
                    "opponent_id": "No Session Assigned",
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        dfs.append((batch.batch_id, df))
    
    excel_file = output_dir / "batches_mail_merge.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        for batch_id, df in dfs:
            sheet_name = f"Batch_{batch_id}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    return excel_file
    


if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Setup experiment database tables.")
    argparser.add_argument(
        '--config_yml',
        '-c',
        type=str,
        required=True,
        help="Path to experiment config YAML file."
    )
    argparser.add_argument(
        '--push_to_db',
        '-db',
        action='store_true',
        help="If set, will push generated configs to the experiment database."
    )
    argparser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="data/templates/default",
        help="Directory to save generated files."
    )
    argparser.add_argument(
        '--backup_path',
        '-b',
        type=str,
        default="../backups",
        help="Path to save backups of existing database tables before pushing new data."
    )
    args = argparser.parse_args()

    # load config yaml into ExperimentConfig
    config_yml_path = Path(args.config_yml)
    assert config_yml_path.exists(), f"Config YAML file not found: {config_yml_path}"
    config = ExperimentConfig.from_yaml(config_yml_path)
    
    # setup SetupDB clients
    sql_session_manager = SetupDB(sql_params=config.sql.session_manager)
    sql_persuasio = SetupDB(sql_params=config.sql.persuasio)
    sql_logs = SetupDB(sql_params=config.sql.logs)

    # generate human and model participants
    human_codes = set()
    humans = generate_human_participants(config.humans, human_codes)
    models = generate_ai_participants(config.models)
    all_participants = humans + models + [create_admin_participant()]
    
    # generate all sessions and batches
    sessions, batches, c = make_all_sessions(
        humans=humans,
        models=models,
        config=config,
    )
    
    # save sessions and participants and batches to json
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    participants_file = output_dir / "participants.json"
    with open(participants_file, "w") as f:
        json.dump([p.model_dump(mode="json") for p in all_participants], f, indent=2)
    print(f"  Wrote {len(all_participants)} participants to {participants_file}")

    sessions_file = output_dir / "sessions.json"
    with open(sessions_file, "w") as f:
        json.dump([s.model_dump(mode="json") for s in sessions], f, indent=2)
    print(f"  Wrote {len(sessions)} sessions to {sessions_file}")
    
    batches_file = output_dir / "batches.json"
    with open(batches_file, "w") as f:
        json.dump([b.model_dump(mode="json") for b in batches], f, indent=2)
    print(f"  Wrote {len(batches)} batches to {batches_file}")
    
    # print summary
    # for each batch, print:
    # count of H-H and H-M sessions
    # list of session IDs
    # list of pairs of participant ID in each session
    # list of missing participants (humans not in any session)
    human_ids = {p.participant_id for p in humans}
    print("\nBatches Summary:")
    for batch in batches: 
        assigned_humans = set()
        hh_count = 0
        hm_count = 0
        session_participants = []
        for session in sessions:
            if session.session_id in batch.session_ids:
                p1 = session.parameters.first_speaker
                p2 = session.parameters.second_speaker
                    
                session_participants.append((p1, p2))
                
                if p1 in human_ids and p2 in human_ids:
                    hh_count += 1
                elif p1 in human_ids or p2 in human_ids:
                    hm_count += 1
                # track assigned humans
                if p1 in human_ids:
                    assigned_humans.add(p1)
                if p2 in human_ids:
                    assigned_humans.add(p2)
        missing_h = human_ids - assigned_humans
        
        print(f"=== Batch {batch.batch_id} ===")
        print(f"| H-H x{hh_count}, H-M x{hm_count}")
        print(f"| Session IDs: {batch.session_ids}")
        print(f"| Participant pairs: {session_participants}")
        print(f"| Missing humans: {missing_h}")
        
    # for each human, print list of matchups
    print("\nHuman Matchups:")
    h_matchups = {h.participant_id: [] for h in humans}
    for session in sessions:
        p1 = session.parameters.first_speaker
        p2 = session.parameters.second_speaker
        if p1 in h_matchups:
            h_matchups[p1].append(p2)
        if p2 in h_matchups:
            h_matchups[p2].append(p1)
    for h_id, matchups in h_matchups.items():
        print(f"| {h_id} ({len(matchups)}): {matchups}")
        
    # for each model, print count of assigned sessions
    print("\nModel Session Counts:")
    m_session_counts = {m.participant_id: 0 for m in models}
    for session in sessions:
        p1 = session.parameters.first_speaker
        p2 = session.parameters.second_speaker
        if p1 in m_session_counts:
            m_session_counts[p1] += 1
        if p2 in m_session_counts:
            m_session_counts[p2] += 1
    for m_id, count in m_session_counts.items():
        print(f"| {m_id}: {count}")
        
    # for each participant, print count of first/second speaker assignments
    print("\nParticipant Speaker Role Counts:")
    p_role_counts = {p.participant_id: {"first": 0, "second": 0} for p in all_participants}
    for session in sessions:
        p1 = session.parameters.first_speaker
        p2 = session.parameters.second_speaker
        if p1 in p_role_counts:
            p_role_counts[p1]["first"] += 1
        if p2 in p_role_counts:
            p_role_counts[p2]["second"] += 1
    for p_id, counts in p_role_counts.items():
        print(f"| {p_id}: {counts['first']} first, {counts['second']} second")
        
    # for each participant, print count of left/right political position assignments
    print("\nParticipant Political Position Counts:")
    p_position_counts = {p.participant_id: {"left": 0, "right": 0} for p in all_participants}
    for session in sessions:
        p1 = session.parameters.first_speaker
        p2 = session.parameters.second_speaker
        if p1 in p_position_counts:
            if session.parameters.first_speaker_political_political_position_range == "20:40":
                p_position_counts[p1]["left"] += 1
            else:
                p_position_counts[p1]["right"] += 1
        if p2 in p_position_counts:
            if session.parameters.second_speaker_political_position_range == "20:40":
                p_position_counts[p2]["left"] += 1
            else:
                p_position_counts[p2]["right"] += 1
    for p_id, counts in p_position_counts.items():
        print(f"| {p_id}: {counts['left']} L, {counts['right']} R")
        
    ### PUSH TO DB ###
    if args.push_to_db:
        backup_path = Path(args.backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        push_to_db(
            sql_session_manager=sql_session_manager,
            sql_persuasio=sql_persuasio,
            sql_logs=sql_logs,
            participants=all_participants,
            sessions=sessions,
            backup_path=backup_path,
        )  
        print(f"\nPushed to databases, backups saved to {backup_path}")
            
    
    ### SAVE BATCHES FOR MAIL MERGE ###
    # generate dfs for each batch with columns: user id, email, auth code, session code, opponent id
    # save dfs as separate sheets in excel file
    # if emails provided, use, otherwise set empty email col
    if config.humans.emails is None:
        h2email = {h.participant_id: "" for h in humans}
    else:
        assert len(config.humans.emails) >= len(humans), f"Not enough emails {len(config.humans.emails)} provided for humans {len(humans)}."
        h2email = {h.participant_id: email for h, email in zip(humans, config.humans.emails)}
    
    excel_file = make_mail_merge(
        batches=batches,
        sessions=sessions,
        humans=humans,
        output_dir=output_dir,
    )
    
    print(f"\nWrote mail merge Excel file to {excel_file}")


def main():
    """Entry point for sm_setup (without --push_to_db)."""
    sys.argv = [sys.argv[0], '-c'] + sys.argv[1:]
    runpy.run_path(__file__, run_name='__main__')


def main_with_db():
    """Entry point for sm_setup_db (with --push_to_db)."""
    sys.argv = [sys.argv[0], '-c'] + sys.argv[1:] + ['--push_to_db']
    runpy.run_path(__file__, run_name='__main__')