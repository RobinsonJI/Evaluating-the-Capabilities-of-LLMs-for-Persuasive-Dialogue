import math
import random
from typing import List, Tuple, Set, Dict

from session_manager.models.entities import Participant, Session, SessionParameters
from session_manager.models.enums import SessionStatus, SpeakerType, ModelName
from session_manager.setup.models import ExperimentConfig, Batch
from session_manager.setup.utility import get_model_name_from_participant, generate_session_id, get_human_participants_from_session


def construct_hh_matchings(n: int, c: int = None) -> Tuple[List[Tuple[int, int]], int]:
    """Construct balanced H-H pairs using 1-factorization of K_n.

    Uses round-robin rotation to generate edge-disjoint perfect matchings
    of the complete graph K_n. Selects the first C matchings where
    C = ceil(ln(1 + n^2)), guaranteeing exactly C debates per human.

    Args:
        n: Number of humans (indices will be 0 to n-1)
        c: Debates per human. If None, uses ceil(ln(1 + n^2)).

    Returns:
        Tuple of:
            - List of (human_idx, human_idx) pairs (0-indexed)
            - c: number of debates per human
    """
    # Handle odd n by adding dummy vertex
    is_odd = n % 2 != 0
    n_eff = n + 1 if is_odd else n
    dummy = n_eff - 1 if is_odd else None

    max_matchings = n_eff - 1

    # Determine c
    if c is None:
        c = math.ceil(math.log1p(n ** 2))

    if c > max_matchings:
        raise ValueError(f"c={c} exceeds maximum possible matchings ({max_matchings}) for n={n}")

    if c < 1:
        raise ValueError(f"c must be at least 1, got {c}")

    matchings = []
    vertices = list(range(0, n_eff - 1))  # 0 to n_eff-2
    fixed = n_eff - 1  # last vertex is fixed

    for _ in range(n_eff - 1):
        matching = [(fixed, vertices[0])]
        for i in range(1, n_eff // 2):
            matching.append(tuple(sorted([vertices[i], vertices[n_eff - 1 - i]])))
        matchings.append(matching)
        vertices = vertices[1:] + vertices[:1]  # rotate

    # Filter out dummy pairs and take first c matchings
    pairs = []
    for m in matchings[:c]:
        for p in m:
            if dummy is None or dummy not in p:
                pairs.append(p)

    return pairs, c


# =============================================================================
# Session Creation
# =============================================================================

def create_session_with_positions(
    p1: Participant,
    p2: Participant,
    p1_position: str,
    p2_position: str,
    topic: str,
    who_starts_first: str,
    human_model_name: ModelName,
    existing_ids: Set[str],
) -> Session:
    """Create a single session with specified political positions and starter.

    Args:
        p1: First participant
        p2: Second participant
        p1_position: Political position range for p1 (e.g., "20:40" for left)
        p2_position: Political position range for p2 (e.g., "60:80" for right)
        topic: Debate topic
        who_starts_first: participant_id of who speaks first (must be p1 or p2's ID)
        human_model_name: Model used for classifying human utterances
        existing_ids: Set to track existing session IDs (modified in place)

    Returns:
        A Session object with configured SessionParameters
    """
    session_id = generate_session_id(existing_ids)

    # Determine speaker order based on who_starts_first
    if who_starts_first == p1.participant_id:
        first_speaker_id = p1.participant_id
        second_speaker_id = p2.participant_id
        first_speaker_type = p1.participant_type
        second_speaker_type = p2.participant_type
        first_speaker_model = get_model_name_from_participant(p1)
        second_speaker_model = get_model_name_from_participant(p2)
        first_position = p1_position
        second_position = p2_position
    else:
        first_speaker_id = p2.participant_id
        second_speaker_id = p1.participant_id
        first_speaker_type = p2.participant_type
        second_speaker_type = p1.participant_type
        first_speaker_model = get_model_name_from_participant(p2)
        second_speaker_model = get_model_name_from_participant(p1)
        first_position = p2_position
        second_position = p1_position

    params = SessionParameters(
        session_id=session_id,
        debate_topic=topic,
        first_speaker=first_speaker_id,
        second_speaker=second_speaker_id,
        first_speaker_type=first_speaker_type,
        second_speaker_type=second_speaker_type,
        first_speaker_model_name=first_speaker_model,
        second_speaker_model_name=second_speaker_model,
        # Note: Using exact field names from entities.py (including the typo)
        first_speaker_political_political_position_range=first_position,
        second_speaker_political_position_range=second_position,
        human_model_name=human_model_name,
    )

    return Session(
        session_id=session_id,
        parameters=params,
        status=SessionStatus.STARTED,
    )


def make_hh_sessions(
    humans: List[Participant],
    config: ExperimentConfig,
    existing_ids: Set[str],
    c: int = None,
) -> Tuple[List[Session], int]:
    """Generate H-H debate sessions using 1-factorization.

    Creates one session per pair with balanced conditions:
    - Who starts first alternates
    - Political positions alternate

    Args:
        humans: List of human Participant objects
        config: Experiment configuration
        existing_ids: Set of existing session IDs (modified in place)
        c: Debates per human. If None, uses ceil(ln(1 + n^2)).

    Returns:
        Tuple of (list of Session objects, c value used)
    """
    pairs, c = construct_hh_matchings(len(humans), c)

    sessions = []
    for idx, (i, j) in enumerate(pairs):
        p1, p2 = humans[i], humans[j]  # 0-indexed now

        # Alternate conditions across pairs for balance
        # condition 0: p1 starts, p1 left
        # condition 1: p1 starts, p1 right
        # condition 2: p2 starts, p2 left
        # condition 3: p2 starts, p2 right
        condition = idx % 4
        who_starts = p1 if condition in [0, 1] else p2
        p1_is_left = condition in [0, 3]

        p1_position = config.left_position if p1_is_left else config.right_position
        p2_position = config.right_position if p1_is_left else config.left_position

        session = create_session_with_positions(
            p1, p2,
            p1_position=p1_position,
            p2_position=p2_position,
            topic=config.debate_topic,
            who_starts_first=who_starts.participant_id,
            human_model_name=config.humans.human_model_name,
            existing_ids=existing_ids,
        )
        sessions.append(session)

    return sessions, c


# =============================================================================
# Batch Assignment with On-the-fly H-M Session Generation
# =============================================================================

def calculate_batch_structure(
    n_humans: int, n_hh: int, n_hm: int
) -> List[Tuple[int, int]]:
    """Distribute H-H and H-M debates proportionally across minimum batches.

    Args:
        n_humans: Number of humans total
        n_hh: Total number of H-H debates
        n_hm: Total number of H-M debates

    Returns:
        List of (hh_count, hm_count) tuples, one per batch
    """
    total_slots = n_hh * 2 + n_hm
    n_batches = -(-total_slots // n_humans)  # ceil division

    batches = []
    hh_left, hm_left = n_hh, n_hm

    for i in range(n_batches):
        batches_remaining = n_batches - i

        # Distribute remaining evenly
        hh_target = -(-hh_left // batches_remaining)
        hm_target = -(-hm_left // batches_remaining)

        # Cap by batch capacity
        hh_take = min(hh_target, hh_left, n_humans // 2)
        slots_used = hh_take * 2
        hm_take = min(hm_target, hm_left, n_humans - slots_used)

        batches.append((hh_take, hm_take))
        hh_left -= hh_take
        hm_left -= hm_take

    return batches


def create_batches_with_sessions(
    hh_sessions: List[Session],
    humans: List[Participant],
    models: List[Participant],
    config: ExperimentConfig,
    existing_ids: Set[str],
) -> Tuple[List[Session], List[Batch]]:
    """Assign H-H sessions to batches and generate H-M sessions on-the-fly.

    This ensures optimal batching by creating H-M sessions to fill remaining
    slots in each batch, guaranteeing each model gets exactly `repeats` debates.

    Args:
        hh_sessions: List of human-human Session objects
        humans: List of human Participant objects
        models: List of AI Participant objects
        config: Experiment configuration
        existing_ids: Set of existing session IDs (modified in place)

    Returns:
        Tuple of:
            - List of H-M Session objects created
            - List of Batch objects with session_ids
    """
    n_humans = len(humans)
    n_models = len(models)
    n_hh = len(hh_sessions)
    n_hm = n_models * config.repeats

    # Build human_id -> index mapping
    human_id_to_idx = {h.participant_id: i for i, h in enumerate(humans)}
    human_ids = set(human_id_to_idx.keys())

    batch_structure = calculate_batch_structure(n_humans, n_hh, n_hm)

    hh_remaining = list(hh_sessions)
    batches: List[Batch] = []
    hm_sessions: List[Session] = []

    # Track H-M assignment state
    hm_needed: Dict[int, int] = {m: config.repeats for m in range(n_models)}
    hm_per_human: Dict[int, int] = {h: 0 for h in range(n_humans)}
    hm_assigned: Set[Tuple[int, int]] = set()

    hm_condition_idx = 0  # For alternating conditions

    for batch_id, (n_hh_target, n_hm_target) in enumerate(batch_structure, start=1):
        batch_session_ids: List[str] = []
        humans_in_batch: Set[str] = set()

        # Fill H-H first (each uses 2 humans)
        hh_added = 0
        for session in hh_remaining[:]:
            if hh_added >= n_hh_target:
                break
            session_humans = get_human_participants_from_session(session, human_ids)
            if set(session_humans).isdisjoint(humans_in_batch):
                batch_session_ids.append(session.session_id)
                humans_in_batch.update(session_humans)
                hh_remaining.remove(session)
                hh_added += 1

        # Generate H-M sessions on-the-fly to fill remaining slots
        # Get available humans (not in batch yet), sorted by fewest H-M debates
        available_human_ids = [h.participant_id for h in humans if h.participant_id not in humans_in_batch]
        available_human_ids.sort(key=lambda hid: hm_per_human[human_id_to_idx[hid]])

        hm_added = 0
        for human_id in available_human_ids:
            if hm_added >= n_hm_target:
                break

            human_idx = human_id_to_idx[human_id]
            human = humans[human_idx]

            # Pick model that needs debates and hasn't debated this human
            for model_idx in sorted(range(n_models), key=lambda m: -hm_needed[m]):
                if hm_needed[model_idx] > 0 and (human_idx, model_idx) not in hm_assigned:
                    ai = models[model_idx]

                    # Alternate conditions for balance
                    condition = hm_condition_idx % 4
                    who_starts = human if condition in [0, 1] else ai
                    human_is_left = condition in [0, 3]

                    human_position = config.left_position if human_is_left else config.right_position
                    ai_position = config.right_position if human_is_left else config.left_position

                    session = create_session_with_positions(
                        human, ai,
                        p1_position=human_position,
                        p2_position=ai_position,
                        topic=config.debate_topic,
                        who_starts_first=who_starts.participant_id,
                        human_model_name=config.humans.human_model_name,
                        existing_ids=existing_ids,
                    )

                    hm_sessions.append(session)
                    batch_session_ids.append(session.session_id)
                    humans_in_batch.add(human_id)

                    hm_needed[model_idx] -= 1
                    hm_per_human[human_idx] += 1
                    hm_assigned.add((human_idx, model_idx))
                    hm_added += 1
                    hm_condition_idx += 1
                    break

        batches.append(Batch(batch_id=batch_id, session_ids=batch_session_ids))

    return hm_sessions, batches


# =============================================================================
# Main Entry Point
# =============================================================================

def make_all_sessions(
    humans: List[Participant],
    models: List[Participant],
    config: ExperimentConfig,
    existing_ids: Set[str] = None,
) -> Tuple[List[Session], List[Batch], int]:
    """Generate all sessions (H-H and H-M) and assign to batches.

    H-H sessions are generated first using 1-factorization.
    H-M sessions are generated on-the-fly during batch assignment to ensure
    optimal batching and that each model gets exactly `repeats` debates.

    Args:
        humans: List of human Participant objects
        models: List of AI Participant objects
        config: Experiment configuration
        existing_ids: Set of existing session IDs (optional, created if None)

    Returns:
        Tuple of:
            - All Session objects (H-H + H-M)
            - List of Batch objects with session_ids
            - c value (debates per human for H-H)
    """
    if existing_ids is None:
        existing_ids = set()

    # Generate H-H sessions
    hh_sessions, c = make_hh_sessions(humans, config, existing_ids, config.humans.c)

    # Create batches and generate H-M sessions on-the-fly
    hm_sessions, batches = create_batches_with_sessions(
        hh_sessions, humans, models, config, existing_ids
    )

    all_sessions = hh_sessions + hm_sessions
    return all_sessions, batches, c
