import os
import json
from datetime import datetime as dt
import psycopg2 as pg
import argparse
import numpy as np
import random
import itertools
import math
from collections import Counter, deque
import string
from typing import List, Tuple, Dict


# def construct_hh_matchings(n: int) -> (List[tuple], int):
#     """ Construct n-1 perfect matchings of K_n.

#     Args:
#         n (int): Number of humans to be matched.

#     Returns:
#         List[tuple], int: A list of pairs representing the matchings, and c (num. matchings per human).
#     """
#     c = math.ceil(np.log1p(n**2))
#     matchings = []
#     vertices = list(range(1, n))  # 1 to n-1
#     fixed = n  # vertex n is fixed

#     for _ in range(n - 1):
#         matching = [(fixed, vertices[0])]  # pair fixed with first
#         # pair remaining symmetrically
#         for i in range(1, n // 2):
#             matching.append(tuple(sorted([vertices[i], vertices[n - 1 - i]])))
#         matchings.append(matching)
#         vertices = vertices[1:] + vertices[:1]  # rotate
        
#     pairs = [p for m in matchings[:c] for p in m]

#     return pairs, c

def construct_hh_matchings(n: int, c: int | None = None) -> Tuple[List[Tuple[int, int]], int]:
    """Construct c perfect matchings of K_n using 1-factorization.

    Args:
        n: Number of humans to be matched.
        c: Number of debates per human. If None, uses ceil(ln(1 + n²)).

    Returns:
        List of (i, j) pairs representing the matchings, and c (num matchings per human).

    Raises:
        ValueError: If c > n-1 (not enough matchings exist in K_n).
    """
    # Handle odd n by adding dummy vertex
    is_odd = n % 2 != 0
    n_eff = n + 1 if is_odd else n
    dummy = n_eff if is_odd else None

    max_matchings = n_eff - 1

    # Determine c
    if c is None:
        c = math.ceil(np.log1p(n**2))

    if c > max_matchings:
        raise ValueError(f"c={c} exceeds maximum possible matchings ({max_matchings}) for n={n}")

    if c < 1:
        raise ValueError(f"c must be at least 1, got {c}")

    matchings = []
    vertices = list(range(1, n_eff))
    fixed = n_eff

    for _ in range(n_eff - 1):
        matching = [(fixed, vertices[0])]
        for i in range(1, n_eff // 2):
            matching.append(tuple(sorted([vertices[i], vertices[n_eff - 1 - i]])))
        matchings.append(matching)
        vertices = vertices[1:] + vertices[:1]

    # Filter out dummy pairs
    pairs = []
    for m in matchings[:c]:
        for p in m:
            if dummy is None or dummy not in p:
                pairs.append(p)

    return pairs, c


def generate_human_model_debates(humans: list, models: list, repeats: int) -> list:
    """ Generate human-model debate assignments.
    
    Args:
        humans (list): List of human identifiers.
        models (list): List of model identifiers.
        repeats (int): Number of debates per model.
    Returns:
        list: List of (human, model) debate assignments. 
    
    """
    total_debates = len(models) * repeats
    n = len(humans)

    # Create pool with exact counts
    base_count = total_debates // n
    remainder = total_debates % n

    human_pool = []
    shuffled = humans.copy()
    random.shuffle(shuffled)
    for i, h in enumerate(shuffled):
        count = base_count + (1 if i < remainder else 0)
        human_pool.extend([h] * count)

    random.shuffle(human_pool)

    # Assign R debates per model
    debates = []
    idx = 0
    for model in models:
        for _ in range(repeats):
            debates.append((human_pool[idx], model))
            idx += 1

    return debates


def calculate_batch_structure(n_humans: int, n_hh: int, n_hm: int) -> List[Tuple[int, int]]:
    """ Distribute H-H and H-M proportionally across minimum batches.
    
    Args:
        n_humans (int): Number of humans total.
        n_hh (int): Total number of H-H debates.
        n_hm (int): Total number of H-M debates.
        
    Returns:
        List[Tuple[int, int]]: List of (n_hh, n_hm) per batch.
    """
    total_slots = n_hh * 2 + n_hm
    n_batches = -(-total_slots // n_humans)  # ceil

    batches = []
    hh_left, hm_left = n_hh, n_hm

    for i in range(n_batches):
        batches_remaining = n_batches - i

        # Distribute remaining evenly (ceiling to not leave stragglers)
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


def create_batches(
    hh_pairs: List[Tuple[int, int]], 
    hm_pairs: List[Tuple[int, int]], 
    humans: List[int], 
    batch_structure: List[Tuple[int, int]], 
    verbose=False) -> List[List[Tuple[int, int]]]:
    """ Create batches of debates based on the provided structure.
    
    Args:
        hh_pairs (List[Tuple[int, int]]): List of H-H debate pairs.
        hm_pairs (List[Tuple[int, int]]): List of H-M debate pairs.
        humans (List[int]): List of human identifiers.
        batch_structure (List[Tuple[int, int]]): List of (n_hh, n_hm) per batch.
        verbose (bool): Whether to print batch details.
    Returns:
        List[List[Tuple[int, int]]]: List of batches, each containing debate pairs.
    """
    humans_set = set(humans)
    hh_remaining = hh_pairs.copy()
    hm_remaining = hm_pairs.copy()
    batches = []

    for nhh, nhm in batch_structure:
        batch = []
        humans_in_batch = set()

        # Fill H-H first (uses 2 humans each)
        for pair in hh_remaining[:]:
            if len([p for p in batch if p in hh_pairs]) >= nhh:
                break
            if set(pair).isdisjoint(humans_in_batch):
                batch.append(pair)
                humans_in_batch.update(pair)
                hh_remaining.remove(pair)

        # Fill H-M with remaining humans
        for pair in hm_remaining[:]:
            if len([p for p in batch if p in hm_pairs]) >= nhm:
                break
            human = pair[0]  # (human, model)
            if human not in humans_in_batch:
                batch.append(pair)
                humans_in_batch.add(human)
                hm_remaining.remove(pair)

        batches.append(batch)
        
    if verbose:
        for i, b in enumerate(batches):
            print(f"""===== Batch {i+1} =====
        | H-M x {len([p for p in b if p in hm_pairs])}
        | H-H x {len([p for p in b if p in hh_pairs])} 
        | Missing Humans: {list(set(humans) - {h for p in b for h in p if h in humans})}
        | {', '.join(f"{p[0]}/{p[1]}" for p in b)}""")

    return batches


def experiment_setup(
    n_h: int,
    n_m: int,
    repeats: int,
    c: int | None = None,
    verbose: bool=True
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]], List[List[Tuple[int, int]]], int]:
    """ Set up the experiment batches.

    Args:
        n_h (int): Number of humans.
        n_m (int): Number of models.
        repeats (int): Number of debates per model.
        c (int | None): Debates per human. If None, uses ceil(ln(1 + n²)).
        verbose (bool): Whether to print batch details.
    Returns:
        Tuple containing: H-H pairs, H-M pairs, batch structure, batches, and c.
    """
    H = list(range(1, n_h + 1))
    M = [string.ascii_uppercase[i] for i in range(n_m)]
    hh_pairs, c = construct_hh_matchings(n_h, c=c)
    batch_structure = calculate_batch_structure(n_h, len(hh_pairs), n_m * repeats)
    batches, hm_pairs = create_batches_v2(hh_pairs, H, M, repeats, batch_structure)

    if verbose:
        total_pairs = len(hh_pairs) + len(hm_pairs)
        total_speakers = n_h + n_m
        expected_pairs = total_speakers * np.log(total_speakers)
        print(f"\n=== N={n_h}, M={n_m}, C={c}, H-H={len(hh_pairs)}, H-M={len(hm_pairs)}, total={total_pairs} {'=' if total_pairs == expected_pairs else '>' if total_pairs > expected_pairs else '<'} nln(n)={expected_pairs:.0f} ===")
        hh_set = set(hh_pairs)
        for i, b in enumerate(batches):
            hh_in_batch = [p for p in b if p in hh_set]
            hm_in_batch = [p for p in b if p not in hh_set]
            humans_in_batch = {h for p in b for h in (p if isinstance(p[1], int) else [p[0]])}
            missing = sorted(set(H) - humans_in_batch)
            print(f"| B{i+1}: H-H x {len(hh_in_batch)}, H-M x {len(hm_in_batch)}, Missing: {missing}, Pairs: {', '.join(f'{p[0]}/{p[1]}' for p in b)}")
        print(f"| Total Batches: {len(batches)}")
        
        # print lists of matchups for each human
        h_matchups = {h: [] for h in H}
        for p in hh_pairs:
            h_matchups[p[0]].append(p[1])
            h_matchups[p[1]].append(p[0])
        for p in hm_pairs:
            h_matchups[p[0]].append(p[1])
            
        print("\nHuman Matchups:")
        for h in H:
            print(f"| h{h}: {', '.join(str(x) for x in h_matchups[h])}")
        
        # print counts of matchups for each model
        m_counter = Counter()
        for p in hm_pairs:
            m_counter[p[1]] += 1
        print("\nModel Debate Counts:")
        for m in M:
            print(f"| {m}: {m_counter[m]}")

    return hh_pairs, hm_pairs, batch_structure, batches, c


def create_batches_v2(hh_pairs, humans, models, repeats, batch_structure):
    """Assign H-H pairs to batches, then generate H-M to fill remaining slots."""
    hh_remaining = list(hh_pairs)
    batches = []

    hm_needed = {m: repeats for m in models}
    hm_per_human = {h: 0 for h in humans}
    hm_assigned = set()  # Track (human, model) pairs already assigned

    for n_hh, n_hm in batch_structure:
        batch = []
        humans_in_batch = set()

        # Fill H-H first
        for pair in hh_remaining[:]:
            if len(batch) >= n_hh:
                break
            if set(pair).isdisjoint(humans_in_batch):
                batch.append(pair)
                humans_in_batch.update(pair)
                hh_remaining.remove(pair)

        # Fill H-M with remaining humans
        available_humans = [h for h in humans if h not in humans_in_batch]
        available_humans.sort(key=lambda h: hm_per_human[h])

        hm_added = 0
        for h in available_humans:
            if hm_added >= n_hm:
                break
            # Pick model that needs debates and hasn't debated this human
            for m in sorted(hm_needed.keys(), key=lambda m: -hm_needed[m]):
                if hm_needed[m] > 0 and (h, m) not in hm_assigned:
                    batch.append((h, m))
                    hm_needed[m] -= 1
                    hm_per_human[h] += 1
                    hm_assigned.add((h, m))
                    humans_in_batch.add(h)
                    hm_added += 1
                    break

        batches.append(batch)

    hm_pairs = [p for b in batches for p in b if p not in hh_pairs]

    return batches, hm_pairs


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Experiment Setup")
    # n_humans -nh (accept single number or comma separated list)
    parser.add_argument(
        '--n_humans', 
        '-nh', 
        type=str, 
        required=True,
        help='Number of humans (single number or comma separated list)'
    )
    parser.add_argument(
        '--n_models',
        '-nm',
        type=int,
        default=9,
        help='Number of models'
    )
    parser.add_argument(
        '--c',
        '-c',
        type=str,
        default=None,
        help='Debates per human (single number or comma separated list matching n_humans). If not specified, uses ceil(ln(1 + n²))'
    )
    args = parser.parse_args()
    n_h = [int(x) for x in args.n_humans.split(',')]

    # Parse c values
    if args.c is None:
        c_values = [None] * len(n_h)
    else:
        c_values = [int(x) for x in args.c.split(',')]
        if len(c_values) == 1:
            # Single c value applies to all n
            c_values = c_values * len(n_h)
        elif len(c_values) != len(n_h):
            raise ValueError(f"Length of c ({len(c_values)}) must match length of n_humans ({len(n_h)})")

    for n, c in zip(n_h, c_values):
        experiment_setup(n, args.n_models, repeats=4, c=c, verbose=True)