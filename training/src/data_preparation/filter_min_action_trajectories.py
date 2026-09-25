#!/usr/bin/env python3
"""Filter a generated world-model trajectory JSON by minimum action count.

Motivation: the full uncurated TOUCAN (toucan_1_5m, 1.52M trajectories / 26GB) is dominated
by short records -- 10.5%% have NO tool call at all and 26.7%% have exactly one. Single-action
trajectories contribute one history-free transition each; multi-action ones are the records
that teach history-conditioned dynamics. Keeping only ``action_count >= min_actions`` trims
the file to the trajectories a world model actually learns multi-step structure from.

Works on the streaming-written format of generate_adp_world_model_trajectories.py (one
trajectory per line inside a JSON array), so a 26GB file filters in one pass without parsing:
the per-line ``"action_count": N`` field is matched textually and kept lines are copied
verbatim.

Usage:
  uv run python src/data_preparation/filter_min_action_trajectories.py \
      trajectories/toucan_1_5m_world_model_trajectories.json \
      trajectories/toucan_1_5m_multiturn_world_model_trajectories.json \
      --min-actions 2
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ACTION_COUNT = re.compile(r'"action_count": (\d+)')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--min-actions", type=int, default=2)
    args = parser.parse_args()

    kept = dropped = 0
    with args.input.open("r", encoding="utf-8") as reader, args.output.open("w", encoding="utf-8") as writer:
        writer.write("[")
        first = True
        for line in reader:
            match = ACTION_COUNT.search(line)
            if not match:
                continue  # the "[" / "]" array delimiter lines
            if int(match.group(1)) < args.min_actions:
                dropped += 1
                continue
            body = line.rstrip("\n").rstrip(",")
            writer.write(("\n" if first else ",\n") + body)
            first = False
            kept += 1
        writer.write("\n]\n" if not first else "]\n")
    total = kept + dropped
    print(f"kept {kept:,}/{total:,} trajectories (action_count >= {args.min_actions}); "
          f"dropped {dropped:,} -> {args.output}")


if __name__ == "__main__":
    main()
