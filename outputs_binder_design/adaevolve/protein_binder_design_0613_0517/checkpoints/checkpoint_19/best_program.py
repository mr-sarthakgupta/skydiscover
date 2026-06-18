# EVOLVE-BLOCK-START
"""4-hotspot beam-search candidate for the 3DI3 IL-7Ralpha binder-design benchmark.

Strategy: Use beam-search (n_branch=2, beam_width=2) with 4 strategic hotspots
B77+B82+B84+B138. Beam-search explores multiple trajectories simultaneously and
selects the best at each denoising step, yielding more directed exploration than
best-of-n independent replicas. The 4-hotspot set provides dense N-terminal
cluster coverage (B77/B82/B84) plus mid-domain anchor B138 (LYS, 3.37A), which
spans a broader epitope surface than the prior B77+B84+B143 set. Retain the
proven sampling parameters (log bb_ca schedule, power local_latents, 1/t2 noise,
sc_scale_score=1.8) that drove the current best score.
"""

from __future__ import annotations

from typing import Any


def propose_design_run() -> dict[str, Any]:
    """Beam-search with 4 hotspots B77+B82+B84+B138 for directed IL-7Ra binder design.

    Hotspot rationale:
      B77 (LYS, 2.73 A) — tightest heavy-atom contact in the complex
      B82 (ILE, 3.55 A) — hydrophobic core of the dense N-terminal cluster
      B84 (LEU, 3.50 A) — hydrophobic core anchor from prior art
      B138 (LYS, 3.37 A) — mid-domain anchor, different region from B143

    Algorithm: beam-search (n_branch=2, beam_width=2) replaces best-of-n to
    explore multiple denoising trajectories simultaneously, selecting the best
    partial designs at each step for more directed reward optimization.

    Sampling: retain proven log/power schedules and 1/t2 local noise with
    sc_scale_score=1.8 from the current best (score=0.8447).
    """
    return {
        "hotspot_residues": ["B77", "B82", "B84", "B138"],
        "binder_length": [68, 105],
        "checkpoint_selection": "complexa_default",
        "denoising_steps": 120,
        "self_conditioning": True,
        "seed": 7,
        "num_length_samples": 2,
        "batch_size": 2,
        "sampling": {
            "bb_ca_schedule": {"mode": "log", "p": 1.5},
            "local_latents_schedule": {"mode": "power", "p": 2.0},
            "bb_ca_noise": {"mode": "1/t", "p": 1.0},
            "local_latents_noise": {"mode": "1/t2", "p": 1.0},
            "bb_ca_sc_scale_noise": 0.08,
            "local_latents_sc_scale_noise": 0.08,
            "bb_ca_sc_scale_score": 1.8,
            "local_latents_sc_scale_score": 1.8,
        },
        "search": {
            "algorithm": "beam-search",
            "best_of_n": {"replicas": 2},
            "beam_search": {"n_branch": 2, "beam_width": 2},
            "fk_steering": {"n_branch": 2, "beam_width": 2, "temperature": 0.1},
            "mcts": {
                "n_simulations": 4,
                "exploration_prob": 0.4,
                "exploration_constant": 1.0,
            },
        },
    }


def run_discovery() -> dict[str, Any]:
    """Compatibility entry point used by the benchmark evaluator."""
    return propose_design_run()


# EVOLVE-BLOCK-END


if __name__ == "__main__":
    import json

    print(json.dumps(propose_design_run(), indent=2))
