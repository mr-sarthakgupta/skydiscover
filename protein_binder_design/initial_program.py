# EVOLVE-BLOCK-START
"""Seed candidate for the 3DI3 IL-7Ralpha binder-design benchmark.

SkyDiscover evolves the dictionary returned by ``propose_design_run``.  The
benchmark evaluator validates this dictionary and turns it into a bounded
Proteina-Complexa generation run against 3DI3 chain B.
"""

from __future__ import annotations

from typing import Any


def propose_design_run() -> dict[str, Any]:
    """Return a constrained Proteina-Complexa binder-design run specification."""
    return {
        "hotspot_residues": ["B62", "B84", "B143"],
        "binder_length": [70, 110],
        "checkpoint_selection": "complexa_default",
        "denoising_steps": 80,
        "self_conditioning": True,
        "seed": 7,
        "num_length_samples": 2,
        "batch_size": 2,
        "sampling": {
            "bb_ca_schedule": {"mode": "log", "p": 2.0},
            "local_latents_schedule": {"mode": "power", "p": 2.0},
            "bb_ca_noise": {"mode": "1/t", "p": 1.0},
            "local_latents_noise": {"mode": "tan", "p": 1.0},
            "bb_ca_sc_scale_noise": 0.08,
            "local_latents_sc_scale_noise": 0.08,
            "bb_ca_sc_scale_score": 1.0,
            "local_latents_sc_scale_score": 1.0,
        },
        "search": {
            "algorithm": "best-of-n",
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
