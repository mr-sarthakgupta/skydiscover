# EVOLVE-BLOCK-START
"""Improved candidate for the 3DI3 IL-7Ralpha binder-design benchmark.

Strategy: Replace B62 with B77 (the tightest heavy-atom contact at 2.73 A,
LYS) while keeping proven anchors B84 (hydrophobic core, 3.50 A) and B143
(C-terminal domain anchor). This 3-hotspot set spans the receptor with the
strongest contact signal. Increase denoising steps to 120 (stage-2 cap) for
higher sample quality. Strengthen sc_scale_score to 1.8 for both bb_ca and
local_latents to amplify self-conditioning guidance at the interface. Use
log schedule (p=1.5) for bb_ca to spread backbone denoising more evenly, and
power schedule (p=2.5) for local latents to concentrate side-chain refinement
near t=0. Use 1/t2 noise for local latents (sharper near t=0) to improve
side-chain placement at the interface. Tighten binder length to [65, 100] to
encourage compact, well-folded binders that can engage the tight epitope.
"""

from __future__ import annotations

from typing import Any


def propose_design_run() -> dict[str, Any]:
    """Focused 3-hotspot design: B77 (tightest contact) + B84 + B143 anchors.

    Hotspot rationale:
      B77 (LYS, 2.73 A) — strongest heavy-atom contact, replaces B62
      B84 (LEU, 3.50 A) — hydrophobic core anchor from prior art
      B143 (residue, prior art) — C-terminal domain coverage

    Sampling changes vs baseline:
      - denoising_steps 80 -> 120 (stage-2 cap, more refinement)
      - bb_ca_schedule p 2.0 -> 1.5 (more uniform backbone denoising)
      - local_latents_schedule power p 2.5 (sharper side-chain refinement)
      - local_latents_noise 1/t2 (sharper noise near t=0 for side chains)
      - sc_scale_score 1.0 -> 1.8 (stronger self-conditioning guidance)
      - binder_length [70,110] -> [65,100] (compact binders for tight epitope)
      - seed changed to 13 for new design space exploration
    """
    return {
        "hotspot_residues": ["B77", "B84", "B143"],
        "binder_length": [65, 100],
        "checkpoint_selection": "complexa_default",
        "denoising_steps": 120,
        "self_conditioning": True,
        "seed": 13,
        "num_length_samples": 2,
        "batch_size": 2,
        "sampling": {
            "bb_ca_schedule": {"mode": "log", "p": 1.5},
            "local_latents_schedule": {"mode": "power", "p": 2.5},
            "bb_ca_noise": {"mode": "1/t", "p": 1.0},
            "local_latents_noise": {"mode": "1/t2", "p": 1.0},
            "bb_ca_sc_scale_noise": 0.08,
            "local_latents_sc_scale_noise": 0.08,
            "bb_ca_sc_scale_score": 1.8,
            "local_latents_sc_scale_score": 1.8,
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
