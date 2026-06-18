# EVOLVE-BLOCK-START
"""5-hotspot beam-search v2 for 3DI3 IL-7Ralpha binder-design benchmark.

Builds on the current best (0.9020): same 5-hotspot set B77+B80+B82+B84+B138
and beam-search algorithm, but with targeted changes to escape the local optimum:

1. Seed: 42 -> 123 — explore a different design-space trajectory while keeping
   all other parameters identical to the winning configuration.
2. sc_scale_score: 2.0 -> 2.5 — stronger self-conditioning score guidance
   pushes the model harder toward the 5-hotspot interface during denoising,
   potentially improving interface contacts and lowering i_pae.
3. local_latents_noise: tan -> 1/t — sharper noise schedule for side-chain
   latents concentrates refinement in the final denoising steps, which can
   improve sequence-structure compatibility at the interface.
4. bb_ca_schedule p: 1.5 -> 1.8 — slightly more log-concentrated backbone
   denoising budget near t=0 for finer backbone geometry at the interface.
5. binder_length: [70,110] -> [72,112] — small shift to sample slightly longer
   binders that can wrap further around the 5-residue epitope surface.

The beam-search (n_branch=2, beam_width=2) produces 12 valid designs per run,
maximising the diversity bonus while directing search toward high-reward regions.
"""

from __future__ import annotations

from typing import Any


def propose_design_run() -> dict[str, Any]:
    """Beam-search v2: 5 hotspots B77+B80+B82+B84+B138, seed=123, sc_score=2.5.

    Hotspot rationale (contact-derived, all within 3.6 A of IL-7 in 3DI3):
      B77 (LYS, 2.73 A) — tightest heavy-atom contact; charged N-terminal anchor
      B80 (LEU, 3.50 A) — hydrophobic core; fills gap in B77-B84 cluster
      B82 (ILE, 3.55 A) — hydrophobic core of the dense N-terminal cluster
      B84 (LEU, 3.50 A) — hydrophobic core anchor validated by prior art
      B138 (LYS, 3.37 A) — mid-domain charged anchor; cross-region coverage

    Changes vs current best (seed=42, sc_score=2.0, tan local noise, p=1.5):
      - seed: 42 -> 123 (fresh trajectory in same configuration)
      - sc_scale_score: 2.0 -> 2.5 (stronger interface self-conditioning)
      - local_latents_noise: tan -> 1/t (sharper side-chain refinement)
      - bb_ca_schedule p: 1.5 -> 1.8 (more concentrated backbone near t=0)
      - binder_length: [70,110] -> [72,112] (slightly longer binders)
    """
    return {
        "hotspot_residues": ["B77", "B80", "B82", "B84", "B138"],
        "binder_length": [72, 112],
        "checkpoint_selection": "complexa_default",
        "denoising_steps": 120,
        "self_conditioning": True,
        "seed": 123,
        "num_length_samples": 2,
        "batch_size": 2,
        "sampling": {
            "bb_ca_schedule": {"mode": "log", "p": 1.8},
            "local_latents_schedule": {"mode": "power", "p": 2.0},
            "bb_ca_noise": {"mode": "1/t", "p": 1.0},
            "local_latents_noise": {"mode": "1/t", "p": 1.0},
            "bb_ca_sc_scale_noise": 0.08,
            "local_latents_sc_scale_noise": 0.08,
            "bb_ca_sc_scale_score": 2.5,
            "local_latents_sc_scale_score": 2.5,
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
