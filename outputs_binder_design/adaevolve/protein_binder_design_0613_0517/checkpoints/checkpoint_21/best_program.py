# EVOLVE-BLOCK-START
"""5-hotspot beam-search candidate for the 3DI3 IL-7Ralpha binder-design benchmark.

Strategy: Expand hotspot coverage to 5 residues spanning the full dense contact
cluster B77+B80+B82+B84+B138. The B77-B84 cluster (B77 LYS 2.73A, B80 LEU 3.50A,
B82 ILE 3.55A, B84 LEU 3.50A) forms a contiguous hydrophobic/charged patch on
IL-7Ralpha; adding B80 fills the gap left by the 4-hotspot set. B138 (LYS 3.37A)
provides the mid-domain anchor. This 5-hotspot selection covers the entire primary
binding interface, encouraging the model to design binders that engage the full
surface patch rather than isolated anchor points.

Key changes vs current best (0.8822):
  - Hotspots: B77+B82+B84+B138 -> B77+B80+B82+B84+B138 (add B80 LEU 3.50A)
  - local_latents_noise: 1/t2 -> tan (smoother for multi-residue interfaces)
  - sc_scale_score: 1.8 -> 2.0 (stronger self-conditioning for 5-hotspot guidance)
  - binder_length: [68,105] -> [70,110] (slightly wider for larger interface)
  - seed: 7 -> 42 (fresh design space exploration)
"""

from __future__ import annotations

from typing import Any


def propose_design_run() -> dict[str, Any]:
    """Beam-search with 5 hotspots B77+B80+B82+B84+B138 for full-patch IL-7Ra design.

    Hotspot rationale (contact-derived, all within 3.6 A):
      B77 (LYS, 2.73 A) — tightest heavy-atom contact; charged anchor
      B80 (LEU, 3.50 A) — hydrophobic core; fills gap in B77-B84 cluster
      B82 (ILE, 3.55 A) — hydrophobic core of the dense N-terminal cluster
      B84 (LEU, 3.50 A) — hydrophobic core anchor from prior art
      B138 (LYS, 3.37 A) — mid-domain charged anchor; cross-region coverage

    Algorithm: beam-search (n_branch=2, beam_width=2) explores multiple
    denoising trajectories simultaneously for directed reward optimization,
    retaining the 12-valid-design throughput of the current best.

    Sampling: log bb_ca schedule, power local_latents schedule, 1/t backbone
    noise, tan local_latents noise (smoother for multi-residue interface),
    sc_scale_score=2.0 for stronger 5-hotspot self-conditioning guidance.
    """
    return {
        "hotspot_residues": ["B77", "B80", "B82", "B84", "B138"],
        "binder_length": [70, 110],
        "checkpoint_selection": "complexa_default",
        "denoising_steps": 120,
        "self_conditioning": True,
        "seed": 42,
        "num_length_samples": 2,
        "batch_size": 2,
        "sampling": {
            "bb_ca_schedule": {"mode": "log", "p": 1.5},
            "local_latents_schedule": {"mode": "power", "p": 2.0},
            "bb_ca_noise": {"mode": "1/t", "p": 1.0},
            "local_latents_noise": {"mode": "tan", "p": 1.0},
            "bb_ca_sc_scale_noise": 0.08,
            "local_latents_sc_scale_noise": 0.08,
            "bb_ca_sc_scale_score": 2.0,
            "local_latents_sc_scale_score": 2.0,
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
