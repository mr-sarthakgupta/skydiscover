# Protein Binder Design Benchmark

SkyDiscover benchmark for evolving Proteina-Complexa binder-design run
specifications against RCSB 3DI3 chain B, human IL-7Ralpha ectodomain.

The evolved program does not run models directly. It returns a constrained
dictionary of hotspots, binder length, sampling settings, and test-time search
settings. `evaluator.py` validates that dictionary, launches Proteina-Complexa
generation, reads Proteina's `rewards_*.csv`, and reports `combined_score`.

## Files

- `initial_program.py`: seed EVOLVE-BLOCK returning a Proteina run spec.
- `evaluator.py`: SkyDiscover evaluator with `evaluate`, `evaluate_stage1`,
  and `evaluate_stage2`.
- `config.yaml`: LLM prompt and cascade evaluator settings.
- `prepare_3di3_assets.py`: reproducible RCSB download and target extraction.
- `assets/3di3.pdb`: raw RCSB PDB coordinates.
- `assets/3di3.cif`: raw RCSB mmCIF coordinates.
- `assets/3di3_chain_b_il7ra_target.pdb`: chain-B-only Proteina target PDB.
- `assets/target_metadata.json`: allowed hotspots and interface candidates.

## Target

The benchmark targets IL-7Ralpha, chain B of 3DI3. The seed uses the
RFdiffusion/OpenProtein prior hotspots `B62`, `B84`, and `B143`, but the LLM is
expected to choose hotspots as part of the optimization problem. The evaluator
accepts any modeled chain-B residue listed in `assets/target_metadata.json`.

The cleaned target PDB strips chain A and glycan chains C-E so Proteina sees a
standard protein target. The raw coordinate files remain in `assets/` for
reproducibility.

## Environment

The evaluator expects Proteina-Complexa at the sibling directory
`../Proteina-Complexa` relative to the `skydiscover/` checkout (that is,
`protein-ag/Proteina-Complexa` when both repos live in the same workspace).

Override with:

```bash
export PROTEINA_COMPLEXA_ROOT=/path/to/Proteina-Complexa
```

Required for real model evaluations:

```bash
export CKPT_PATH=/path/to/complexa/checkpoints
export AF2_DIR=/path/to/alphafold/params
```

`CKPT_PATH` should contain `complexa.ckpt` and `complexa_ae.ckpt`. The default
seed uses `checkpoint_selection: "complexa_default"`, which maps to
`complexa.ckpt`.

For schema-only smoke tests that do not run Proteina:

```bash
export SKYDISCOVER_BINDER_VALIDATE_ONLY=1
```

### 24 GB GPU memory limits

Proteina loads Complexa plus AF2 multimer on the same GPU. On cards with ~24 GB
VRAM (e.g. RTX 3090), the evaluator caps stage-2 search to avoid OOM:

```bash
export SKYDISCOVER_BINDER_STAGE2_BATCH_SIZE=2
export SKYDISCOVER_BINDER_STAGE2_NUM_LENGTH_SAMPLES=2
export SKYDISCOVER_BINDER_STAGE2_REPLICAS=2
export SKYDISCOVER_BINDER_MAX_BATCH_SIZE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`run_protein_binder_design_experiment.sh` and `protein-ag/.env` set these by
default. Tighten further (e.g. `MAX_BATCH_SIZE=2`) if OOM persists.

Noise schedule modes must match Proteina's implementation: use `1/t` or `tan`
for `bb_ca_noise` / `local_latents_noise`. Modes like `power` or `log` for
noise are rejected by the evaluator and would crash Proteina if passed through.

## Run

From the SkyDiscover repo:

```bash
uv run skydiscover-run \
  benchmarks/protein_binder_design/initial_program.py \
  benchmarks/protein_binder_design/evaluator.py \
  -c benchmarks/protein_binder_design/config.yaml \
  -s adaevolve \
  -i 20
```

Local evaluator smoke test:

```bash
SKYDISCOVER_BINDER_VALIDATE_ONLY=1 \
python benchmarks/protein_binder_design/evaluator.py \
  benchmarks/protein_binder_design/initial_program.py
```

Regenerate target assets:

```bash
python benchmarks/protein_binder_design/prepare_3di3_assets.py
```

## Scoring

`evaluator.py` runs Proteina's `search_binder_local_pipeline` with Hydra
overrides for the benchmark target and evolved hyperparameters. It reads the
latest `rewards_*.csv` and scores finite `total_reward` values.

Returned metrics include:

- `combined_score`: bounded SkyDiscover objective, higher is better.
- `best_total_reward`: best Proteina generation reward.
- `mean_total_reward`: mean finite Proteina generation reward.
- `num_valid_designs`: number of generated designs with finite reward.
- `selected_hotspots`: hotspot residues chosen by the candidate.
- `search_algorithm`: Proteina test-time search strategy.

Cascade evaluation uses a smaller Stage 1 run to reject invalid or weak
candidates before Stage 2 performs the fuller bounded run.
