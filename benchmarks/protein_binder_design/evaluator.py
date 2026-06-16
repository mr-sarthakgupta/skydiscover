"""Evaluator for the 3DI3 IL-7Ralpha Proteina-Complexa binder benchmark."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[2]
ASSETS_DIR = BENCHMARK_DIR / "assets"
TARGET_METADATA_PATH = ASSETS_DIR / "target_metadata.json"
TARGET_NAME = "SKY_3DI3_IL7RA"
PROTEINA_ROOT = Path(
    os.environ.get("PROTEINA_COMPLEXA_ROOT", str(REPO_ROOT / "Proteina-Complexa"))
).expanduser()
PROTEINA_CONFIG_NAME = "search_binder_local_pipeline"
PROTEINA_CONFIG_DIR = PROTEINA_ROOT / "configs"
PROTEINA_PYTHON = Path(
    os.environ.get("PROTEINA_PYTHON", sys.executable)
).expanduser()
DEFAULT_TIMEOUT_SECONDS = 7200

_FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "input"}
_ALLOWED_IMPORT_MODULES = {"__future__", "typing"}
_ALLOWED_SEARCH_ALGORITHMS = {"single-pass", "best-of-n", "beam-search", "fk-steering", "mcts"}
_ALLOWED_CHECKPOINT_ALIASES = {"complexa_default": "complexa.ckpt", "complexa": "complexa.ckpt"}
_ALLOWED_SCHEDULE_MODES = {"log", "power", "uniform"}
# Must match Proteina get_gt() in product_space_flow_matcher.py — not every string
# in the upstream config docs is implemented.
_ALLOWED_NOISE_MODES = {"1/t", "tan", "1-t/t", "1/t2", "1/t1p5", "2/t", "2/t2", "2/t1p5"}


class CandidateValidationError(ValueError):
    """Raised when an evolved candidate violates the benchmark contract."""


def _load_metadata() -> dict[str, Any]:
    with TARGET_METADATA_PATH.open() as f:
        return json.load(f)


TARGET_METADATA = _load_metadata()
ALLOWED_HOTSPOTS = set(TARGET_METADATA["target"]["allowed_hotspots"])
INTERFACE_HOTSPOTS = [item["hotspot"] for item in TARGET_METADATA["interface_hotspot_candidates"]]
PRIOR_HOTSPOTS = TARGET_METADATA["prior_art_hotspots"]["hotspots"]
TARGET_PDB_PATH = ASSETS_DIR / TARGET_METADATA["target"]["target_pdb"]


def _is_module_docstring(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If) or node.orelse:
        return False
    test = node.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or len(test.comparators) != 1:
        return False
    return (
        isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and isinstance(test.ops[0], ast.Eq)
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _validate_candidate_source(program_path: str) -> None:
    source = Path(program_path).read_text()
    tree = ast.parse(source, filename=program_path)
    has_entrypoint = False

    for node in tree.body:
        if _is_module_docstring(node) or _is_main_guard(node):
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module not in _ALLOWED_IMPORT_MODULES:
                raise CandidateValidationError(f"import from {node.module!r} is not allowed")
            continue
        if isinstance(node, ast.Assign):
            try:
                ast.literal_eval(node.value)
            except Exception as exc:
                raise CandidateValidationError("top-level assignments must be literal constants") from exc
            continue
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                raise CandidateValidationError("function decorators are not allowed")
            if node.name in {"propose_design_run", "run_discovery"}:
                has_entrypoint = True
            for child in ast.walk(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    raise CandidateValidationError("imports inside evolved functions are not allowed")
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id in _FORBIDDEN_CALLS:
                    raise CandidateValidationError(f"call to {child.func.id}() is not allowed")
            continue
        raise CandidateValidationError(f"top-level {type(node).__name__} is not allowed")

    if not has_entrypoint:
        raise CandidateValidationError("candidate must define propose_design_run() or run_discovery()")


def _load_candidate_spec(program_path: str) -> dict[str, Any]:
    _validate_candidate_source(program_path)
    spec = importlib.util.spec_from_file_location("binder_candidate", program_path)
    if spec is None or spec.loader is None:
        raise CandidateValidationError(f"could not load candidate: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    fn = getattr(module, "propose_design_run", None) or getattr(module, "run_discovery", None)
    if fn is None:
        raise CandidateValidationError("missing propose_design_run() or run_discovery()")
    result = fn()
    if not isinstance(result, dict):
        raise CandidateValidationError("candidate entry point must return a dict")
    return result


def _as_int(value: Any, name: str, low: int, high: int) -> int:
    if isinstance(value, bool):
        raise CandidateValidationError(f"{name} must be an integer")
    try:
        out = int(value)
    except Exception as exc:
        raise CandidateValidationError(f"{name} must be an integer") from exc
    if out < low or out > high:
        raise CandidateValidationError(f"{name} must be between {low} and {high}")
    return out


def _as_float(value: Any, name: str, low: float, high: float) -> float:
    try:
        out = float(value)
    except Exception as exc:
        raise CandidateValidationError(f"{name} must be a number") from exc
    if not math.isfinite(out) or out < low or out > high:
        raise CandidateValidationError(f"{name} must be finite and between {low} and {high}")
    return out


def _normalise_hotspots(value: Any) -> list[str]:
    if value is None:
        value = PRIOR_HOTSPOTS
    if not isinstance(value, (list, tuple)):
        raise CandidateValidationError("hotspot_residues must be a list")
    hotspots = []
    for item in value:
        if not isinstance(item, str):
            raise CandidateValidationError("hotspot residues must be strings like 'B84'")
        residue = item.strip().upper()
        if not re.fullmatch(r"B\d+[A-Z]?", residue):
            raise CandidateValidationError(f"invalid hotspot residue format: {item!r}")
        if residue not in ALLOWED_HOTSPOTS:
            raise CandidateValidationError(f"hotspot {residue!r} is not a modeled chain-B residue")
        if residue not in hotspots:
            hotspots.append(residue)
    if not 1 <= len(hotspots) <= 8:
        raise CandidateValidationError("choose between 1 and 8 hotspot residues")
    return hotspots


def _normalise_length_range(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise CandidateValidationError("binder_length must be [min_len, max_len]")
    low = _as_int(value[0], "binder_length[0]", 40, 220)
    high = _as_int(value[1], "binder_length[1]", 40, 220)
    if low > high:
        raise CandidateValidationError("binder_length minimum cannot exceed maximum")
    if high - low > 120:
        raise CandidateValidationError("binder_length range is too wide for this benchmark")
    return [low, high]


def _normalise_schedule(value: Any, name: str, allowed_modes: set[str]) -> dict[str, Any]:
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise CandidateValidationError(f"{name} must be a dict")
    mode = str(value.get("mode", "log" if "schedule" in name else "1/t"))
    if mode not in allowed_modes:
        raise CandidateValidationError(f"{name}.mode {mode!r} is unsupported")
    return {"mode": mode, "p": _as_float(value.get("p", 1.0), f"{name}.p", 0.05, 8.0)}


def _normalise_spec(raw: dict[str, Any], *, stage: str) -> dict[str, Any]:
    sampling = raw.get("sampling", {})
    if sampling is None:
        sampling = {}
    if not isinstance(sampling, dict):
        raise CandidateValidationError("sampling must be a dict")

    search = raw.get("search", {})
    if search is None:
        search = {}
    if not isinstance(search, dict):
        raise CandidateValidationError("search must be a dict")

    algorithm = str(search.get("algorithm", "best-of-n"))
    if algorithm not in _ALLOWED_SEARCH_ALGORITHMS:
        raise CandidateValidationError(f"unsupported search algorithm: {algorithm!r}")

    nsteps = _as_int(raw.get("denoising_steps", 80), "denoising_steps", 10, 400)
    if stage == "stage1":
        nsteps = min(nsteps, _as_int(os.environ.get("SKYDISCOVER_BINDER_STAGE1_NSTEPS", 20), "stage1_nsteps", 5, 80))
    else:
        nsteps = min(nsteps, _as_int(os.environ.get("SKYDISCOVER_BINDER_MAX_NSTEPS", 120), "max_nsteps", 10, 400))

    stage2_batch_cap = _as_int(os.environ.get("SKYDISCOVER_BINDER_STAGE2_BATCH_SIZE", 2), "stage2_batch_cap", 1, 8)
    stage2_length_cap = _as_int(
        os.environ.get("SKYDISCOVER_BINDER_STAGE2_NUM_LENGTH_SAMPLES", 2),
        "stage2_length_cap",
        1,
        8,
    )
    stage2_replica_cap = _as_int(os.environ.get("SKYDISCOVER_BINDER_STAGE2_REPLICAS", 2), "stage2_replica_cap", 1, 8)

    batch_size = _as_int(raw.get("batch_size", 2), "batch_size", 1, 8)
    num_length_samples = _as_int(raw.get("num_length_samples", 2), "num_length_samples", 1, 8)
    if stage == "stage1":
        batch_size = min(batch_size, 1)
        num_length_samples = min(num_length_samples, 1)
    else:
        batch_size = min(batch_size, stage2_batch_cap)
        num_length_samples = min(num_length_samples, stage2_length_cap)

    best_of_n = search.get("best_of_n", {}) if isinstance(search.get("best_of_n", {}), dict) else {}
    beam_search = search.get("beam_search", {}) if isinstance(search.get("beam_search", {}), dict) else {}
    fk_steering = search.get("fk_steering", {}) if isinstance(search.get("fk_steering", {}), dict) else {}
    mcts = search.get("mcts", {}) if isinstance(search.get("mcts", {}), dict) else {}

    spec = {
        "hotspot_residues": _normalise_hotspots(raw.get("hotspot_residues")),
        "binder_length": _normalise_length_range(raw.get("binder_length", [70, 110])),
        "checkpoint_selection": str(raw.get("checkpoint_selection", "complexa_default")),
        "denoising_steps": nsteps,
        "self_conditioning": bool(raw.get("self_conditioning", True)),
        "seed": _as_int(raw.get("seed", 7), "seed", 0, 2**31 - 1),
        "num_length_samples": num_length_samples,
        "batch_size": batch_size,
        "sampling": {
            "bb_ca_schedule": _normalise_schedule(
                sampling.get("bb_ca_schedule", {"mode": "log", "p": 2.0}),
                "bb_ca_schedule",
                _ALLOWED_SCHEDULE_MODES,
            ),
            "local_latents_schedule": _normalise_schedule(
                sampling.get("local_latents_schedule", {"mode": "power", "p": 2.0}),
                "local_latents_schedule",
                _ALLOWED_SCHEDULE_MODES,
            ),
            "bb_ca_noise": _normalise_schedule(
                sampling.get("bb_ca_noise", {"mode": "1/t", "p": 1.0}),
                "bb_ca_noise",
                _ALLOWED_NOISE_MODES,
            ),
            "local_latents_noise": _normalise_schedule(
                sampling.get("local_latents_noise", {"mode": "tan", "p": 1.0}),
                "local_latents_noise",
                _ALLOWED_NOISE_MODES,
            ),
            "bb_ca_sc_scale_noise": _as_float(sampling.get("bb_ca_sc_scale_noise", 0.1), "bb_ca_sc_scale_noise", 0.0, 2.0),
            "local_latents_sc_scale_noise": _as_float(
                sampling.get("local_latents_sc_scale_noise", 0.1),
                "local_latents_sc_scale_noise",
                0.0,
                2.0,
            ),
            "bb_ca_sc_scale_score": _as_float(sampling.get("bb_ca_sc_scale_score", 1.0), "bb_ca_sc_scale_score", 0.0, 5.0),
            "local_latents_sc_scale_score": _as_float(
                sampling.get("local_latents_sc_scale_score", 1.0),
                "local_latents_sc_scale_score",
                0.0,
                5.0,
            ),
        },
        "search": {
            "algorithm": algorithm,
            "best_of_n": {"replicas": _as_int(best_of_n.get("replicas", 2), "best_of_n.replicas", 1, 8)},
            "beam_search": {
                "n_branch": _as_int(beam_search.get("n_branch", 2), "beam_search.n_branch", 1, 8),
                "beam_width": _as_int(beam_search.get("beam_width", 2), "beam_search.beam_width", 1, 8),
            },
            "fk_steering": {
                "n_branch": _as_int(fk_steering.get("n_branch", 2), "fk_steering.n_branch", 1, 8),
                "beam_width": _as_int(fk_steering.get("beam_width", 2), "fk_steering.beam_width", 1, 8),
                "temperature": _as_float(fk_steering.get("temperature", 0.1), "fk_steering.temperature", 0.0, 5.0),
            },
            "mcts": {
                "n_simulations": _as_int(mcts.get("n_simulations", 4), "mcts.n_simulations", 1, 64),
                "exploration_prob": _as_float(mcts.get("exploration_prob", 0.4), "mcts.exploration_prob", 0.0, 1.0),
                "exploration_constant": _as_float(
                    mcts.get("exploration_constant", 1.0),
                    "mcts.exploration_constant",
                    0.0,
                    10.0,
                ),
            },
        },
    }
    if stage == "stage1":
        spec["search"]["best_of_n"]["replicas"] = min(spec["search"]["best_of_n"]["replicas"], 1)
        spec["search"]["beam_search"]["n_branch"] = min(spec["search"]["beam_search"]["n_branch"], 1)
        spec["search"]["beam_search"]["beam_width"] = min(spec["search"]["beam_search"]["beam_width"], 1)
        spec["search"]["fk_steering"]["n_branch"] = min(spec["search"]["fk_steering"]["n_branch"], 1)
        spec["search"]["fk_steering"]["beam_width"] = min(spec["search"]["fk_steering"]["beam_width"], 1)
        spec["search"]["mcts"]["n_simulations"] = min(spec["search"]["mcts"]["n_simulations"], 2)
    else:
        spec["search"]["best_of_n"]["replicas"] = min(spec["search"]["best_of_n"]["replicas"], stage2_replica_cap)
        spec["search"]["beam_search"]["n_branch"] = min(spec["search"]["beam_search"]["n_branch"], stage2_replica_cap)
        spec["search"]["beam_search"]["beam_width"] = min(spec["search"]["beam_search"]["beam_width"], stage2_replica_cap)
        spec["search"]["fk_steering"]["n_branch"] = min(spec["search"]["fk_steering"]["n_branch"], stage2_replica_cap)
        spec["search"]["fk_steering"]["beam_width"] = min(spec["search"]["fk_steering"]["beam_width"], stage2_replica_cap)
    return spec


def _resolve_checkpoint(spec: dict[str, Any]) -> tuple[Path, str, Path]:
    ckpt_root = Path(os.environ.get("CKPT_PATH", str(PROTEINA_ROOT / "ckpts"))).expanduser()
    selection = spec["checkpoint_selection"]
    ckpt_name = _ALLOWED_CHECKPOINT_ALIASES.get(selection, selection)
    if "/" in ckpt_name or "\\" in ckpt_name or not ckpt_name.endswith(".ckpt"):
        raise CandidateValidationError("checkpoint_selection must be an allowed alias or checkpoint filename")
    ckpt_file = ckpt_root / ckpt_name
    ae_file = ckpt_root / "complexa_ae.ckpt"
    return ckpt_root, ckpt_name, ae_file


def _hydra_list(values: list[Any]) -> str:
    return json.dumps(values, separators=(",", ":"))


def _build_overrides(spec: dict[str, Any], root_path: Path) -> list[str]:
    ckpt_root, ckpt_name, ae_file = _resolve_checkpoint(spec)
    sampling = spec["sampling"]
    search = spec["search"]
    computed_max_batch = max(
        spec["batch_size"],
        spec["batch_size"] * spec["num_length_samples"],
        spec["batch_size"] * spec["num_length_samples"] * search["best_of_n"]["replicas"],
        spec["batch_size"]
        * search["beam_search"]["n_branch"]
        * search["beam_search"]["beam_width"],
    )
    # Cap peak GPU memory on consumer GPUs (e.g. 24 GB).  When max_batch_size is
    # too large, BestOfN skips chunking and runs the full expanded batch at once.
    max_batch_cap = _as_int(os.environ.get("SKYDISCOVER_BINDER_MAX_BATCH_SIZE", 4), "max_batch_cap", 1, 32)
    max_batch_size = min(computed_max_batch, max_batch_cap)

    return [
        f"++root_path={root_path}",
        "++run_name=skydiscover_3di3_il7ra",
        "++gen_njobs=1",
        f"++seed={spec['seed']}",
        f"++ckpt_path={ckpt_root}",
        f"++ckpt_name={ckpt_name}",
        f"++autoencoder_ckpt_path={ae_file}",
        f"++generation.task_name={TARGET_NAME}",
        f"++generation.target_dict_cfg.{TARGET_NAME}.source=skydiscover",
        f"++generation.target_dict_cfg.{TARGET_NAME}.target_filename=3di3_chain_b_il7ra_target",
        f"++generation.target_dict_cfg.{TARGET_NAME}.target_path={TARGET_PDB_PATH}",
        f"++generation.target_dict_cfg.{TARGET_NAME}.target_input={TARGET_METADATA['target']['proteina_target_input']}",
        f"++generation.target_dict_cfg.{TARGET_NAME}.hotspot_residues={_hydra_list(spec['hotspot_residues'])}",
        f"++generation.target_dict_cfg.{TARGET_NAME}.binder_length={_hydra_list(spec['binder_length'])}",
        f"++generation.target_dict_cfg.{TARGET_NAME}.pdb_id=3di3",
        f"++generation.dataloader.batch_size={spec['batch_size']}",
        f"++generation.dataloader.dataset.nres.nsamples={spec['num_length_samples']}",
        f"++generation.args.nsteps={spec['denoising_steps']}",
        f"++generation.args.self_cond={str(spec['self_conditioning']).lower()}",
        f"++generation.search.algorithm={search['algorithm']}",
        f"++generation.search.max_batch_size={max_batch_size}",
        f"++generation.search.step_checkpoints={_hydra_list([0, spec['denoising_steps']])}",
        f"++generation.search.best_of_n.replicas={search['best_of_n']['replicas']}",
        f"++generation.search.beam_search.n_branch={search['beam_search']['n_branch']}",
        f"++generation.search.beam_search.beam_width={search['beam_search']['beam_width']}",
        f"++generation.search.fk_steering.n_branch={search['fk_steering']['n_branch']}",
        f"++generation.search.fk_steering.beam_width={search['fk_steering']['beam_width']}",
        f"++generation.search.fk_steering.temperature={search['fk_steering']['temperature']}",
        f"++generation.search.mcts.n_simulations={search['mcts']['n_simulations']}",
        f"++generation.search.mcts.exploration_prob={search['mcts']['exploration_prob']}",
        f"++generation.search.mcts.exploration_constant={search['mcts']['exploration_constant']}",
        f"++generation.model.bb_ca.schedule.mode={sampling['bb_ca_schedule']['mode']}",
        f"++generation.model.bb_ca.schedule.p={sampling['bb_ca_schedule']['p']}",
        f"++generation.model.local_latents.schedule.mode={sampling['local_latents_schedule']['mode']}",
        f"++generation.model.local_latents.schedule.p={sampling['local_latents_schedule']['p']}",
        f"++generation.model.bb_ca.gt.mode={sampling['bb_ca_noise']['mode']}",
        f"++generation.model.bb_ca.gt.p={sampling['bb_ca_noise']['p']}",
        f"++generation.model.local_latents.gt.mode={sampling['local_latents_noise']['mode']}",
        f"++generation.model.local_latents.gt.p={sampling['local_latents_noise']['p']}",
        f"++generation.model.bb_ca.simulation_step_params.sc_scale_noise={sampling['bb_ca_sc_scale_noise']}",
        f"++generation.model.local_latents.simulation_step_params.sc_scale_noise={sampling['local_latents_sc_scale_noise']}",
        f"++generation.model.bb_ca.simulation_step_params.sc_scale_score={sampling['bb_ca_sc_scale_score']}",
        f"++generation.model.local_latents.simulation_step_params.sc_scale_score={sampling['local_latents_sc_scale_score']}",
    ]


def _check_runtime_requirements(spec: dict[str, Any]) -> None:
    if not PROTEINA_ROOT.exists():
        raise RuntimeError(f"Proteina-Complexa repo not found: {PROTEINA_ROOT}")
    if not (PROTEINA_ROOT / "configs" / f"{PROTEINA_CONFIG_NAME}.yaml").exists():
        raise RuntimeError(f"Proteina config not found: {PROTEINA_CONFIG_NAME}")
    if not TARGET_PDB_PATH.exists():
        raise RuntimeError(f"target PDB not found: {TARGET_PDB_PATH}")

    ckpt_root, ckpt_name, ae_file = _resolve_checkpoint(spec)
    ckpt_file = ckpt_root / ckpt_name
    missing = []
    if not ckpt_file.exists():
        missing.append(str(ckpt_file))
    if not ae_file.exists():
        missing.append(str(ae_file))
    af2_dir = os.environ.get("AF2_DIR")
    if not af2_dir or not Path(af2_dir).exists():
        missing.append("AF2_DIR")
    if missing:
        raise RuntimeError(
            "missing Proteina runtime assets: "
            + ", ".join(missing)
            + ". Set CKPT_PATH and AF2_DIR, or run with SKYDISCOVER_BINDER_VALIDATE_ONLY=1 for schema checks."
        )


def _run_proteina_generate(spec: dict[str, Any], *, stage: str) -> dict[str, Any]:
    if os.environ.get("SKYDISCOVER_BINDER_VALIDATE_ONLY") == "1":
        return {
            "combined_score": 0.0,
            "validation_only": 1.0,
            "num_valid_designs": 0,
            "best_total_reward": 0.0,
            "mean_total_reward": 0.0,
            "selected_hotspots": ",".join(spec["hotspot_residues"]),
            "stage": stage,
        }

    _check_runtime_requirements(spec)
    timeout = int(os.environ.get("SKYDISCOVER_BINDER_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))

    with tempfile.TemporaryDirectory(prefix=f"skydiscover_binder_{stage}_") as tmp:
        root_path = Path(tmp) / "generation"
        root_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(PROTEINA_PYTHON),
            "-m",
            "proteinfoundation.generate",
            "--config-path",
            str(PROTEINA_CONFIG_DIR),
            "--config-name",
            PROTEINA_CONFIG_NAME,
            *_build_overrides(spec, root_path),
        ]
        env = os.environ.copy()
        env["HYDRA_FULL_ERROR"] = "1"
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        src_path = str(PROTEINA_ROOT / "src")
        env["PYTHONPATH"] = src_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

        start = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(PROTEINA_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        elapsed = time.time() - start
        if proc.returncode != 0:
            raise RuntimeError(
                f"Proteina generation failed with exit code {proc.returncode}\n"
                f"STDOUT:\n{proc.stdout[-4000:]}\nSTDERR:\n{proc.stderr[-4000:]}"
            )
        reward_csvs = sorted(root_path.rglob("rewards_*.csv"))
        if not reward_csvs:
            raise RuntimeError(f"Proteina generation completed but no rewards_*.csv found under {root_path}")
        return _score_reward_csv(reward_csvs[-1], spec=spec, elapsed=elapsed, stage=stage)


def _finite_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _score_reward_csv(csv_path: Path, *, spec: dict[str, Any], elapsed: float, stage: str) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    reward_rows = [(row, _finite_float(row.get("total_reward"))) for row in rows]
    reward_rows = [(row, reward) for row, reward in reward_rows if reward is not None]
    if not reward_rows:
        raise RuntimeError(f"no finite total_reward values found in {csv_path}")

    rewards = [reward for _, reward in reward_rows]
    best_row, best_reward = max(reward_rows, key=lambda item: item[1])
    mean_reward = sum(rewards) / len(rewards)
    # Proteina's default AF2 binder reward is often negative interface PAE loss.
    # Convert it to a bounded higher-is-better score while preserving ranking.
    penalty = max(0.0, -best_reward)
    combined = 1.0 / (1.0 + penalty)
    diversity_bonus = min(len({row.get("aatype", "") for row, _ in reward_rows}), 10) / 100.0
    combined_score = min(1.0, combined + diversity_bonus)

    numeric_components: dict[str, float] = {}
    for key in best_row:
        if key in {"pdb_path", "aatype", "sample_type", "metadata_tag"}:
            continue
        value = _finite_float(best_row.get(key))
        if value is not None:
            numeric_components[f"best_{key}"] = value

    return {
        "combined_score": float(combined_score),
        "best_total_reward": float(best_reward),
        "mean_total_reward": float(mean_reward),
        "num_valid_designs": float(len(reward_rows)),
        "eval_time": float(elapsed),
        "stage": stage,
        "selected_hotspot_count": float(len(spec["hotspot_residues"])),
        "binder_min_len": float(spec["binder_length"][0]),
        "binder_max_len": float(spec["binder_length"][1]),
        "denoising_steps": float(spec["denoising_steps"]),
        "selected_hotspots": ",".join(spec["hotspot_residues"]),
        "search_algorithm": spec["search"]["algorithm"],
        "reward_csv": str(csv_path),
        **numeric_components,
    }


def _failure(exc: Exception, *, stage: str) -> dict[str, Any]:
    return {
        "combined_score": 0.0,
        "best_total_reward": float("nan"),
        "mean_total_reward": float("nan"),
        "num_valid_designs": 0.0,
        "stage": stage,
        "error": str(exc),
    }


def _evaluate(program_path: str, *, stage: str) -> dict[str, Any]:
    try:
        raw_spec = _load_candidate_spec(program_path)
        spec = _normalise_spec(raw_spec, stage=stage)
        result = _run_proteina_generate(spec, stage=stage)
        result["spec_json"] = json.dumps(spec, sort_keys=True)
        print(
            "Evaluation: "
            f"stage={stage}, combined_score={result['combined_score']:.6f}, "
            f"best_total_reward={result['best_total_reward']}, "
            f"hotspots={result.get('selected_hotspots', '')}"
        )
        return result
    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        traceback.print_exc()
        return _failure(exc, stage=stage)


def evaluate_stage1(program_path: str) -> dict[str, Any]:
    """Fast cascade stage for schema validation and a tiny Proteina run."""
    return _evaluate(program_path, stage="stage1")


def evaluate_stage2(program_path: str) -> dict[str, Any]:
    """Full benchmark stage for promising candidates."""
    return _evaluate(program_path, stage="stage2")


def evaluate(program_path: str) -> dict[str, Any]:
    """Evaluate one evolved Proteina binder-design run specification."""
    return evaluate_stage2(program_path)


if __name__ == "__main__":
    default_program = BENCHMARK_DIR / "initial_program.py"
    candidate = sys.argv[1] if len(sys.argv) > 1 else str(default_program)
    print(json.dumps(evaluate(candidate), indent=2, sort_keys=True))
