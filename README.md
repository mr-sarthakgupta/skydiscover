<h1 align="center">
  
  <b>LLM Agent + Genetic Search for Binder Design</b>
</h1>

This fork branch applies **SkyDiscover**, an existing LLM-agent plus genetic-search
package, to protein binder design. The upstream project is
[`skydiscover-ai/skydiscover`](https://github.com/skydiscover-ai/skydiscover),
a modular framework for AI-driven scientific and algorithmic discovery across
200+ optimization tasks.

In this branch, the binder-design workflow lives in
[`protein_binder_design/`](protein_binder_design/). An LLM agent proposes and
revises Proteina-Complexa run specifications, while AdaEvolve genetic search
explores hotspots, binder lengths, sampling schedules, and test-time search
settings for the 3DI3 IL-7Ralpha system.

SkyDiscover's core adaptive algorithms are:

- **[AdaEvolve](https://arxiv.org/abs/2602.20133)**, which dynamically adjusts its optimization behavior based on observed progress.
- **[EvoX](https://arxiv.org/abs/2602.23413)**, which dynamically evolves the optimization (evolution) strategy itself using LLMs on the fly.

<p align="center"> Binder Design Branch: Using SkyDiscover for Protein Binder Search</p>
  <p align="center">
  <a href="https://skydiscover-ai.github.io/blog.html"><img src="https://img.shields.io/badge/blog-SkyDiscover-orange?style=flat-square" alt="Blog" /></a>
  <a href="https://arxiv.org/abs/2602.20133"><img src="https://img.shields.io/badge/paper-AdaEvolve-red?style=flat-square" alt="AdaEvolve Paper" /></a>
  <a href="https://arxiv.org/abs/2602.23413"><img src="https://img.shields.io/badge/paper-EvoX-lightblue?style=flat-square" alt="EvoX Paper" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" /></a>
  </p>



   <p align="center">
  <img src="assets/architecture.png" width="720" alt="SkyDiscover architecture"><br>
</p>

---

## Binder Design Workflow

The [`protein_binder_design/`](protein_binder_design/) directory packages this
branch's 3DI3 IL-7Ralpha binder search loop:

- `initial_program.py` is the seed candidate the search mutates.
- `config.yaml` enables the LLM agent and AdaEvolve genetic search.
- `evaluator.py` validates each candidate, launches Proteina-Complexa, and scores
  generated binders from Proteina reward CSVs.
- `assets/` contains the cleaned target PDB, raw 3DI3 coordinates, and hotspot
  metadata used by the evaluator.

The evolved program returns a constrained Python dictionary rather than running
models directly. That keeps model execution, GPU caps, checkpoint selection, and
reward parsing inside the evaluator while the LLM-agent search focuses on
scientific design choices.
