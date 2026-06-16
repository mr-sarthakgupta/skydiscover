"""Download and prepare 3DI3 assets for the binder-design benchmark."""

from __future__ import annotations

import json
import math
import urllib.request
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BENCHMARK_DIR / "assets"
RCSB_FILES = {
    "3di3.pdb": "https://files.rcsb.org/download/3DI3.pdb",
    "3di3.cif": "https://files.rcsb.org/download/3DI3.cif",
}


def _download_assets() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in RCSB_FILES.items():
        dest = ASSETS_DIR / name
        if dest.exists() and dest.stat().st_size > 0:
            continue
        with urllib.request.urlopen(url, timeout=60) as response:
            dest.write_bytes(response.read())


def _parse_atom(line: str) -> dict:
    return {
        "atom": line[12:16].strip(),
        "resname": line[17:20].strip(),
        "chain": line[21].strip(),
        "resseq": int(line[22:26]),
        "icode": line[26].strip(),
        "x": float(line[30:38]),
        "y": float(line[38:46]),
        "z": float(line[46:54]),
        "element": line[76:78].strip() if len(line) >= 78 else "",
    }


def prepare_assets() -> dict:
    _download_assets()
    pdb_path = ASSETS_DIR / "3di3.pdb"
    lines = pdb_path.read_text().splitlines()

    chain_b_lines = [
        line for line in lines if line.startswith("ATOM  ") and len(line) > 21 and line[21] == "B"
    ]
    if not chain_b_lines:
        raise RuntimeError("No chain B ATOM records found in downloaded 3DI3 PDB")

    target_pdb = ASSETS_DIR / "3di3_chain_b_il7ra_target.pdb"
    target_pdb.write_text("\n".join(chain_b_lines + ["TER", "END"]) + "\n")

    atoms = [_parse_atom(line) for line in lines if line.startswith("ATOM  ")]
    protein_atoms = [atom for atom in atoms if atom["element"] != "H"]
    chain_a = [atom for atom in protein_atoms if atom["chain"] == "A"]
    chain_b = [atom for atom in protein_atoms if atom["chain"] == "B"]

    residues = []
    seen = set()
    for atom in chain_b:
        key = (atom["chain"], atom["resseq"], atom["icode"])
        if key in seen:
            continue
        seen.add(key)
        residues.append(
            {
                "hotspot": f"{atom['chain']}{atom['resseq']}{atom['icode']}",
                "chain": atom["chain"],
                "res_id": atom["resseq"],
                "insertion_code": atom["icode"],
                "resname": atom["resname"],
            }
        )

    interface = {}
    cutoff2 = 25.0
    for b_atom in chain_b:
        best2 = None
        for a_atom in chain_a:
            d2 = (
                (b_atom["x"] - a_atom["x"]) ** 2
                + (b_atom["y"] - a_atom["y"]) ** 2
                + (b_atom["z"] - a_atom["z"]) ** 2
            )
            if d2 <= cutoff2 and (best2 is None or d2 < best2):
                best2 = d2
        if best2 is None:
            continue
        key = (b_atom["chain"], b_atom["resseq"], b_atom["icode"])
        entry = interface.setdefault(
            key,
            {
                "hotspot": f"{b_atom['chain']}{b_atom['resseq']}{b_atom['icode']}",
                "resname": b_atom["resname"],
                "min_distance_angstrom": math.sqrt(best2),
            },
        )
        entry["min_distance_angstrom"] = min(entry["min_distance_angstrom"], math.sqrt(best2))

    allowed_hotspots = [residue["hotspot"] for residue in residues]
    prior_hotspots = [hotspot for hotspot in ["B62", "B84", "B143"] if hotspot in allowed_hotspots]
    metadata = {
        "pdb_id": "3DI3",
        "target": {
            "name": "human interleukin-7 receptor alpha ectodomain",
            "chain_id": "B",
            "proteina_target_input": f"B{residues[0]['res_id']}-{residues[-1]['res_id']}",
            "target_pdb": target_pdb.name,
            "modeled_residue_count_chain_b": len(residues),
            "allowed_hotspots": allowed_hotspots,
        },
        "partner_chain": {"chain_id": "A", "name": "human interleukin-7"},
        "prior_art_hotspots": {
            "source": "OpenProtein/RFdiffusion IL-7Ralpha walkthrough, original chain-B positions",
            "hotspots": prior_hotspots,
            "requested_behavior": "Use as a seed/default only; SkyDiscover candidates may choose hotspot residues.",
        },
        "interface_hotspot_candidates": sorted(
            interface.values(),
            key=lambda item: (item["min_distance_angstrom"], item["hotspot"]),
        ),
        "notes": [
            "Raw 3DI3 PDB and mmCIF files are stored beside this metadata.",
            "Chain B target PDB excludes glycan chains C-E and chain A.",
            "Interface candidates are heavy-atom contacts within 5.0 A between chains A and B in the downloaded PDB.",
        ],
    }
    (ASSETS_DIR / "target_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    result = prepare_assets()
    print(
        json.dumps(
            {
                "target_pdb": result["target"]["target_pdb"],
                "modeled_residue_count_chain_b": result["target"]["modeled_residue_count_chain_b"],
                "interface_candidates": len(result["interface_hotspot_candidates"]),
                "prior_hotspots": result["prior_art_hotspots"]["hotspots"],
            },
            indent=2,
        )
    )
