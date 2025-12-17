from __future__ import annotations
import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Step:
    key: str
    label: str
    script: str
    default_args: list[str]


STEPS: list[Step] = [
    Step(
        key="pdf",
        label="PDF -> Text",
        script="src/preprocessing/document_loader.py",
        default_args=[],
    ),
    Step(
        key="chunk",
        label="Chunking (LLM Hybrid)",
        script="src/preprocessing/chunker.py",
        default_args=["--mode", "llm"],
    ),
    Step(
        key="schema",
        label="Schema-Generierung (LLM)",
        script="src/schema/schema_generator.py",
        default_args=[],
    ),
    Step(
        key="extract",
        label="Extraktion (LLM)",
        script="src/extraction/extractor.py",
        default_args=[],
    ),
    Step(
        key="resolve",
        label="Entity Resolution",
        script="src/consolidation/entity_resolver.py",
        default_args=[],
    ),
    Step(
        key="validate",
        label="Validierung",
        script="src/consolidation/validator.py",
        default_args=[],
    ),
    Step(
        key="neo4j",
        label="Neo4j Import",
        script="src/graph/neo4j_importer.py",
        default_args=[],
    ),
]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run full KG pipeline end-to-end.")
    p.add_argument("--from-step", type=str, default="pdf")
    p.add_argument("--to-step", type=str, default="neo4j")
    p.add_argument("--only", type=str, default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def step_index(key: str) -> int:
    keys = [s.key for s in STEPS]
    if key not in keys:
        raise ValueError(f"Unknown step '{key}'. Valid: {keys}")
    return keys.index(key)


def run_step(step: Step, overwrite: bool, dry_run: bool) -> None:
    script_path = Path(step.script)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    cmd = [sys.executable, str(script_path), *step.default_args]
    overwrite_supported = {
        "src/preprocessing/document_loader.py",
        "src/preprocessing/chunker.py",
        "src/schema/schema_generator.py",
        "src/extraction/extractor.py",
        "src/consolidation/entity_resolver.py",
        "src/consolidation/validator.py",
    }
    if overwrite and step.script in overwrite_supported:
        cmd.append("--overwrite")
    print(f"\n=== STEP {step.key}: {step.label} ===")
    print("CMD:", " ".join(cmd))
    if dry_run:
        return
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Step '{step.key}' failed with exit code {proc.returncode}")


def main() -> None:
    args = build_argparser().parse_args()
    if args.only:
        idx = step_index(args.only)
        run_step(STEPS[idx], overwrite=args.overwrite, dry_run=args.dry_run)
        return
    start = step_index(args.from_step)
    end = step_index(args.to_step)
    if start > end:
        raise ValueError("--from-step must be <= --to-step")
    if args.to_step == "neo4j" and not os.getenv("NEO4J_URI"):
        print(
            "WARN: NEO4J_URI ist nicht gesetzt. Neo4j-Step wird vermutlich fehlschlagen."
        )
    for step in STEPS[start : end + 1]:
        run_step(step, overwrite=args.overwrite, dry_run=args.dry_run)
    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
