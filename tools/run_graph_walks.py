# Copyright (c) 2026 Nick van der Merwe
"""Run native GraphWalker walks and retain one artifact set per seed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WalkResult:
    seed: int
    status: str
    path: str
    observation: str
    log: str
    error: str = ""


def _run_seed(seed: int, args: argparse.Namespace, graph_hash: str) -> WalkResult:
    stem = f"seed-{seed}"
    path = args.artifacts / f"{stem}.path.jsonl"
    observation = args.artifacts / f"{stem}.observation.json"
    log = args.artifacts / f"{stem}.pytest.txt"
    command = ["java", "-jar", str(args.jar), "offline", "-d", str(seed), "-m", str(args.model), f"random(length({args.steps}))"]
    generated = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603
    if generated.returncode:
        error = generated.stderr.strip() or generated.stdout.strip() or "GraphWalker failed"
        return _write_result(WalkResult(seed, "path-failed", str(path), str(observation), str(log), error), args, graph_hash)
    path.write_text(generated.stdout, encoding="utf-8")
    environment = {
        **os.environ,
        "GRAPHWALKER_PATH": str(path),
        "GRAPHWALKER_SEED": str(seed),
        "GRAPHWALKER_ARTIFACT": str(observation),
    }
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to run graph walk verification")
    tested = subprocess.run([uv, "run", "pytest", "-q", "tests/test_graph_http.py"], capture_output=True, text=True, env=environment, check=False)  # noqa: S603
    log.write_text(tested.stdout + tested.stderr, encoding="utf-8")
    status = "passed" if tested.returncode == 0 else "failed"
    error = "" if status == "passed" else (tested.stdout + tested.stderr).strip()[-4_000:]
    return _write_result(WalkResult(seed, status, str(path), str(observation), str(log), error), args, graph_hash)


def _write_result(result: WalkResult, args: argparse.Namespace, graph_hash: str) -> WalkResult:
    summary = {
        "seed": result.seed,
        "status": result.status,
        "graph_sha256": graph_hash,
        "transitions": args.steps,
        "path": result.path,
        "observation": result.observation,
        "pytest_log": result.log,
        "error": result.error,
    }
    (args.artifacts / f"seed-{result.seed}.summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jar", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, default=2026)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--parallelism", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    return parser


def main() -> int:
    args = _parser().parse_args()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    graph_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
    seeds = range(args.start_seed, args.start_seed + args.count)
    with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
        futures = [executor.submit(_run_seed, seed, args, graph_hash) for seed in seeds]
        results = sorted((future.result() for future in as_completed(futures)), key=lambda result: result.seed)
    (args.artifacts / "summary.json").write_text(
        json.dumps({"graph_sha256": graph_hash, "results": [asdict(result) for result in results]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    passed = sum(result.status == "passed" for result in results)
    sys.stdout.write(f"graph walks: {passed}/{len(results)} passed; artifacts={args.artifacts}\n")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
