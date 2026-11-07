"""Shared, dependency-free bookkeeping for the fixed November 4 campaign."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


STEPS = (2000, 5000, 7000, 10000, 15000, 20000, 30000)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


def signature(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path):
    plan = read(path)
    expected = plan.get("plan_signature")
    if expected != signature({k: v for k, v in plan.items() if k != "plan_signature"}):
        raise ValueError("Plan signature mismatch")
    if not plan.get("frozen_commit") or not plan.get("code_files"):
        raise ValueError("Plan has not been frozen with code provenance")
    ids = [run["run_id"] for node in plan["nodes"] for run in node["runs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate run IDs in plan")
    if plan.get("allow_experiments") != ["E1", "E2"]:
        raise ValueError("This launch stage allows only the approved E1/E2 queue")
    return plan


def run_signature(plan, run):
    return signature(dict(plan_signature=plan["plan_signature"], run=run))


def verify_code(root, plan):
    code = Path(root) / "code"
    for record in plan["code_files"]:
        path = code / record["path"]
        if not path.is_file() or sha(path) != record["sha256"]:
            raise ValueError(f"Frozen source mismatch: {path}")


def selected_node(plan, node_id):
    matches = [node for node in plan["nodes"] if node["node_id"] == node_id]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicated node: {node_id}")
    return matches[0]


def train_command(root, run, python):
    root = Path(root)
    return [str(python), "-u", "-m", "sparse_grt.training", "--scene",
            str(root / "data" / run["scene"]), "--downsample", str(run["downsample"]),
            "--config", str(root / "fleet" / run["config_path"]), "--output",
            str(root / "runs" / run["run_id"]), "--budget", "fixed", "--growth",
            "log_horizon", "--growth-end", "5000", "--cache-mib", "1200", "--mode", "train",
            "--keep-checkpoints", ",".join(map(str, STEPS)), "--protocol", "cvpr2027",
            "--structural-loss", run["structural_loss"]]


def eval_command(root, run, step, python):
    root = Path(root)
    out = root / "runs" / run["run_id"]
    return [str(python), "-u", str(root / "code" / "eval_full.py"), "--checkpoint",
            str(out / f"ckpt_{step}.pt"), "--data-root", str(root / "data" / run["scene"]),
            "--output", str(out / f"evaluation_{step}.json"), "--run-dir", str(out),
            "--run-id", run["run_id"]]
