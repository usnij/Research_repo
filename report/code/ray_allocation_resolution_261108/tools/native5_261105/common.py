"""Shared, dependency-free bookkeeping for the independent native-5-percent single-view campaign."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


STEPS = (2000, 5000, 7000, 10000, 15000, 20000, 30000)
ARM_SPECS = (("dense", "dense", 1, 6.0, 11),
             ("grid11", "grid", 3, 6.0, 11),
             ("grid4", "grid", 3, 6.0, 4),
             ("coordinate_sigma2", "coordinate", 3, 2.0, 11),
             ("coordinate_sigma3", "coordinate", 3, 3.0, 11),
             ("coordinate_sigma4", "coordinate", 3, 4.0, 11),
             ("coordinate_sigma1p5", "coordinate", 3, 1.5, 11),
             ("coordinate_sigma1", "coordinate", 3, 1.0, 11))



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
    if plan.get("campaign") != "native5_singleview_261105" or plan.get("allow_experiments") != ["N5"]:
        raise ValueError("This launcher only accepts the independent N5 campaign")
    validate_arm_sequence(plan)
    return plan


def validate_arm_sequence(plan):
    nodes = plan.get("nodes", [])
    if len(nodes) != 3 or {n.get("primary_scene") for n in nodes} != {"bicycle", "bonsai", "garden"}:
        raise ValueError("N5 requires exactly the three registered scene owners")
    if nodes[0].get("primary_scene") != "bicycle":
        raise ValueError("Bicycle dense must own the first 2k checkpoint gate")
    for node in nodes:
        runs = node.get("runs", [])
        if len(runs) != len(ARM_SPECS):
            raise ValueError("Each N5 scene requires eight arms, including sigma1.5 then sigma1.0")
        for run, (arm, loss, block, sigma, window) in zip(runs, ARM_SPECS):
            method = run.get("method", {})
            expected = dict(views=1, block=block, sigma=sigma, grid_window=window,
                            grid_sigma=1.5, steps=30000, initialization_seed=777)
            if (run.get("scene") != node["primary_scene"] or run.get("experiment_id") != "N5"
                    or run.get("arm") != arm or run.get("structural_loss") != loss
                    or any(method.get(key) != value for key, value in expected.items())):
                raise ValueError(f"N5 arm order or frozen configuration differs: {run.get('run_id')}")
    if plan.get("first_run_id") != nodes[0]["runs"][0]["run_id"]:
        raise ValueError("First 2k gate must refer to the first Bicycle dense run")


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
            "native_5percent", "--budget-views", "all", "--cache-mib", "1200", "--mode", "train",
            "--keep-checkpoints", ",".join(map(str, STEPS)), "--protocol", "native5_singleview",
            "--structural-loss", run["structural_loss"]]


def eval_command(root, run, step, python):
    root = Path(root)
    out = root / "runs" / run["run_id"]
    return [str(python), "-u", str(root / "code" / "eval_full.py"), "--checkpoint",
            str(out / f"ckpt_{step}.pt"), "--data-root", str(root / "data" / run["scene"]),
            "--output", str(out / f"evaluation_{step}.json"), "--run-dir", str(out),
            "--run-id", run["run_id"]]
