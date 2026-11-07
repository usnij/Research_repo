"""Dependency-free provenance and commands for the eight capacity/duration arms."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


STEPS = (2000, 5000, 7000, 10000, 15000, 20000, 30000)
LONG_STEPS = (*STEPS, 45000, 60000)
CAPS = {"bicycle": (3021939, 6026079), "garden": (2044672, 5709543)}
NODE_SPECS = (("node01", "bicycle", ("highcap_dense",)),
              ("node02", "bicycle", ("highcap_coordinate_sigma2", "highcap_coordinate_sigma4", "long_coordinate_sigma4")),
              ("node03", "garden", ("highcap_dense",)),
              ("node04", "garden", ("highcap_coordinate_sigma2", "highcap_coordinate_sigma4", "long_coordinate_sigma4")))



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
    if (plan.get("campaign") != "native5_capacity_duration_261105"
            or plan.get("allow_experiments") != ["N5_CAP", "N5_60K"]):
        raise ValueError("This launcher only accepts the independent capacity/duration campaign")
    validate_arm_sequence(plan)
    return plan


def validate_arm_sequence(plan):
    nodes = plan.get("nodes", [])
    if len(nodes) != 4 or len({node.get("instance_id") for node in nodes}) != 4:
        raise ValueError("Capacity/duration requires four distinct one-GPU instances")
    for node, (node_id, scene, arms) in zip(nodes, NODE_SPECS):
        if node.get("node_id") != node_id or node.get("primary_scene") != scene:
            raise ValueError("Capacity/duration node order and scene assignment differ")
        runs = node.get("runs", [])
        if len(runs) != len(arms):
            raise ValueError("Capacity/duration requires exactly eight authorized runs")
        for run, arm in zip(runs, arms):
            long = arm == "long_coordinate_sigma4"
            loss = "dense" if arm == "highcap_dense" else "coordinate"
            sigma = 6.0 if loss == "dense" else (2.0 if arm.endswith("sigma2") else 4.0)
            scale = 2 if long else 1
            method = run.get("method", {})
            expected = dict(views=1, block=1 if loss == "dense" else 3, sigma=sigma,
                            grid_window=11, grid_sigma=1.5, steps=30000*scale,
                            initialization_seed=777, cap=CAPS[scene][0 if long else 1],
                            l1_weight=.8, structure_weight=.2, opacity_weight=.01,
                            scale_weight=.01, boundary="all")
            if (run.get("scene") != scene or run.get("experiment_id") != ("N5_60K" if long else "N5_CAP")
                    or run.get("arm") != arm or run.get("structural_loss") != loss
                    or run.get("schedule_scale") != scale
                    or run.get("checkpoint_steps") != list(LONG_STEPS if long else STEPS)
                    or run.get("evaluation_steps") != ([30000, 45000, 60000] if long else list(STEPS))
                    or any(method.get(key) != value for key, value in expected.items())):
                raise ValueError(f"Capacity/duration arm order or frozen configuration differs: {run.get('run_id')}")
            expected_counts = {str(step): native_count(run["initial_N"], method["cap"], step, scale)
                               for step in run["checkpoint_steps"]}
            if run.get("expected_G_by_checkpoint") != expected_counts:
                raise ValueError(f"Expected Gaussian counts differ from the native growth schedule: {run.get('run_id')}")
    if plan.get("first_run_id") != nodes[0]["runs"][0]["run_id"]:
        raise ValueError("First 2k gate must refer to the first Bicycle dense run")


def native_count(initial, cap, completed_steps, schedule_scale=1):
    """Actual backend tests a zero-based index, then saves completed step t+1."""
    n = initial
    for step in range(600, min(completed_steps, 25000*schedule_scale), 100):
        n = min(cap, int(1.05*n))
    return n


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
            "--keep-checkpoints", ",".join(map(str, run["checkpoint_steps"])),
            "--protocol", "native5_capacity_duration",
            "--structural-loss", run["structural_loss"]]


def eval_command(root, run, step, python):
    root = Path(root)
    out = root / "runs" / run["run_id"]
    return [str(python), "-u", str(root / "code" / "eval_full.py"), "--checkpoint",
            str(out / f"ckpt_{step}.pt"), "--data-root", str(root / "data" / run["scene"]),
            "--output", str(out / f"evaluation_{step}.json"), "--run-dir", str(out),
            "--run-id", run["run_id"]]
