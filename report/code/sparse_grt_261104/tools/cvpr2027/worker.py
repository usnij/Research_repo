"""One GPU, one frozen E1/E2 queue; never retry a partial run automatically."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

from common import (STEPS, eval_command, load_plan, now, read, run_signature,
                    selected_node, sha, train_command, verify_code, write)


@contextmanager
def machine_lock(root):
    # Shared with the prior benchmark launcher, including other working roots.
    path = root.parent / ".sparse-grt-gpu0.lock"
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker holds the GPU lock: {path}") from exc
        yield


def gpu_inventory():
    command = ["nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total,power.limit",
               "--format=csv,noheader,nounits"]
    output = subprocess.check_output(command, text=True).strip()
    lines = output.splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"Exactly one GPU is required: {output}")
    fields = [item.strip() for item in lines[0].split(",")]
    if len(fields) != 5 or "RTX 4090" not in fields[1]:
        raise RuntimeError(f"An RTX 4090 is required: {output}")
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                      "--format=csv,noheader"], text=True).strip()
    if active and "No running processes found" not in active:
        raise RuntimeError(f"GPU is occupied; refusing concurrent work: {active}")
    return dict(zip(("uuid", "name", "driver", "memory_mib", "power_limit_w"), fields))


def gate(root, plan, run):
    if run["run_id"] == plan["first_run_id"]:
        baseline = read(root / ".baseline_reevaluation_verified.json")
        if baseline.get("passed") is not True:
            raise RuntimeError("The prior Counter checkpoint has not passed reevaluation")
        return
    ack = read(root / ".first_2k_gate.json")
    required = dict(plan_signature=plan["plan_signature"], first_run_id=plan["first_run_id"],
                    checkpoint_verified=True, step=2000)
    if any(ack.get(key) != value for key, value in required.items()):
        raise RuntimeError("First 2k checkpoint acknowledgement is absent or for another plan")


def preflight(root, plan, node):
    expected_entry = root / "code" / "tools" / "cvpr2027" / "worker.py"
    if Path(__file__).resolve() != expected_entry.resolve():
        raise RuntimeError("Run the committed code/tools/cvpr2027/worker.py, not a staged copy")
    ready = read(root / ".environment_ready.json")
    if ready.get("passed") is not True or ready.get("decoder_verified") is not True:
        raise RuntimeError("Environment and JPEG decoder verification have not passed")
    if ready.get("frozen_commit") != plan["frozen_commit"]:
        raise RuntimeError("Environment readiness belongs to a different source commit")
    expected_scenes = {run["scene"] for run in node["runs"]}
    if not expected_scenes.issubset(set(ready.get("datasets_verified", []))):
        raise RuntimeError("The assigned scene datasets have not all been verified")
    verify_code(root, plan)
    for run in node["runs"]:
        config = root / "fleet" / run["config_path"]
        if sha(config) != run["config_sha256"] or read(config) != run["method"]:
            raise RuntimeError(f"Frozen method config mismatch: {run['run_id']}")
        if run["experiment_id"] not in plan["allow_experiments"]:
            raise RuntimeError("An unapproved experiment was inserted into the queue")
        if not (root / "data" / run["scene"] / "sparse" / "0" / "images.bin").is_file():
            raise RuntimeError(f"Scene is not staged: {run['scene']}")
    if os.environ.get("GRT_GRAD_ACCUM_DOUBLE") != "1":
        raise RuntimeError("Activate the reference environment (GRT_GRAD_ACCUM_DOUBLE=1)")
    return gpu_inventory()


def execute(command, log, code, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise FileExistsError(f"Refusing to overwrite an execution log: {log}")
    with log.open("x", encoding="utf-8") as output:
        child = subprocess.Popen(command, cwd=code, env=env, stdout=output,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        def stop(signum, _frame):
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass
            raise KeyboardInterrupt(f"Worker received signal {signum}")
        previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            code_number = child.wait()
            if code_number:
                raise subprocess.CalledProcessError(code_number, command)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=30)


def verify_training(out, run):
    completed = read(out / "training_complete.json")
    if completed.get("passed") is not True or completed.get("steps") != 30000:
        raise RuntimeError("Training did not complete exactly 30,000 steps")
    if completed.get("N") != run["method"]["cap"]:
        raise RuntimeError("Final Gaussian count differs from the registered cap")
    protocol = read(out / "protocol.json")
    expected = dict(method=run["method"], train_views=run["train_views"],
                    growth="log_horizon", growth_end=5000, structural_loss=run["structural_loss"])
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Actual training protocol differs from the frozen run")
    if protocol.get("observed_only") is not False or protocol.get("multiview_start_step") != 0:
        raise RuntimeError("Regularizer support or MV start differs")
    if read(out / "initial.json")["N"] != run["initial_N"]:
        raise RuntimeError("Initial SfM point count differs")
    geometry = read(out / "ray_geometry.json")
    actual = {(value["width"], value["height"], value["total_rays_if_equal_views"])
              for value in geometry.values()}
    expected_geometry = {(*map(int, shape.split("x")), count)
                         for shape, count in run["rays_by_shape"].items()}
    if actual != expected_geometry or len(geometry) != run["train_views"]:
        raise RuntimeError("Actual rays/resolution differ from the registered input geometry")
    for step in STEPS:
        checkpoint = out / f"ckpt_{step}.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError(f"Missing required checkpoint: {step}")
        proof = read(out / f"checkpoint_verified_{step}.json")
        if proof.get("step") != step or proof.get("load_verified") is not True or proof.get("optimizer_present") is not True:
            raise RuntimeError(f"Checkpoint did not pass reload/optimizer verification: {step}")
    return completed


def verify_evaluation(out, run, step):
    value = read(out / f"evaluation_{step}.json")
    if value.get("step") != step or value.get("views") != run["test_views"]:
        raise RuntimeError(f"Evaluation step/view coverage differs: {step}")
    if value.get("run_id") != run["run_id"]:
        raise RuntimeError("Evaluation belongs to another run")
    if value.get("checkpoint_sha256") != sha(out / f"ckpt_{step}.pt"):
        raise RuntimeError("Evaluated checkpoint hash mismatch")
    if step == 30000 and not value.get("optimizer_state_present"):
        raise RuntimeError("30k checkpoint has no optimizer state")
    if not all(math.isfinite(value[key]) for key in ("PSNR", "SSIM", "LPIPS")):
        raise RuntimeError("Non-finite evaluation metrics")
    return value


def receipt(root, plan, run, hardware, completed):
    out = root / "runs" / run["run_id"]
    values = {step: verify_evaluation(out, run, step) for step in STEPS}
    artifacts = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "checkpoint_latest.pt" and path.suffix in (".json", ".jsonl", ".csv", ".pt"):
            artifacts.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path)))
    return dict(passed=True, run_id=run["run_id"], run_signature=run_signature(plan, run),
                plan_signature=plan["plan_signature"], frozen_commit=plan["frozen_commit"],
                hardware=hardware, training=completed,
                final_metrics={k: values[30000][k] for k in ("PSNR", "SSIM", "LPIPS")},
                files=artifacts, complete_utc=now())


def run_queue(args):
    root = args.root.resolve()
    plan = load_plan(root / "fleet" / "plan.json")
    node = selected_node(plan, args.node)
    hardware = preflight(root, plan, node)
    write(root / "hardware.json", dict(**hardware, node_id=node["node_id"],
                                      instance_id=node["instance_id"], recorded_utc=now()))
    env = dict(os.environ, PYTHONPATH=str(root / "code" / "src"), PYTHONUNBUFFERED="1")
    done = []
    for run in node["runs"]:
        run_id = run["run_id"]
        out = root / "runs" / run_id
        record_path = root / "receipts" / f"{run_id}.json"
        if record_path.exists():
            existing = read(record_path)
            if not existing.get("passed") or existing.get("run_signature") != run_signature(plan, run):
                raise RuntimeError(f"Existing run receipt conflicts with this plan: {run_id}")
            for file in existing["files"]:
                path = root / file["path"]
                if not path.is_file() or sha(path) != file["sha256"]:
                    raise RuntimeError(f"Archived run artifacts changed: {path}")
            done.append(run_id)
            continue
        gate(root, plan, run)
        gpu_inventory()
        if shutil.disk_usage(root).free < 12 * 2**30 + 7000 * run["method"]["cap"]:
            raise RuntimeError("Insufficient space for seven checkpoints and optimizer states")
        write(root / "status.json", dict(phase="training", run_id=run_id, completed=done,
                                        node_id=args.node, updated_utc=now()))
        pending = root / "run_manifests" / f"{run_id}.json"
        if out.exists():
            if not args.resume_evaluation:
                raise RuntimeError(f"Partial run exists; manual investigation required: {out}")
            if not pending.exists() or read(pending)["run_signature"] != run_signature(plan, run):
                raise RuntimeError("Cannot resume evaluation for an unregistered run")
            completed = verify_training(out, run)
        else:
            if pending.exists():
                raise RuntimeError("An earlier launch was attempted; no automatic rerun")
            write(pending, dict(run=run, run_signature=run_signature(plan, run),
                                frozen_commit=plan["frozen_commit"], hardware=hardware,
                                node_id=args.node, instance_id=node["instance_id"], launch_utc=now()))
            execute(train_command(root, run, plan["python"]), root / "logs" / f"{run_id}.train.log",
                    root / "code", env)
            completed = verify_training(out, run)
        write(root / "status.json", dict(phase="evaluation", run_id=run_id, completed=done,
                                        node_id=args.node, updated_utc=now()))
        for step in STEPS:
            evaluation = out / f"evaluation_{step}.json"
            if evaluation.exists():
                if not args.resume_evaluation:
                    raise RuntimeError("Unexpected preexisting evaluation output")
                verify_evaluation(out, run, step)
                continue
            execute(eval_command(root, run, step, plan["python"]),
                    root / "logs" / f"{run_id}.eval{step}.log", root / "code", env)
            verify_evaluation(out, run, step)
        write(record_path, receipt(root, plan, run, hardware, completed))
        done.append(run_id)
        write(root / "status.json", dict(phase="run_complete", run_id=run_id, completed=done,
                                        node_id=args.node, updated_utc=now()))
    write(root / "status.json", dict(phase="complete", completed=done, node_id=args.node, updated_utc=now()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--resume-evaluation", action="store_true",
                        help="Explicitly finish evaluation only after a reviewed evaluation failure")
    args = parser.parse_args()
    # A duplicate launcher must not overwrite the active worker's status file.
    with machine_lock(args.root.resolve()):
        try:
            run_queue(args)
        except BaseException as error:
            write(args.root / "status.json", dict(phase="failed", node_id=args.node, updated_utc=now(),
                                                 error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    main()
