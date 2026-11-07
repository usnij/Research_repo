"""One GPU, one frozen three-arm micropatch queue; training precedes evaluation."""
import argparse
import csv
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

from common import (eval_command, load_plan, native_count, now, read, run_signature,
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
        # The first authorized Garden sigma1.5 run is the checkpoint-save probe. The root
        # operator checks its actual 2k file and transfers a signed-plan ack to
        # the other instances; this worker never approves itself automatically.
        return
    ack = read(root / ".first_2k_gate.json")
    required = dict(plan_signature=plan["plan_signature"], first_run_id=plan["first_run_id"],
                    checkpoint_verified=True, step=2000)
    if any(ack.get(key) != value for key, value in required.items()):
        raise RuntimeError("First 2k checkpoint acknowledgement is absent or for another plan")


def preflight(root, plan, node):
    expected_entry = root / "code" / "tools" / "micropatch_singleview_261105" / "worker.py"
    if Path(__file__).resolve() != expected_entry.resolve():
        raise RuntimeError("Run the committed code/tools/micropatch_singleview_261105/worker.py, not a staged copy")
    ready = read(root / ".environment_ready.json")
    if ready.get("passed") is not True or ready.get("decoder_verified") is not True:
        raise RuntimeError("Environment and JPEG decoder verification have not passed")
    if ready.get("frozen_commit") != plan["frozen_commit"]:
        raise RuntimeError("Environment readiness belongs to a different source commit")
    if (ready.get("plan_signature") != plan["plan_signature"]
            or ready.get("node_id") != node["node_id"]
            or ready.get("instance_id") != node["instance_id"]):
        raise RuntimeError("Environment readiness belongs to a different plan or scene owner")
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
    hardware = gpu_inventory()
    validate_identity(read(root / ".deployment_identity.json"), node, hardware, os.environ)
    return hardware


def validate_identity(identity, node, hardware, environment):
    required_identity = dict(node_id=node["node_id"], instance_id=node["instance_id"],
                             gpu_uuid=hardware["uuid"])
    if any(identity.get(key) != value for key, value in required_identity.items()):
        raise RuntimeError("Deployment node/instance/GPU identity does not match the frozen owner")
    # Vast may expose one of these identifiers. Do not require an undocumented
    # variable, but reject a numeric mismatch when an instance ID is supplied.
    for key in ("CONTAINER_ID", "VAST_CONTAINER_ID"):
        actual_id = environment.get(key)
        if actual_id and actual_id.isdigit() and int(actual_id) != node["instance_id"]:
            raise RuntimeError(f"Vast {key} differs from the registered instance")


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
    if (completed.get("passed") is not True or completed.get("steps") != 30000
            or completed.get("N") != run["expected_G_by_checkpoint"]["30000"]
            or completed.get("from_scratch") is not True
            or completed.get("final_optimizer_included") is not True
            or completed.get("phase") != "training_complete"
            or completed.get("evaluation_seconds") != 0
            or completed.get("evaluation_steps") != run["evaluation_steps"]):
        raise RuntimeError("Training completion, optimizer or train-before-evaluation contract differs")
    protocol = read(out / "protocol.json")
    expected = dict(method=run["method"], train_views=run["train_views"],
        downsample=run["downsample"], growth="native_5percent", growth_end=None,
        structural_loss=run["structural_loss"], execution_protocol="micropatch_singleview",
        checkpoint_steps=run["checkpoint_steps"], evaluation_phase="after_training_external_eval_full",
        sampler="micropatch", sampling_macro_tile=[6,6], observed_patch=[2,2],
        anchor_offsets=[0,2,4], anchor_probability=1/9, exact_ray_count="ceil(H/3)*ceil(W/3)",
        edge_policy="legacy_3x3_points_in_partial_6x6_tiles", observed_only=False,
        observed_RGB_only=True, full_model_regularization=True, multiview_start_step=0,
        native_5percent_growth_preserved=True, forced_cap_arrival=False)
    if any(protocol.get(k) != v for k,v in expected.items()):
        raise RuntimeError("Actual micropatch protocol differs from the frozen run")
    add = read(out / "resolved_config.json")["strategy"]["add"]
    expected_add = dict(start_iteration=500, end_iteration=25000, frequency=100,
                        placement="opacity", max_n_gaussians=run["method"]["cap"])
    if any(add.get(k) != v for k,v in expected_add.items()):
        raise RuntimeError("Native 5-percent growth settings differ")
    if read(out / "resolved_config.json").get("render") != run["renderer_settings"]:
        raise RuntimeError("Original [261103] renderer settings changed")
    growth = read(out / "growth_plan.json")
    expected_growth = dict(policy="native_5percent", audit_only=True, initial=run["initial_N"],
                          target=run["method"]["cap"], start=500, end=25000, frequency=100)
    if any(growth.get(k) != v for k,v in expected_growth.items()):
        raise RuntimeError("Native count audit differs from the untouched 5-percent recipe")
    native = protocol.get("native_add_schedule", {})
    if (native.get("original_add_method_preserved") is not True or
            any(native.get(k) != v for k,v in dict(start_exclusive=500,end_exclusive=25000,
                first_event=600,last_event=24900,frequency=100).items())):
        raise RuntimeError("Native MCMC add operation or schedule was replaced")
    patch = run["structural_loss"] == "patch"
    if patch:
        expected_patch = dict(support="four_actual_observed_contiguous_pixels", weights=[.25]*4,
            moments="population", C1=.0001, C2=.0009, scale="1-SSIM",
            reduction="mean_over_complete_patches_and_RGB_channels", boundary="fallback_rays_L1_only")
        if (protocol.get("patch_loss") != expected_patch or
                protocol.get("coordinate_sigma_config_unused") != 2. or
                protocol.get("ssim_backend") != "torch_observed_patch_population_moments"):
            raise RuntimeError("Actual 2x2 patch-only SSIM semantics differ")
    elif (protocol.get("coordinate_physical_support") != 3*run["method"]["sigma"] or
            protocol.get("coordinate_support_shape") != "axis_aligned_square" or
            protocol.get("ssim_backend") != "coordinate_ssim_all"):
        raise RuntimeError("Coordinate SSIM physical support differs")
    if read(out / "initial.json")["N"] != run["initial_N"]:
        raise RuntimeError("Initial SfM point count differs")
    geometry = read(out / "ray_geometry.json")
    if len(geometry) != run["train_views"]:
        raise RuntimeError("Actual training-view coverage differs")
    for g in geometry.values():
        shape = str(g.get("width"))+"x"+str(g.get("height"))
        if shape not in run["shapes"]:
            raise RuntimeError("Actual input resolution differs")
        expected_geometry = dict(sampler="micropatch", macro_tile=[6,6], patch_shape=[2,2],
            total_rays_if_equal_views=run["rays_by_shape"][shape],
            patch_count=run["complete_2x2_patches_by_shape"][shape],
            patch_rays=4*run["complete_2x2_patches_by_shape"][shape],
            boundary_fallback_rays=run["edge_fallback_rays_by_shape"][shape],
            edge_policy="legacy_3x3_points_in_partial_6x6_tiles")
        if any(g.get(k) != v for k,v in expected_geometry.items()):
            raise RuntimeError("Actual micropatch ray/patch/edge geometry differs")
        if patch and (g.get("structural_boundary_policy") != "exclude_all_fallback_rays"
                      or g.get("sigma_pixels") is not None):
            raise RuntimeError("Patch SSIM must exclude incomplete boundary groups")
        if not patch and g.get("physical_support_radius_pixels") != 3*run["method"]["sigma"]:
            raise RuntimeError("Coordinate geometry uses the wrong physical bandwidth")
    for step in run["checkpoint_steps"]:
        checkpoint = out / f"ckpt_{step}.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError(f"Missing required checkpoint: {step}")
        proof = read(out / f"checkpoint_verified_{step}.json")
        if (proof.get("step") != step or proof.get("load_verified") is not True
                or proof.get("optimizer_present") is not True
                or proof.get("optimizer_moment_shapes_verified") is not True
                or proof.get("bytes") != checkpoint.stat().st_size):
            raise RuntimeError(f"Checkpoint did not pass reload/optimizer verification: {step}")
        if proof.get("G") != run["expected_G_by_checkpoint"][str(step)]:
            raise RuntimeError(f"Checkpoint Gaussian count differs from native growth audit: {step}")
    verify_timing(out, run, completed)
    return completed


def verify_timing(out, run, completed):
    with (out / "step_times.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["step","mean_step_seconds_100","G"]:
            raise RuntimeError("Timing CSV schema differs")
        rows = list(reader)
    if [int(r["step"]) for r in rows] != list(range(100,30001,100)):
        raise RuntimeError("Missing, duplicated or reordered 100-step timing rows")
    seconds = 0.
    for row in rows:
        step, duration, count = int(row["step"]), float(row["mean_step_seconds_100"]), int(row["G"])
        if (not math.isfinite(duration) or duration < 0 or
                count != native_count(run["initial_N"],run["method"]["cap"],step)):
            raise RuntimeError("Invalid 100-step timing/count record")
        seconds += 100*duration
    if not math.isclose(seconds,completed.get("training_seconds",-1),rel_tol=1e-9,abs_tol=1e-6):
        raise RuntimeError("Training duration differs from the recorded full-step timing sum")


def verify_evaluation(out, run, step):
    value = read(out / f"evaluation_{step}.json")
    if value.get("step") != step or value.get("views") != run["test_views"]:
        raise RuntimeError(f"Evaluation step/view coverage differs: {step}")
    if value.get("run_id") != run["run_id"]:
        raise RuntimeError("Evaluation belongs to another run")
    if value.get("checkpoint_sha256") != sha(out / f"ckpt_{step}.pt"):
        raise RuntimeError("Evaluated checkpoint hash mismatch")
    if not value.get("optimizer_state_present"):
        raise RuntimeError("Evaluated checkpoint has no optimizer state")
    if value.get("N") != run["expected_G_by_checkpoint"][str(step)]:
        raise RuntimeError("Evaluated Gaussian count differs from the frozen growth schedule")
    if not all(math.isfinite(value[key]) for key in ("PSNR", "SSIM", "LPIPS")):
        raise RuntimeError("Non-finite evaluation metrics")
    return value


def receipt(root, plan, run, hardware, completed):
    out = root / "runs" / run["run_id"]
    values = {step: verify_evaluation(out, run, step) for step in run["evaluation_steps"]}
    artifacts = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "checkpoint_latest.pt" and path.suffix in (".json", ".jsonl", ".csv", ".pt"):
            artifacts.append(dict(path=str(path.relative_to(root)), bytes=path.stat().st_size, sha256=sha(path)))
    return dict(passed=True, run_id=run["run_id"], run_signature=run_signature(plan, run),
                plan_signature=plan["plan_signature"], frozen_commit=plan["frozen_commit"],
                hardware=hardware, training=completed,
                final_step=run["method"]["steps"],
                final_metrics={k: values[run["method"]["steps"]][k] for k in ("PSNR", "SSIM", "LPIPS")},
                files=artifacts, complete_utc=now())


def required_disk_bytes(run):
    """All retained optimizer checkpoints, 20% headroom, two transient files, 4GiB reserve.

    712 bytes/G is measured in the N5 final checkpoints. Counts at early
    checkpoints follow the actual native integer growth, rather than assuming
    that all seven/nine files contain the final cap from the start.
    """
    sizes = [math.ceil(1.2 * 712 * run["expected_G_by_checkpoint"][str(step)])
             for step in run["checkpoint_steps"]]
    return sum(sizes) + 2*max(sizes) + 4*2**30


def required_queue_disk_bytes(runs):
    """Reserve all remaining retained checkpoints plus only two transient files."""
    sizes = [math.ceil(1.2*712*run["expected_G_by_checkpoint"][str(step)])
             for run in runs for step in run["checkpoint_steps"]]
    return sum(sizes)+2*max(sizes)+4*2**30 if sizes else 0


def verify_artifacts(root, existing):
    for file in existing["files"]:
        path = root / file["path"]
        if not path.is_file() or sha(path) != file["sha256"]:
            raise RuntimeError(f"Archived run artifacts changed: {path}")


def cuda_oom_record(root, plan, run, hardware, phase, log, evaluation_step=None):
    """Accept only an explicit CUDA OOM traceback, plus exact recorded training state.

    A timeout, host RAM kill, disk error or arbitrary nonzero exit must stop the
    queue. CUDA OOM is a terminal result for this arm, never a request to retry,
    shrink cap, move GPUs or alter any scientific setting.
    """
    out = root / "runs" / run["run_id"]
    with log.open("rb") as stream:
        stream.seek(max(0, log.stat().st_size-128*1024))
        tail = stream.read().decode("utf-8", errors="replace")
    oom_pattern = (r"(?:torch\.(?:cuda\.)?OutOfMemoryError|RuntimeError): CUDA out of memory"
                   r"|CUDA_ERROR_OUT_OF_MEMORY|OPTIX_ERROR_MEMORY_ALLOCATION_FAILED")
    if not re.search(oom_pattern, tail):
        raise RuntimeError("Non-CUDA-OOM execution failure; queue stopped for investigation")
    failure_path = out / "failure.json"
    training_path = out / "training_complete.json"
    if phase == "training":
        if training_path.exists() or not failure_path.is_file():
            raise RuntimeError("CUDA OOM has no unambiguous structured training failure record")
        state = read(failure_path)
        if state.get("error_kind") != "cuda_out_of_memory" or state.get("passed") is not False:
            raise RuntimeError("Training failure does not independently identify CUDA OOM")
    elif phase == "evaluation":
        state = read(training_path)
        if state.get("passed") is not True or state.get("steps") != run["method"]["steps"]:
            raise RuntimeError("Evaluation OOM lacks a successful training completion record")
        state = dict(state, step=state["steps"], G=state["N"])
    else:
        raise ValueError("Unknown execution phase")
    step, count = state.get("step"), state.get("G", state.get("N"))
    if (type(step) is not int or not 0 <= step <= run["method"]["steps"]
            or type(count) is not int or not run["initial_N"] <= count <= run["method"]["cap"]):
        raise RuntimeError("CUDA OOM step/G is absent or outside the authorized run bounds")
    paths = [log, root / "run_manifests" / f"{run['run_id']}.json"]
    paths.extend(p for p in (failure_path, training_path, out / "progress.json",
                            out / "protocol.json", out / "growth_plan.json") if p.is_file())
    return dict(passed=False, terminal=True, status="cuda_oom", phase=phase,
                run_id=run["run_id"], run_signature=run_signature(plan, run),
                plan_signature=plan["plan_signature"], frozen_commit=plan["frozen_commit"],
                step=step, attempted_step=state.get("attempted_step"), G=count,
                last_completed_G=state.get("last_completed_G"),
                evaluation_step=evaluation_step, training_seconds=state.get("training_seconds"),
                hardware=hardware, failure=state, recorded_utc=now(),
                files=[dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p)) for p in paths])


def run_queue(args):
    root = args.root.resolve()
    plan = load_plan(root / "fleet" / "plan.json")
    node = selected_node(plan, args.node)
    hardware = preflight(root, plan, node)
    write(root / "hardware.json", dict(**hardware, node_id=node["node_id"],
                                      instance_id=node["instance_id"], recorded_utc=now()))
    env = dict(os.environ, PYTHONPATH=str(root / "code" / "src"), PYTHONUNBUFFERED="1")
    remaining = [run for run in node["runs"]
                 if not (root / "runs" / run["run_id"]).exists()
                 and not (root / "receipts" / (run["run_id"]+".json")).exists()
                 and not (root / "failure_receipts" / (run["run_id"]+".json")).exists()]
    required = required_queue_disk_bytes(remaining)
    if required and shutil.disk_usage(root).free < required:
        raise RuntimeError(f"Insufficient space to retain the remaining scene queue: {required} bytes required")
    done, failed = [], []
    for run in node["runs"]:
        run_id = run["run_id"]
        out = root / "runs" / run_id
        record_path = root / "receipts" / f"{run_id}.json"
        failure_record_path = root / "failure_receipts" / f"{run_id}.json"
        if record_path.exists() and failure_record_path.exists():
            raise RuntimeError("A run cannot have both success and terminal-failure receipts")
        if record_path.exists():
            existing = read(record_path)
            if not existing.get("passed") or existing.get("run_signature") != run_signature(plan, run):
                raise RuntimeError(f"Existing run receipt conflicts with this plan: {run_id}")
            verify_artifacts(root, existing)
            done.append(run_id)
            continue
        if failure_record_path.exists():
            existing = read(failure_record_path)
            if (existing.get("status") != "cuda_oom" or existing.get("terminal") is not True
                    or existing.get("run_signature") != run_signature(plan, run)):
                raise RuntimeError("Existing failure receipt conflicts with this plan")
            verify_artifacts(root, existing)
            failed.append(run_id)
            continue
        gate(root, plan, run)
        if gpu_inventory() != hardware:
            raise RuntimeError("GPU, driver or power limit changed within the scene queue")
        write(root / "status.json", dict(phase="training", run_id=run_id, completed=done,
                                        failed=failed, node_id=args.node, updated_utc=now()))
        pending = root / "run_manifests" / f"{run_id}.json"
        if out.exists():
            if not args.resume_evaluation:
                raise RuntimeError(f"Partial run exists; manual investigation required: {out}")
            if not pending.exists() or read(pending)["run_signature"] != run_signature(plan, run):
                raise RuntimeError("Cannot resume evaluation for an unregistered run")
            if read(pending).get("hardware") != hardware:
                raise RuntimeError("Evaluation must remain on the registered training hardware")
            completed = verify_training(out, run)
        else:
            if pending.exists():
                raise RuntimeError("An earlier launch was attempted; no automatic rerun")
            required = required_disk_bytes(run)
            free = shutil.disk_usage(root).free
            if free < required:
                raise RuntimeError(f"Insufficient checkpoint space: {free} free bytes, {required} required")
            write(pending, dict(run=run, run_signature=run_signature(plan, run),
                                frozen_commit=plan["frozen_commit"], hardware=hardware,
                                node_id=args.node, instance_id=node["instance_id"], launch_utc=now(),
                                free_disk_bytes=free, required_disk_bytes=required))
            train_log = root / "logs" / f"{run_id}.train.log"
            try:
                execute(train_command(root, run, plan["python"]), train_log, root / "code", env)
            except subprocess.CalledProcessError:
                failure = cuda_oom_record(root, plan, run, hardware, "training", train_log)
                write(failure_record_path, failure)
                failed.append(run_id)
                write(root / "status.json", dict(phase="arm_cuda_oom", run_id=run_id,
                      completed=done, failed=failed, step=failure["step"], G=failure["G"],
                      node_id=args.node, updated_utc=now()))
                continue
            completed = verify_training(out, run)
        write(root / "status.json", dict(phase="evaluation", run_id=run_id, completed=done,
                                        failed=failed, node_id=args.node, updated_utc=now()))
        evaluation_failed = False
        for step in run["evaluation_steps"]:
            evaluation = out / f"evaluation_{step}.json"
            if evaluation.exists():
                if not args.resume_evaluation:
                    raise RuntimeError("Unexpected preexisting evaluation output")
                verify_evaluation(out, run, step)
                continue
            evaluation_log = root / "logs" / f"{run_id}.eval{step}.log"
            try:
                execute(eval_command(root, run, step, plan["python"]), evaluation_log, root / "code", env)
            except subprocess.CalledProcessError:
                failure = cuda_oom_record(root, plan, run, hardware, "evaluation", evaluation_log, step)
                write(failure_record_path, failure)
                failed.append(run_id)
                write(root / "status.json", dict(phase="arm_cuda_oom", run_id=run_id,
                      completed=done, failed=failed, step=failure["step"], G=failure["G"],
                      evaluation_step=step, node_id=args.node, updated_utc=now()))
                evaluation_failed = True
                break
            verify_evaluation(out, run, step)
        if evaluation_failed:
            continue
        write(record_path, receipt(root, plan, run, hardware, completed))
        done.append(run_id)
        write(root / "status.json", dict(phase="run_complete", run_id=run_id, completed=done,
                                        failed=failed, node_id=args.node, updated_utc=now()))
    write(root / "status.json", dict(phase="complete_with_failures" if failed else "complete",
                                    completed=done, failed=failed, node_id=args.node, updated_utc=now()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--check-only", action="store_true", help="Run readiness/code/GPU checks without training")
    parser.add_argument("--resume-evaluation", action="store_true",
                        help="Explicitly finish evaluation only after a reviewed evaluation failure")
    args = parser.parse_args()
    # A duplicate launcher must not overwrite the active worker's status file.
    with machine_lock(args.root.resolve()):
        try:
            if args.check_only:
                plan = load_plan(args.root / "fleet" / "plan.json")
                print(json.dumps(preflight(args.root.resolve(), plan, selected_node(plan,args.node))))
            else:
                run_queue(args)
        except BaseException as error:
            write(args.root / "status.json", dict(phase="failed", node_id=args.node, updated_utc=now(),
                                                 error=f"{type(error).__name__}: {error}"))
            raise


if __name__ == "__main__":
    main()
