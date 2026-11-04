"""Prepare E1/E2 queues; freezing requires the actual final source commit.

Private deployment information must be supplied through --hosts, never added
to this source file. Its JSON schema is {"nodes": [{"node_id": str,
"primary_scene": str, "instance_id": int, "direct": {"host": str, "port": int},
"proxy": {"host": str, "port": int} | null, "assigned_scenes": [str]}]}.
The first node owns Counter; the other first scenes are Bonsai/Bicycle/Garden.
"""
import argparse
from pathlib import Path
import shlex
import subprocess

from common import STEPS, eval_command, read, sha, signature, train_command, write


def load_assignments(path):
    nodes = read(path)["nodes"]
    if len(nodes) != 4 or nodes[0].get("primary_scene") != "counter":
        raise ValueError("Expected four scene owners, with Counter first for the 2k gate")
    if {node.get("primary_scene") for node in nodes} != {"counter", "bonsai", "bicycle", "garden"}:
        raise ValueError("The first E1/E2 stage requires Counter, Bonsai, Bicycle and Garden")
    if len({node["node_id"] for node in nodes}) != len(nodes):
        raise ValueError("Duplicate node IDs")
    if len({node["instance_id"] for node in nodes}) != len(nodes):
        raise ValueError("Multiple scene queues cannot target the same instance")
    scenes = []
    for node in nodes:
        if not isinstance(node["instance_id"], int) or node["instance_id"] <= 0:
            raise ValueError("Each instance_id must be a positive integer")
        if not isinstance(node["node_id"], str) or not node["node_id"]:
            raise ValueError("Each node needs a nonempty identifier")
        for key in ("direct", "proxy"):
            endpoint = node.get(key)
            if endpoint is None and key == "proxy":
                continue
            if not isinstance(endpoint, dict) or not isinstance(endpoint.get("host"), str) or not endpoint["host"]:
                raise ValueError(f"Invalid {key} hostname")
            if not isinstance(endpoint.get("port"), int) or not 1 <= endpoint["port"] <= 65535:
                raise ValueError(f"Invalid {key} SSH port")
        if node["primary_scene"] not in node["assigned_scenes"]:
            raise ValueError("Primary scene must belong to the instance's assigned scenes")
        scenes.extend(node["assigned_scenes"])
    if len(scenes) != len(set(scenes)):
        raise ValueError("A scene cannot be assigned to more than one instance")
    return nodes


def prepare(args):
    output = args.output.resolve()
    plan_path = output / "plan.json"
    if plan_path.exists():
        raise FileExistsError("A frozen plan already exists; do not replace it")
    nodes = []
    for assignment in load_assignments(args.hosts):
        node_id = assignment["node_id"]
        scene = assignment["primary_scene"]
        source = args.payload / "scenes" / scene
        base = read(source / "structure.json")
        audit = read(source / "audit.json")
        if base["cap"] != audit["target_N"]:
            raise ValueError(f"Cap provenance mismatch for {scene}")
        runs = []
        arms = [("E1", "coordinate"), ("E1", "grid"), ("E2", "dense")]
        if scene == "garden":
            arms = [("E2", "dense")]
        for experiment, loss in arms:
            block = 1 if loss == "dense" else (3 if scene == "bicycle" else 4)
            config = dict(base, views=1, block=block, steps=30000, initialization_seed=777)
            if config["l1_weight"] != .8 or config["structure_weight"] != .2:
                raise ValueError("Unexpected source image-loss weights")
            if config["opacity_weight"] != .01 or config["scale_weight"] != .01:
                raise ValueError("Unexpected source regularizer weights")
            rate = "dense" if block == 1 else f"r{block**2}"
            run_id = f"cvpr2027_{experiment}_{scene}_k1_{loss}_{rate}_seed777"
            config_path = f"configs/{run_id}.json"
            write(output / config_path, config)
            shape_rays = {}
            for shape in audit["shapes"]:
                width, height = map(int, shape.split("x"))
                shape_rays[shape] = ((width + block - 1)//block) * ((height + block - 1)//block)
            runs.append(dict(run_id=run_id, experiment_id=experiment, scene=scene,
                             arm=f"K1_{loss}", structural_loss=loss, method=config,
                             config_path=config_path, config_sha256=sha(output/config_path),
                             downsample=audit["downsample"], train_views=audit["train_views"],
                             test_views=audit["test_views"], initial_N=audit["initial_N"],
                             shapes=audit["shapes"], rays_by_shape=shape_rays,
                             nominal_ray_fraction=1/block**2,
                             cap_source=dict(rule="fine_tracks_v1", views="all",
                                             structure_sha256=sha(source/"structure.json"),
                                             audit_sha256=sha(source/"audit.json"))))
        nodes.append(dict(node_id=node_id, instance_id=assignment["instance_id"],
                          direct=assignment["direct"], proxy=assignment.get("proxy"),
                          assigned_scenes=assignment["assigned_scenes"], runs=runs))
    plan = dict(schema_version=1, campaign="cvpr2027_261104",
                remote_root=args.remote_root, python=args.python,
                renderer="261103_exact_preserved_by_user_2026-11-04",
                allow_experiments=["E1", "E2"], nodes=nodes,
                steps=list(STEPS), first_run_id=nodes[0]["runs"][0]["run_id"],
                first_2k_gate_required=True, no_automatic_retry=True,
                forbidden_before_new_signal=["E4", "E5", "E7", "E8", "X1"],
                growth=dict(policy="log_horizon", first=600, interval=100, end=5000),
                frozen_commit=None, code_files=[])
    write(output / "plan.draft.json", plan)
    return plan


def freeze(args):
    output = args.output.resolve()
    if (output / "plan.json").exists():
        raise FileExistsError("Refusing to overwrite frozen plan")
    plan = read(output / "plan.draft.json")
    code = args.code.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code, text=True).strip()
    if commit != args.commit:
        raise ValueError("--commit must equal the actual source HEAD")
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=code, text=True).strip()
    if dirty:
        raise ValueError("Commit the protocol implementation before freezing")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=code).decode().split("\0")
    paths = [Path(name) for name in names if name and (code/name).is_file()]
    required = {"eval_full.py", "src/sparse_grt/training.py", "src/sparse_grt/config.py"}
    if not required.issubset(set(map(str, paths))):
        raise ValueError("Evaluator and training code must be committed")
    plan["code_files"] = [dict(path=str(path), sha256=sha(code/path)) for path in sorted(paths)]
    plan["frozen_commit"] = commit
    plan["plan_signature"] = signature(plan)
    write(output / "plan.json", plan)
    lines = [f"# Frozen source commit: {commit}", f"# Plan signature: {plan['plan_signature']}"]
    for node in plan["nodes"]:
        lines.append(f"\n# {node['node_id']}, instance {node['instance_id']}")
        for run in node["runs"]:
            lines.extend([f"# {run['run_id']}", shlex.join(train_command(plan["remote_root"], run, plan["python"]))])
            lines.extend(shlex.join(eval_command(plan["remote_root"], run, step, plan["python"])) for step in STEPS)
    (output / "commands.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--payload", type=Path, required=True)
    create.add_argument("--hosts", type=Path, required=True,
                        help="Private deployment JSON; see this script's module docstring for its schema")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--remote-root", required=True)
    create.add_argument("--python", required=True)
    seal = sub.add_parser("freeze")
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--code", type=Path, required=True)
    seal.add_argument("--commit", required=True)
    args = parser.parse_args()
    plan = prepare(args) if args.action == "prepare" else freeze(args)
    print(f"{args.action}: {len(plan['nodes'])} nodes, {sum(len(n['runs']) for n in plan['nodes'])} runs")


if __name__ == "__main__":
    main()
