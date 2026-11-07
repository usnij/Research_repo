"""Prepare an independent N5 queue; freeze only after the final source commit.

Hosts are private deployment inputs. Each node owns one scene and all eight arms.
This script writes a draft, not a launch authorization or an E1/E2 amendment.
"""
import argparse
from pathlib import Path
import shlex
import subprocess

from common import ARM_SPECS, STEPS, eval_command, read, sha, signature, train_command, write


SCENES = {"bicycle", "bonsai", "garden"}
ARMS = ARM_SPECS


def load_assignments(path):
    nodes = read(path)["nodes"]
    if len(nodes) != 3 or nodes[0].get("primary_scene") != "bicycle":
        raise ValueError("Expected three owners, with Bicycle first for the actual 2k save gate")
    if {node.get("primary_scene") for node in nodes} != SCENES:
        raise ValueError("This campaign requires Bicycle, Bonsai and Garden")
    if len({node["node_id"] for node in nodes}) != len(nodes):
        raise ValueError("Duplicate node IDs")
    if len({node["instance_id"] for node in nodes}) != len(nodes):
        raise ValueError("One GPU instance cannot own multiple parallel scene queues")
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
        if node.get("assigned_scenes") != [node["primary_scene"]]:
            raise ValueError("Each scene and all its arms must stay on a single owner")
        scenes.extend(node["assigned_scenes"])
    if len(scenes) != len(set(scenes)):
        raise ValueError("A scene cannot be assigned to more than one instance")
    return nodes


def native_count(initial, cap, completed_steps):
    count = initial
    for update_index in range(completed_steps):
        if 500 < update_index < 25000 and update_index % 100 == 0:
            count = min(cap, int(1.05 * count))
    return count


def prepare(args):
    output = args.output.resolve()
    if (output / "plan.json").exists():
        raise FileExistsError("A frozen plan already exists; do not replace it")
    nodes = []
    for assignment in load_assignments(args.hosts):
        scene = assignment["primary_scene"]
        source = args.payload.resolve() / "scenes" / scene
        base = read(source / "structure.json")
        audit = read(source / "audit.json")
        if base["cap"] != audit["target_N"] or audit["scene"] != scene:
            raise ValueError(f"Source cap/scene provenance mismatch: {scene}")
        if (base["l1_weight"], base["structure_weight"], base["opacity_weight"], base["scale_weight"]) != (.8,.2,.01,.01):
            raise ValueError("The established image-loss and regularizer weights must be preserved")
        if audit["initial_N"] <= 0 or audit["initial_N"] >= base["cap"]:
            raise ValueError("Expected a nonempty SfM initialization below the scene cap")
        runs = []
        for arm, loss, block, sigma, window in ARMS:
            config = dict(base, views=1, block=block, steps=30000, initialization_seed=777,
                          sigma=sigma, grid_window=window, grid_sigma=1.5)
            rate = "full" if block == 1 else "r9"
            run_id = f"native5_N5_{scene}_k1_{arm}_{rate}_seed777"
            config_path = f"configs/{run_id}.json"
            write(output / config_path, config)
            shape_rays = {}
            for shape in audit["shapes"]:
                width, height = map(int, shape.split("x"))
                shape_rays[shape] = ((width + block - 1)//block) * ((height + block - 1)//block)
            runs.append(dict(run_id=run_id, experiment_id="N5", scene=scene, arm=arm,
                             structural_loss=loss, method=config, config_path=config_path,
                             config_sha256=sha(output/config_path), downsample=audit["downsample"],
                             train_views=audit["train_views"], test_views=audit["test_views"],
                             initial_N=audit["initial_N"], shapes=audit["shapes"],
                             rays_by_shape=shape_rays, nominal_ray_fraction=1/block**2,
                             expected_G_by_checkpoint={str(step):native_count(audit["initial_N"],base["cap"],step) for step in STEPS},
                             cap_source=dict(rule="fine_tracks_v1", views="all", source=str(source),
                                             structure_sha256=sha(source/"structure.json"),
                                             audit_sha256=sha(source/"audit.json"))))
        nodes.append(dict(node_id=assignment["node_id"], instance_id=assignment["instance_id"],
                          direct=assignment["direct"], proxy=assignment.get("proxy"),
                          primary_scene=scene, assigned_scenes=[scene], runs=runs))
    plan = dict(schema_version=1, campaign="native5_singleview_261105", remote_root=args.remote_root,
                python=args.python, renderer="261103_exact_preserved_by_user_2026-11-04",
                allow_experiments=["N5"], nodes=nodes, steps=list(STEPS),
                first_run_id=nodes[0]["runs"][0]["run_id"], first_node_id=nodes[0]["node_id"],
                first_2k_gate_required=True, no_automatic_retry=True,
                prior_campaign_records_untouched=True, automatic_checkpoint_deletion=False,
                growth=dict(policy="native_5percent", start_exclusive=500, first_update_index=600,
                            interval=100, end_exclusive=25000, nominal_rate=.05,
                            count_rule="min(cap, int(1.05 * current_N))", five_k_cutoff=False),
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
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=code, text=True).strip()
    if dirty:
        raise ValueError("Commit the complete implementation before freezing")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=code).decode().split("\0")
    paths = [Path(name) for name in names if name and (code/name).is_file()]
    required = {"eval_full.py", "src/sparse_grt/training.py", "src/sparse_grt/config.py",
                "tools/native5_261105/worker.py", "tools/native5_261105/common.py"}
    if not required.issubset(set(map(str, paths))):
        raise ValueError("Evaluator, training code and N5 launcher must all be committed")
    plan["code_files"] = [dict(path=str(path), sha256=sha(code/path)) for path in sorted(paths)]
    plan["frozen_commit"] = commit
    plan["plan_signature"] = signature(plan)
    write(output / "plan.json", plan)
    lines = [f"# N5 frozen source commit: {commit}", f"# Plan signature: {plan['plan_signature']}"]
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
    create.add_argument("--hosts", type=Path, required=True)
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
