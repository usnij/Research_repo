"""Export machine-produced campaign records, without rounding or invented values.

This is a runtime data-pipeline serializer, not a spreadsheet/workbook author.
The export contains only this campaign. It does not merge prior E1/E2 or historical benchmark records.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import shlex

from common import STEPS, eval_command, load_plan, read, run_signature, train_command, write


RUN_COLUMNS = ["run_id", "experiment_id", "scene", "arm", "densification", "loss", "K",
               "block_size", "R_total", "resolution", "renderer_settings", "cap", "final_G",
               "seed", "observed_only", "MV_start_step", "instance", "driver", "power_limit",
               "commit", "training_seconds", "PSNR", "SSIM", "LPIPS", "status", "DS",
               "nominal_ray_fraction", "plan_signature", "sigma_pixels", "grid_window", "grid_sigma"]
CURVE_COLUMNS = ["run_id", "step", "training_seconds", "G", "PSNR", "SSIM", "LPIPS"]
VIEW_COLUMNS = ["run_id", "view", "PSNR", "SSIM", "LPIPS", "image"]
LOG_COLUMNS = ["step", "mean_step_seconds_100", "G"]


def maybe(path):
    return read(path) if path.is_file() else {}


def csv_file(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: ("NA" if row.get(column) is None else row[column])
                             for column in columns})
    pending.replace(path)


def metrics(value):
    result = {key: value.get(key) for key in ("PSNR", "SSIM", "LPIPS")}
    if value and not all(isinstance(v, (float, int)) and math.isfinite(v) for v in result.values()):
        raise ValueError("Non-finite or missing metric in an evaluation output")
    return result


def checked_evaluation(path, run, step):
    result = maybe(path)
    if not result:
        return {}
    if (result.get("run_id") != run["run_id"] or result.get("step") != step
            or result.get("views") != run["test_views"]):
        raise ValueError(f"Mismatched evaluation: {path}")
    rows = result.get("per_view", [])
    if len(rows) != run["test_views"] or len({r["view"] for r in rows}) != len(rows):
        raise ValueError(f"Missing/duplicated test views: {path}")
    for row in rows:
        metrics(row)
    aggregate = metrics(result)
    for key, value in aggregate.items():
        if not math.isclose(value, sum(row[key] for row in rows)/len(rows), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Aggregate/per-view mismatch: {path}, {key}")
    return result


def export(plan, roots, output):
    output.mkdir(parents=True, exist_ok=True)
    runs, curves, views = [], [], []
    commands = []
    for node in plan["nodes"]:
        root = roots.get(node["node_id"])
        hardware = maybe(root / "hardware.json") if root else {}
        for run in node["runs"]:
            method = run["method"]
            out = root / "runs" / run["run_id"] if root else None
            evidence = maybe(root / "run_manifests" / f"{run['run_id']}.json") if root else {}
            if evidence and evidence.get("run_signature") != run_signature(plan, run):
                raise ValueError("Collected run manifest belongs to another frozen plan")
            completed = maybe(root / "receipts" / f"{run['run_id']}.json") if root else {}
            if completed and completed.get("run_signature") != run_signature(plan, run):
                raise ValueError("Collected completion receipt belongs to another frozen plan")
            config = maybe(out / "resolved_config.json") if out else {}
            protocol = maybe(out / "protocol.json") if out else {}
            progress = maybe(out / "progress.json") if out else {}
            training = maybe(out / "training_complete.json") if out else {}
            final = checked_evaluation(out / "evaluation_30000.json", run, 30000) if out else {}
            status = "planned"
            if out and out.is_dir():
                status = "started"
            if progress:
                status = f"training_{progress['step']}"
            if training:
                status = "training_complete_evaluation_pending"
            if final:
                status = "final_evaluated_intermediate_or_receipt_pending"
            if completed:
                status = "complete"
            geometry = maybe(out / "ray_geometry.json") if out else {}
            actual_R = sorted({item["total_rays_if_equal_views"] for item in geometry.values()})
            if not actual_R:
                # Planned count is explicitly distinguished by status='planned'.
                actual_R = sorted(set(run["rays_by_shape"].values()))
            renderer = config.get("render") if config else None
            if renderer is None:
                renderer_value = plan["renderer"] + ":planned"
            else:
                renderer_value = json.dumps(renderer, sort_keys=True, separators=(",", ":"))
            row = dict(run_id=run["run_id"], experiment_id=run["experiment_id"], scene=run["scene"],
                       arm=run["arm"], densification="MCMC/native_5percent_every_100_after_500_before_25000_cap",
                       loss=f"0.8*L1+0.2*{run['structural_loss']}_SSIM;opacity=0.01;scale=0.01",
                       K=method["views"], block_size=method["block"],
                       R_total=actual_R[0] if len(actual_R) == 1 else json.dumps(actual_R),
                       resolution=";".join(sorted(run["shapes"])), renderer_settings=renderer_value,
                       cap=method["cap"], final_G=final.get("N", training.get("N")),
                       seed=method["initialization_seed"], observed_only=protocol.get("observed_only"),
                       MV_start_step=protocol.get("multiview_start_step"), instance=node["instance_id"],
                       driver=hardware.get("driver"), power_limit=hardware.get("power_limit_w"),
                       commit=plan["frozen_commit"], training_seconds=training.get("training_seconds"),
                       **metrics(final), status=status, DS=run["downsample"],
                       nominal_ray_fraction=run["nominal_ray_fraction"], plan_signature=plan["plan_signature"],
                       sigma_pixels=method["sigma"] if run["structural_loss"] == "coordinate" else None,
                       grid_window=method.get("grid_window") if run["structural_loss"] == "grid" else None,
                       grid_sigma=method.get("grid_sigma") if run["structural_loss"] == "grid" else None)
            runs.append(row)
            for step in STEPS:
                proof = maybe(out / f"checkpoint_verified_{step}.json") if out else {}
                value = checked_evaluation(out / f"evaluation_{step}.json", run, step) if out else {}
                if not proof and not value:
                    continue
                if proof and (proof.get("step") != step or proof.get("load_verified") is not True):
                    raise ValueError("Invalid checkpoint verification receipt")
                curves.append(dict(run_id=run["run_id"], step=step,
                                   training_seconds=proof.get("training_seconds"),
                                   G=proof.get("G", value.get("N")), **metrics(value)))
            if final:
                views.extend(dict(run_id=run["run_id"], view=view["view"], image=view["image"],
                                  **metrics(view)) for view in final["per_view"])
            if out and (out / "step_times.csv").is_file():
                with (out / "step_times.csv").open(newline="", encoding="utf-8") as handle:
                    log = list(csv.DictReader(handle))
                good = []
                for item in log:
                    if any(item.get(column) in (None, "") for column in LOG_COLUMNS):
                        # A concurrently copied last row may still be incomplete.
                        if item is log[-1]:
                            break
                        raise ValueError("Incomplete nonfinal 100-step timing row")
                    good.append({column: item[column] for column in LOG_COLUMNS})
                if len({row["step"] for row in good}) != len(good):
                    raise ValueError("Duplicated timing step")
                csv_file(output / "logs" / f"{run['run_id']}.csv", LOG_COLUMNS, good)
            commands.extend([f"# {run['run_id']}, {node['node_id']}, instance {node['instance_id']}",
                             shlex.join(train_command(plan["remote_root"], run, plan["python"]))])
            commands.extend(shlex.join(eval_command(plan["remote_root"], run, step, plan["python"])) for step in STEPS)
    csv_file(output / "runs.csv", RUN_COLUMNS, runs)
    csv_file(output / "curves.csv", CURVE_COLUMNS, curves)
    csv_file(output / "views.csv", VIEW_COLUMNS, views)
    (output / "commands.txt").write_text("\n".join(commands)+"\n", encoding="utf-8")
    write(output / "export_scope.json", dict(campaign=plan["campaign"], plan_signature=plan["plan_signature"],
                                             missing_value="NA", precision="unrounded serialized evaluator floats",
                                             historical_reused_runs_included=False,
                                             note="This export covers only the 24 independent N5 runs. Never overwrite or relabel E1/E2 records."))
    print(json.dumps(dict(runs=len(runs), curves=len(curves), views=len(views), output=str(output))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--node-root", action="append", default=[], metavar="NODE=/LOCAL/COLLECTED/ROOT")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {}
    for item in args.node_root:
        node, path = item.split("=", 1)
        if node in roots:
            raise ValueError(f"Duplicated node: {node}")
        roots[node] = Path(path).resolve()
    export(load_plan(args.plan), roots, args.output.resolve())


if __name__ == "__main__":
    main()
