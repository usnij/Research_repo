"""CPU-only guards for the exact eight-run capacity/duration queue."""
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import subprocess

from common import (CAPS, LONG_STEPS, NODE_SPECS, STEPS, load_plan, native_count,
                    read, run_signature, sha, signature, train_command, write)
from worker import (cuda_oom_record, gate, machine_lock, required_disk_bytes,
                    run_queue, validate_identity, verify_training)
from export_results import export


def fixture_plan():
    nodes = []
    for index, (node_id, scene, arms) in enumerate(NODE_SPECS):
        runs = []
        for arm in arms:
            long = arm.startswith("long_")
            dense = arm.endswith("dense")
            scale = 2 if long else 1
            method = dict(views=1, block=1 if dense else 3,
                          sigma=6. if dense else (2. if arm.endswith("sigma2") else 4.),
                          grid_window=11, grid_sigma=1.5, steps=30000*scale,
                          initialization_seed=777, cap=CAPS[scene][0 if long else 1],
                          l1_weight=.8, structure_weight=.2, opacity_weight=.01,
                          scale_weight=.01, boundary="all")
            initial = 54275 if scene == "bicycle" else 138766
            checkpoints = list(LONG_STEPS if long else STEPS)
            run = dict(run_id=f"{scene}_{arm}", scene=scene, arm=arm,
                       experiment_id="N5_60K" if long else "N5_CAP", method=method,
                       schedule_scale=scale, structural_loss="dense" if dense else "coordinate",
                       checkpoint_steps=checkpoints,
                       evaluation_steps=[30000,45000,60000] if long else list(STEPS),
                       initial_N=initial, train_views=2, test_views=2,
                       shapes={"103x83":2}, rays_by_shape={"103x83":8549 if dense else 980},
                       downsample=4, config_path=f"configs/{scene}_{arm}.json",
                       nominal_ray_fraction=1. if dense else 1./9.)
            run["expected_G_by_checkpoint"] = {str(s): native_count(initial, method["cap"], s, scale)
                                               for s in checkpoints}
            runs.append(run)
        nodes.append(dict(node_id=node_id, primary_scene=scene, instance_id=index+1, runs=runs))
    plan = dict(campaign="native5_capacity_duration_261105", allow_experiments=["N5_CAP","N5_60K"],
                frozen_commit="fixture", code_files=[dict(path="fixture")], nodes=nodes,
                first_run_id=nodes[0]["runs"][0]["run_id"], python="/env/python",
                remote_root="/workspace/capacity", renderer="261103")
    return dict(plan, plan_signature=signature(plan))


def training_fixture(out, run):
    out.mkdir(parents=True, exist_ok=True)
    scale = run["schedule_scale"]
    cap = run["method"]["cap"]
    write(out/"training_complete.json", dict(passed=True, steps=run["method"]["steps"], N=cap))
    write(out/"protocol.json", dict(method=run["method"], train_views=2,
          growth="native_5percent", growth_end=None, structural_loss=run["structural_loss"],
          execution_protocol="native5_capacity_duration", schedule_scale=scale,
          checkpoint_steps=run["checkpoint_steps"], evaluation_steps=run["evaluation_steps"],
          native_add_schedule=dict(original_add_method_preserved=True),
          observed_only=False, multiview_start_step=0))
    write(out/"resolved_config.json", dict(strategy=dict(add=dict(start_iteration=500,
          end_iteration=25000*scale, frequency=100, placement="opacity", max_n_gaussians=cap))))
    write(out/"growth_plan.json", dict(policy="native_5percent", audit_only=True,
          initial=run["initial_N"], target=cap, start=500, end=25000*scale, frequency=100))
    write(out/"initial.json", dict(N=run["initial_N"]))
    write(out/"ray_geometry.json", {str(v):dict(width=103, height=83,
          total_rays_if_equal_views=run["rays_by_shape"]["103x83"]) for v in range(2)})
    for step in run["checkpoint_steps"]:
        (out/f"ckpt_{step}.pt").write_bytes(b"CPU fixture")
        write(out/f"checkpoint_verified_{step}.json", dict(step=step,
              G=run["expected_G_by_checkpoint"][str(step)], load_verified=True, optimizer_present=True))


class CapacityDurationGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)/"campaign"
        self.root.mkdir()
        self.plan = fixture_plan()

    def tearDown(self):
        self.temp.cleanup()

    def load(self, plan):
        body = {k:v for k,v in plan.items() if k != "plan_signature"}
        write(self.root/"plan.json", dict(body, plan_signature=signature(body)))
        return load_plan(self.root/"plan.json")

    def test_exact_eight_arms_four_distinct_instances_and_horizons(self):
        actual = self.load(self.plan)
        self.assertEqual(sum(len(n["runs"]) for n in actual["nodes"]),8)
        for node in actual["nodes"]:
            for run in node["runs"]:
                command = train_command(self.root, run, "python")
                self.assertEqual(command[command.index("--protocol")+1], "native5_capacity_duration")
                self.assertEqual(command[command.index("--keep-checkpoints")+1], ",".join(map(str,run["checkpoint_steps"])))
                self.assertNotIn("--growth-end",command)
                self.assertEqual(run["expected_G_by_checkpoint"][str(run["method"]["steps"])],run["method"]["cap"])
        for mutation in ("duplicate_id","same_instance","wrong_cap","wrong_horizon","wrong_steps","extra_arm"):
            plan = copy.deepcopy(self.plan)
            run = plan["nodes"][1]["runs"][-1]
            if mutation == "duplicate_id":run["run_id"]=plan["first_run_id"]
            if mutation == "same_instance":plan["nodes"][1]["instance_id"]=1
            if mutation == "wrong_cap":run["method"]["cap"]+=1
            if mutation == "wrong_horizon":run["schedule_scale"]=1
            if mutation == "wrong_steps":run["evaluation_steps"]=[30000]
            if mutation == "extra_arm":plan["nodes"][0]["runs"].append(copy.deepcopy(run))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.load(plan)

    def test_growth_native_float_rounding_and_zero_based_boundary(self):
        self.assertEqual(native_count(100,1000,600),100)
        self.assertEqual(native_count(100,1000,601),105)
        self.assertEqual(native_count(100,1000,600,2),100)
        self.assertEqual(native_count(100,1000,601,2),105)
        self.assertEqual(native_count(100,1000,60000,2),1000)

    def test_60k_requires_nine_optimizer_checkpoints_and_scaled_growth(self):
        run=self.plan["nodes"][1]["runs"][-1];out=self.root/"run"
        training_fixture(out,run)
        self.assertEqual(verify_training(out,run)["steps"],60000)
        bad=read(out/"growth_plan.json");bad["end"]=25000;write(out/"growth_plan.json",bad)
        with self.assertRaisesRegex(RuntimeError,"count audit"):verify_training(out,run)
        training_fixture(out,run);(out/"ckpt_45000.pt").unlink()
        with self.assertRaisesRegex(RuntimeError,"45000"):verify_training(out,run)

    def test_disk_estimate_reflects_early_counts_and_all_saved_checkpoints(self):
        run=self.plan["nodes"][0]["runs"][0]
        actual=required_disk_bytes(run)
        self.assertLess(actual, 12*2**30+7000*run["method"]["cap"])
        self.assertGreater(actual, sum(run["expected_G_by_checkpoint"].values())*712)
        self.assertGreater(required_disk_bytes(self.plan["nodes"][1]["runs"][-1]), 4*2**30)

    def test_first_2k_gate_and_cross_root_gpu_lock(self):
        gate(self.root,self.plan,self.plan["nodes"][0]["runs"][0])
        run=self.plan["nodes"][1]["runs"][0]
        with self.assertRaises(FileNotFoundError):gate(self.root,self.plan,run)
        write(self.root/".first_2k_gate.json",dict(plan_signature=self.plan["plan_signature"],
              first_run_id=self.plan["first_run_id"],checkpoint_verified=True,step=2000))
        gate(self.root,self.plan,run)
        other=self.root.parent/"other";other.mkdir()
        with machine_lock(self.root),self.assertRaises(RuntimeError):
            with machine_lock(other):pass

    def test_identity_rejects_wrong_instance_or_gpu(self):
        node=self.plan["nodes"][0];hardware=dict(uuid="GPU-A")
        identity=dict(node_id=node["node_id"],instance_id=node["instance_id"],gpu_uuid="GPU-A")
        validate_identity(identity,node,hardware,{})
        with self.assertRaises(RuntimeError):validate_identity(identity,node,hardware,{"CONTAINER_ID":"999"})
        with self.assertRaises(RuntimeError):validate_identity(dict(identity,gpu_uuid="GPU-B"),node,hardware,{})

    def oom_fixture(self,run):
        out=self.root/"runs"/run["run_id"];out.mkdir(parents=True)
        log=self.root/"logs"/f"{run['run_id']}.train.log";log.parent.mkdir(exist_ok=True)
        log.write_text("Traceback (most recent call last):\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.00 GiB\n")
        write(out/"failure.json",dict(passed=False,error_kind="cuda_out_of_memory",phase="update",step=7400,
              attempted_step=7401,N=2000000,G=2000000,last_completed_G=1900000,training_seconds=700.))
        write(self.root/"run_manifests"/f"{run['run_id']}.json",dict(run_signature=run_signature(self.plan,run)))
        return out,log

    def test_only_correlated_cuda_oom_becomes_terminal_result(self):
        run=self.plan["nodes"][1]["runs"][0];out,log=self.oom_fixture(run)
        value=cuda_oom_record(self.root,self.plan,run,{},"training",log)
        self.assertEqual((value["status"],value["step"],value["G"]),("cuda_oom",7400,2000000))
        log.write_text("RuntimeError: disk is full\n")
        with self.assertRaisesRegex(RuntimeError,"Non-CUDA-OOM"):cuda_oom_record(self.root,self.plan,run,{},"training",log)
        log.write_text("torch.OutOfMemoryError: CUDA out of memory\n")
        write(out/"failure.json",dict(passed=False,error_kind="runtime_error",step=7400,G=2000000))
        with self.assertRaisesRegex(RuntimeError,"independently"):cuda_oom_record(self.root,self.plan,run,{},"training",log)

    def test_queue_skips_registered_terminal_oom_never_retries(self):
        node=self.plan["nodes"][0];run=node["runs"][0];out,log=self.oom_fixture(run)
        failure=cuda_oom_record(self.root,self.plan,run,{},"training",log)
        write(self.root/"failure_receipts"/f"{run['run_id']}.json",failure)
        with patch("worker.load_plan",return_value=self.plan),patch("worker.preflight",return_value={}),\
             patch("worker.execute",side_effect=AssertionError("Forbidden retry")):
            run_queue(SimpleNamespace(root=self.root,node="node01",resume_evaluation=False))
        status=read(self.root/"status.json")
        self.assertEqual(status["phase"],"complete_with_failures")
        self.assertEqual(status["failed"],[run["run_id"]])

    def test_actual_failed_cuda_arms_advance_once_and_optix_is_recognized(self):
        node=self.plan["nodes"][1]
        launched=[]
        def execute(command,log,code,environment):
            run=node["runs"][len(launched)];launched.append(run["run_id"])
            self.assertIn(run["run_id"],str(command))
            out,_=self.oom_fixture(run)
            log.write_text("Traceback (most recent call last):\nRuntimeError: OPTIX_ERROR_MEMORY_ALLOCATION_FAILED\n")
            raise subprocess.CalledProcessError(1,command)
        with patch("worker.load_plan",return_value=self.plan),patch("worker.preflight",return_value={}),\
             patch("worker.gate"),patch("worker.gpu_inventory",return_value={}),\
             patch("worker.shutil.disk_usage",return_value=SimpleNamespace(free=150*2**30)),\
             patch("worker.execute",side_effect=execute):
            run_queue(SimpleNamespace(root=self.root,node="node02",resume_evaluation=False))
        expected=[run["run_id"] for run in node["runs"]]
        self.assertEqual(launched,expected)
        self.assertEqual(read(self.root/"status.json")["failed"],expected)
        self.assertEqual(len(list((self.root/"failure_receipts").glob("*.json"))),3)

    def test_export_long_final_is_60k_and_oom_has_stop_step_and_count(self):
        run=self.plan["nodes"][1]["runs"][-1];out=self.root/"runs"/run["run_id"]
        training_fixture(out,run)
        def evaluation(step,value):
            return dict(run_id=run["run_id"],step=step,N=run["method"]["cap"],views=2,
                        PSNR=value,SSIM=.8,LPIPS=.2,
                        per_view=[dict(view=i,image=f"{i}.png",PSNR=value,SSIM=.8,LPIPS=.2) for i in range(2)])
        write(out/"evaluation_30000.json",evaluation(30000,25.))
        write(out/"evaluation_45000.json",evaluation(45000,26.))
        write(out/"evaluation_60000.json",evaluation(60000,27.))
        failed_run=self.plan["nodes"][1]["runs"][0];_,log=self.oom_fixture(failed_run)
        failure=cuda_oom_record(self.root,self.plan,failed_run,{},"training",log)
        write(self.root/"failure_receipts"/f"{failed_run['run_id']}.json",failure)
        output=self.root/"export"
        export(self.plan,{"node02":self.root},output)
        with (output/"runs.csv").open() as stream:rows={r["run_id"]:r for r in csv.DictReader(stream)}
        self.assertEqual(float(rows[run["run_id"]]["PSNR"]),27.)
        self.assertEqual(rows[failed_run["run_id"]]["failure_G"],"2000000")
        self.assertEqual(rows[failed_run["run_id"]]["status"],"cuda_oom")


if __name__=="__main__":unittest.main()
