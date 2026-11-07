"""CPU launch-safety checks for the independent N5 queue. No GPU calls."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from common import load_plan, read, signature, train_command, write
from prepare_plan import ARMS, load_assignments, native_count, prepare
from worker import gate, machine_lock, validate_identity, verify_training


class NativeFiveFleetGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_first_dense_run_alone_is_exempt_from_2k_ack(self):
        plan = dict(plan_signature="new-plan", first_run_id="native5-first")
        gate(self.root, plan, dict(run_id="native5-first"))
        with self.assertRaises(FileNotFoundError):
            gate(self.root, plan, dict(run_id="native5-next"))
        ack = dict(plan_signature="old-E1-plan", first_run_id="native5-first", step=2000, checkpoint_verified=True)
        write(self.root / ".first_2k_gate.json", ack)
        with self.assertRaises(RuntimeError):
            gate(self.root, plan, dict(run_id="native5-next"))
        write(self.root / ".first_2k_gate.json", dict(ack, plan_signature="new-plan"))
        gate(self.root, plan, dict(run_id="native5-next"))

    def test_cross_campaign_plan_is_rejected(self):
        body = dict(campaign="cvpr2027_261104", allow_experiments=["E1","E2"],
                    frozen_commit="fixture", code_files=[dict(path="fixture")],
                    nodes=[dict(runs=[dict(run_id="old-E1")])])
        write(self.root / "plan.json", dict(body, plan_signature=signature(body)))
        with self.assertRaisesRegex(ValueError, "independent N5"):
            load_plan(self.root / "plan.json")

    def test_no_duplicate_run_even_with_valid_signature(self):
        body = dict(campaign="native5_singleview_261105", allow_experiments=["N5"],
                    frozen_commit="fixture", code_files=[dict(path="fixture")],
                    nodes=[dict(runs=[dict(run_id="dup"),dict(run_id="dup")])])
        write(self.root / "plan.json", dict(body, plan_signature=signature(body)))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_plan(self.root / "plan.json")

    def test_machine_lock_covers_different_campaign_roots(self):
        a,b = self.root/"old",self.root/"new"
        a.mkdir();b.mkdir()
        with machine_lock(a):
            with self.assertRaises(RuntimeError):
                with machine_lock(b):
                    self.fail("Another queue acquired the same GPU")

    def test_node_instance_and_gpu_identity_must_match(self):
        node = dict(node_id="node02",instance_id=123)
        hardware = dict(uuid="GPU-fixture")
        identity = dict(node_id="node02",instance_id=123,gpu_uuid="GPU-fixture")
        validate_identity(identity,node,hardware,{})
        for corrupted in [dict(identity,node_id="node03"),dict(identity,instance_id=124),dict(identity,gpu_uuid="GPU-other")]:
            with self.assertRaises(RuntimeError):
                validate_identity(corrupted,node,hardware,{})
        with self.assertRaises(RuntimeError):
            validate_identity(identity,node,hardware,{"CONTAINER_ID":"124"})

    def test_native_growth_boundaries_and_cap(self):
        self.assertEqual(native_count(100,120,600),100)
        self.assertEqual(native_count(100,120,601),105)
        self.assertEqual(native_count(100,120,701),110)
        self.assertEqual(native_count(100,120,901),120)
        self.assertEqual(native_count(100,120,30000),120)
        self.assertGreater(native_count(100,1000000,6001),native_count(100,1000000,5001))

    def test_completed_grid4_run_checks_growth_counts_and_window(self):
        out=self.root/"completed";out.mkdir()
        run=dict(method=dict(cap=1000,grid_window=4,grid_sigma=1.5), initial_N=100,
                 train_views=2, structural_loss="grid", rays_by_shape={"103x83":980},
                 expected_G_by_checkpoint={str(step):native_count(100,1000,step)
                                           for step in (2000,5000,7000,10000,15000,20000,30000)})
        write(out/"training_complete.json",dict(passed=True,steps=30000,N=1000))
        proto=dict(method=run["method"],train_views=2,growth="native_5percent",growth_end=None,
                   structural_loss="grid",execution_protocol="native5_singleview",
                   native_add_schedule=dict(original_add_method_preserved=True),
                   compact_window=4,compact_sigma=1.5,observed_only=False,multiview_start_step=0)
        write(out/"protocol.json",proto)
        write(out/"resolved_config.json",dict(strategy=dict(add=dict(start_iteration=500,
              end_iteration=25000,frequency=100,placement="opacity",max_n_gaussians=1000))))
        write(out/"growth_plan.json",dict(policy="native_5percent",audit_only=True,
              initial=100,target=1000,start=500,end=25000,frequency=100))
        write(out/"initial.json",dict(N=100))
        write(out/"ray_geometry.json",{str(v):dict(width=103,height=83,total_rays_if_equal_views=980)
                                     for v in range(2)})
        for step,count in run["expected_G_by_checkpoint"].items():
            (out/f"ckpt_{step}.pt").write_bytes(b"fixture-only")
            write(out/f"checkpoint_verified_{step}.json",dict(step=int(step),G=count,
                  load_verified=True,optimizer_present=True))
        self.assertEqual(verify_training(out,run)["N"],1000)
        write(out/"protocol.json",dict(proto,compact_window=11))
        with self.assertRaisesRegex(RuntimeError,"compact SSIM window"):
            verify_training(out,run)
        write(out/"protocol.json",proto)
        proof=read(out/"checkpoint_verified_2000.json");write(out/"checkpoint_verified_2000.json",dict(proof,G=1000))
        with self.assertRaisesRegex(RuntimeError,"Gaussian count"):
            verify_training(out,run)

    def test_prepare_all_24_arms_and_native_commands(self):
        payload = self.root/"payload"
        nodes=[]
        for i,scene in enumerate(["bicycle","bonsai","garden"],1):
            base=dict(views=4,block=8,sigma=6.,l1_weight=.8,structure_weight=.2,
                      opacity_weight=.01,scale_weight=.01,cap=1000,steps=30000,
                      initialization_seed=777,view_seed=123,ray_seed=456,boundary="all")
            audit=dict(scene=scene,target_N=1000,initial_N=100,downsample=2,
                       train_views=10,test_views=2,shapes={"103x83":12})
            write(payload/"scenes"/scene/"structure.json",base)
            write(payload/"scenes"/scene/"audit.json",audit)
            nodes.append(dict(node_id=f"node{i:02}",primary_scene=scene,instance_id=i,
                              direct=dict(host="node.example.invalid",port=22),proxy=None,assigned_scenes=[scene]))
        hosts=self.root/"hosts.json";write(hosts,dict(nodes=nodes))
        args=SimpleNamespace(output=self.root/"fleet",payload=payload,hosts=hosts,
                             remote_root="/workspace/native5_singleview_261105",python="/env/python")
        plan=prepare(args)
        self.assertEqual(sum(len(n["runs"]) for n in plan["nodes"]),24)
        self.assertFalse((args.output/"plan.json").exists())
        for node in plan["nodes"]:
            self.assertEqual([r["arm"] for r in node["runs"]],[a[0] for a in ARMS])
            for run in node["runs"]:
                config=read(args.output/run["config_path"])
                self.assertEqual(config["views"],1)
                self.assertEqual(run["experiment_id"],"N5")
                command=train_command(plan["remote_root"],run,plan["python"])
                self.assertNotIn("--growth-end",command)
                self.assertEqual(command[command.index("--growth")+1],"native_5percent")
                self.assertEqual(command[command.index("--protocol")+1],"native5_singleview")
        self.assertEqual([r["method"]["sigma"] for r in plan["nodes"][0]["runs"][-2:]],[1.5,1.0])
        body=dict(plan,frozen_commit="fixture",code_files=[dict(path="fixture",sha256="unused")])
        frozen=args.output/"test_frozen_plan.json"
        write(frozen,dict(body,plan_signature=signature(body)))
        self.assertEqual(len(load_plan(frozen)["nodes"][0]["runs"]),8)
        truncated=dict(body,nodes=[dict(body["nodes"][0],runs=body["nodes"][0]["runs"][:6]),*body["nodes"][1:]])
        write(frozen,dict(truncated,plan_signature=signature(truncated)))
        with self.assertRaisesRegex(ValueError,"eight arms"):
            load_plan(frozen)
        wrong_order=dict(body,nodes=[dict(body["nodes"][0],runs=[*body["nodes"][0]["runs"][:-2],
                          body["nodes"][0]["runs"][-1],body["nodes"][0]["runs"][-2]]),*body["nodes"][1:]])
        write(frozen,dict(wrong_order,plan_signature=signature(wrong_order)))
        with self.assertRaisesRegex(ValueError,"arm order"):
            load_plan(frozen)
        nodes[1]["instance_id"]=nodes[0]["instance_id"]
        write(hosts,dict(nodes=nodes))
        with self.assertRaisesRegex(ValueError,"One GPU"):
            load_assignments(hosts)


if __name__ == "__main__":
    unittest.main()
