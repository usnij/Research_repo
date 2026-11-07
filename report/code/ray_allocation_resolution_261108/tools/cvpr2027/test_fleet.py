"""CPU checks for launch gates and experiment identity; no training is run."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from common import load_plan, signature, write
from export_results import checked_evaluation, export
from prepare_plan import load_assignments
from worker import gate, machine_lock


class FleetGuards(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = dict(plan_signature="test-plan", first_run_id="first")

    def tearDown(self):
        self.temp.cleanup()

    def test_first_run_needs_successful_reference_eval(self):
        with self.assertRaises(FileNotFoundError):
            gate(self.root, self.plan, dict(run_id="first"))
        write(self.root / ".baseline_reevaluation_verified.json", dict(passed=False))
        with self.assertRaises(RuntimeError):
            gate(self.root, self.plan, dict(run_id="first"))
        write(self.root / ".baseline_reevaluation_verified.json", dict(passed=True))
        gate(self.root, self.plan, dict(run_id="first"))

    def test_other_runs_need_exact_two_thousand_gate(self):
        ack = dict(plan_signature="test-plan", first_run_id="first", step=2000, checkpoint_verified=True)
        with self.assertRaises(FileNotFoundError):
            gate(self.root, self.plan, dict(run_id="second"))
        write(self.root / ".first_2k_gate.json", dict(ack, step=1999))
        with self.assertRaises(RuntimeError):
            gate(self.root, self.plan, dict(run_id="second"))
        write(self.root / ".first_2k_gate.json", dict(ack, plan_signature="another-plan"))
        with self.assertRaises(RuntimeError):
            gate(self.root, self.plan, dict(run_id="second"))
        write(self.root / ".first_2k_gate.json", ack)
        gate(self.root, self.plan, dict(run_id="second"))

    def test_duplicate_run_rejected_even_when_signature_is_valid(self):
        plan = dict(frozen_commit="fixture", code_files=[{"path": "fixture"}],
                    allow_experiments=["E1", "E2"],
                    nodes=[dict(runs=[dict(run_id="dup"), dict(run_id="dup")])])
        write(self.root / "plan.json", dict(plan, plan_signature=signature(plan)))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_plan(self.root / "plan.json")

    def test_machine_lock_prevents_concurrent_workers(self):
        a, b = self.root / "a", self.root / "b"
        a.mkdir(); b.mkdir()
        with machine_lock(a):
            with self.assertRaises(RuntimeError):
                with machine_lock(b):
                    self.fail("Second worker acquired same GPU lock")

    def test_private_hosts_are_external_and_scene_owners_unique(self):
        nodes = [dict(node_id=f"node{i}", instance_id=i+1, primary_scene=scene,
                      assigned_scenes=[scene], direct=dict(host=f"node{i}.example.invalid", port=22))
                 for i, scene in enumerate(("counter", "bonsai", "bicycle", "garden"))]
        path = self.root / "private_hosts.json"
        write(path, dict(nodes=nodes))
        self.assertEqual(len(load_assignments(path)), 4)
        nodes[1]["assigned_scenes"].append("counter")
        write(path, dict(nodes=nodes))
        with self.assertRaisesRegex(ValueError, "more than one"):
            load_assignments(path)

    def test_evaluator_mismatch_and_duplicate_views_rejected(self):
        path = self.root / "eval.json"
        run = dict(run_id="r", test_views=2)
        row = dict(PSNR=2.0, SSIM=0.5, LPIPS=0.3)
        value = dict(run_id="r", step=30000, views=2, **row,
                     per_view=[dict(view=0, **row), dict(view=1, **row)])
        write(path, value)
        self.assertEqual(checked_evaluation(path, run, 30000)["views"], 2)
        write(path, dict(value, PSNR=3.0))
        with self.assertRaisesRegex(ValueError, "Aggregate"):
            checked_evaluation(path, run, 30000)
        value["per_view"][1]["view"] = 0
        write(path, value)
        with self.assertRaisesRegex(ValueError, "duplicated"):
            checked_evaluation(path, run, 30000)


if __name__ == "__main__":
    unittest.main()
