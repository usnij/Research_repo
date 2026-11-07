"""Failure-path tests for deployment and completed-run acceptance (no GPU)."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from benchmark_common import sha, verify_files, verify_complete, write
from benchmark_fleet import BOOTSTRAP, nodes_from
from benchmark_prepare import allocate
from benchmark_worker import verify_run_protocol


class BenchmarkPipelineTests(unittest.TestCase):
    def test_changed_data_and_escaping_path_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'image';p.write_bytes(b'abc')
            records=[dict(path='image',bytes=3,sha256=sha(p))]
            verify_files(root,records)
            p.write_bytes(b'abd')
            with self.assertRaises(ValueError):verify_files(root,records)
            with self.assertRaises(ValueError):verify_files(root,[dict(records[0],path='../image')])

    def completed_fixture(self, root):
        write(root/'complete.json',dict(passed=True,steps=30000,N=100,from_scratch=True))
        write(root/'protocol.json',dict(method=dict(cap=100)))
        e=dict(views=1,N=100,PSNR=30.,SSIM=.9,LPIPS=.1)
        for step in (7000,15000,30000):
            write(root/f'evaluation_{step}.json',dict(e,per_view=[e]))

    def test_incomplete_and_missing_final_evaluation_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.completed_fixture(root)
            verify_complete(root)
            (root/'evaluation_30000.json').unlink()
            with self.assertRaises(FileNotFoundError):verify_complete(root)
            self.completed_fixture(root)
            write(root/'complete.json',dict(passed=True,steps=15000,N=100,from_scratch=True))
            with self.assertRaises(ValueError):verify_complete(root)

    def test_resumed_run_cannot_masquerade_as_from_scratch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.completed_fixture(root)
            write(root/'complete.json',dict(passed=True,steps=30000,N=100,from_scratch=False))
            with self.assertRaises(ValueError):verify_complete(root)

    def test_scene_assignments_unique(self):
        rows=[dict(scene=str(i),scheduling_weight=i+1) for i in range(13)]
        plan=allocate(rows,8)
        assigned=[s for n in plan for s in n['scenes']]
        self.assertEqual(sorted(assigned),sorted(r['scene'] for r in rows))
        self.assertEqual(len(plan),8)

    def test_duplicate_jobs_and_hosts_rejected(self):
        one=dict(name='gpu01',host='example.com',port=22,root='/workspace/check',scenes=['counter'])
        nodes_from(dict(nodes=[one]))
        for two in (dict(one,name='gpu02'),dict(one,name='gpu02',host='elsewhere.com')):
            with self.assertRaises(ValueError):nodes_from(dict(nodes=[one,two]))

    def test_bootstrap_shell_is_valid(self):
        body=BOOTSTRAP.replace('__ROOT__',"'/workspace/check'").replace('__EXISTING__',"''")
        subprocess.run(['bash','-n'],input=body,text=True,check=True)

    def test_method_change_rejected_after_smoke(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cfg=dict(cap=100,l1_weight=.8)
            audit=dict(train_views=4,target_N=100,initial_N=50,rays_per_update=400)
            write(root/'protocol.json',dict(method=cfg,train_views=4,budget_method='fine_tracks_v1',
                  extra_training_rays=0,growth='log_horizon',growth_end=5000))
            write(root/'budget.json',dict(cap=100,test_RGB_used=False))
            write(root/'initial.json',dict(N=50))
            write(root/'ray_geometry.json',{'0':dict(total_rays_if_equal_views=400)})
            verify_run_protocol(root,audit,cfg)
            with self.assertRaises(ValueError):verify_run_protocol(root,audit,dict(cfg,l1_weight=1.))


if __name__=='__main__':unittest.main()
