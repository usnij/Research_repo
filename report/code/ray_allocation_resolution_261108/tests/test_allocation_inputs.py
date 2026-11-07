"""CPU contracts for exact quotas, residual feedback, and view-plan geometry."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import torch

from sparse_grt.allocation_inputs import (
    ARMS, AllocationInputs, allocation_budget, grid_ray_count,
    initial_score_map, sample_coordinates,
)
from sparse_grt.config import MethodConfig
from sparse_grt.inputs import SelectedInputs
from sparse_grt.losses import coordinate_ssim_all as C


class Dataset:
    def __init__(self, height=96, width=101):
        generator = torch.Generator().manual_seed(143)
        self.rgb = [torch.randint(256, (1, height, width, 3), dtype=torch.uint8,
                                  generator=generator) for _ in range(5)]
        self.ro = torch.rand(1, height, width, 3, generator=generator)
        self.rd = torch.rand(1, height, width, 3, generator=generator)
        self.poses = [torch.eye(4)[None] for _ in self.rgb]
        for i, pose in enumerate(self.poses):
            pose[0, :3, 3] = torch.tensor([i, i*2, -i])

    def __len__(self): return len(self.rgb)
    def __getitem__(self, i): return dict(data=self.rgb[i], pose=self.poses[i], intr=0)
    def _lazy_worker_intrinsics_cache(self): return {0: ({}, self.ro, self.rd, 'test')}


def config(**kw):
    return replace(MethodConfig(views=4, block=6, sigma=6., boundary='all',
                                sampler='point', l1_weight=.8), **kw)


class AllocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_scene_budgets_and_exact_two_phase_totals(self):
        for h, w, n6, n8, late in ((1038,1558,44980,25350,64610),
                                    (840,1297,30380,17115,43645),
                                    (822,1237,28359,15965,40753)):
            for arm in ARMS:
                first = allocation_budget(h,w,arm,15000)
                last = allocation_budget(h,w,arm,15001)
                self.assertEqual(first['target_count']+last['target_count'],2*n6)
                self.assertEqual(15000*4*(first['target_count']+last['target_count']),30000*4*n6)
                self.assertEqual(first['base_count']+first['bonus_count'],first['target_count'])
                self.assertEqual(last['base_count']+last['bonus_count'],last['target_count'])
                if arm in ('schedule','combined'):
                    self.assertEqual(first['target_count'],n8)
                    self.assertEqual(last['target_count'],late)
            self.assertGreater(allocation_budget(h,w,'schedule',15001)['thinned_count'],0)
        with self.assertRaises(ValueError): allocation_budget(96,101,'unknown',1)
        with self.assertRaises(ValueError): allocation_budget(96,101,'uniform',0)
        # Some tiny toy images cannot meet 2*N6-N8 by thinning a block-5 grid.
        with self.assertRaises(ValueError): allocation_budget(72,80,'schedule',15001)

    def test_gt_sample_luminance_std_including_partial_boundaries(self):
        generator = torch.Generator().manual_seed(90)
        rgb = torch.randint(256,(11,19,3),generator=generator,dtype=torch.uint8)
        got = initial_score_map(rgb)
        expected = []
        for y in range(0,11,8):
            for x in range(0,19,8):
                patch = rgb[y:y+8,x:x+8].double()/255.
                lum = patch @ torch.tensor([.2126,.7152,.0722],dtype=torch.float64)
                expected.append(lum.std(unbiased=True))
        torch.testing.assert_close(got,torch.stack(expected).float(),atol=1e-7,rtol=1e-6)
        self.assertEqual(initial_score_map(torch.full((9,9,3),91,dtype=torch.uint8)).count_nonzero(),0)
        self.assertEqual(initial_score_map(torch.tensor([[[7,83,201]]],dtype=torch.uint8)).item(),0)
        with self.assertRaises(TypeError): initial_score_map(rgb.float())

    def test_sample_bounds_exact_counts_stable_block_order_and_private_rng(self):
        h,w = 96,101
        scores = torch.arange(grid_ray_count(h,w,8),dtype=torch.float32)+1
        for arm in ARMS:
            for step in (1,15000,15001,30000):
                before = torch.get_rng_state().clone()
                sample = sample_coordinates(h,w,arm,step,777,scores)
                torch.testing.assert_close(torch.get_rng_state(),before,atol=0,rtol=0)
                self.assertEqual(sample.y.numel(),sample.budget['target_count'])
                self.assertEqual(int(sample.is_base.sum()),sample.budget['base_count'])
                self.assertTrue(bool(((sample.y>=0)&(sample.y<h)&(sample.x>=0)&(sample.x<w)).all()))
                if arm in ('residual','combined'):
                    self.assertTrue(bool((sample.score_blocks[1:]>=sample.score_blocks[:-1]).all()))
                    for block in sample.score_blocks.unique():
                        flags = sample.is_base[sample.score_blocks==block]
                        # Stable sort preserves the base-before-bonus concatenation.
                        self.assertTrue(bool((flags[:-1].long()>=flags[1:].long()).all()))
                if arm in ('uniform','schedule'):
                    self.assertEqual(torch.unique(sample.y*w+sample.x).numel(),sample.y.numel())

    def test_zero_score_fallback_and_repeated_pixel_draws_are_preserved(self):
        h,w=96,101
        scores=torch.zeros(grid_ray_count(h,w,8))
        uniform=sample_coordinates(h,w,'residual',1,23,scores)
        self.assertEqual(uniform.y.numel(),grid_ray_count(h,w,6))
        scores[0]=1
        focused=sample_coordinates(h,w,'combined',15001,23,scores)
        self.assertTrue(bool((focused.score_blocks[~focused.is_base]==0).all()))
        # More than 64 bonus rays in a single 8x8 block guarantees duplicates.
        self.assertGreater(focused.budget['bonus_count'],64)
        self.assertLess(torch.unique(focused.y*w+focused.x).numel(),focused.y.numel())

    def test_uniform_is_bitwise_existing_adapter_and_resident_choices_match(self):
        ds=Dataset()
        for resident in (0,10**8):
            old=SelectedInputs(ds,SimpleNamespace,config(),device='cpu',
                               resident_budget_bytes=resident,weight_fn=C.reference_weights)
            new=AllocationInputs(ds,SimpleNamespace,config(),allocation_arm='uniform',device='cpu',
                                 resident_budget_bytes=resident,weight_fn=C.reference_weights)
            a,_,pa=old.make([4,0,2,1],777)
            b,_,pb=new.make([4,0,2,1],777)
            for key in ('rays_ori','rays_dir','rgb_gt','T_to_world'):
                torch.testing.assert_close(getattr(a,key),getattr(b,key),atol=0,rtol=0)
            for x,y in zip(pa,pb):
                for key in ('gt','weights','y','x'):
                    torch.testing.assert_close(x[key],y[key],atol=0,rtol=0)
                self.assertEqual(x['shape'],y['shape'])
                self.assertFalse(y['point_mode'])

    def test_nonuniform_plans_gather_real_camera_rays_rgb_and_offsets(self):
        ds=Dataset()
        for arm in ('schedule','residual','combined'):
            inputs=AllocationInputs(ds,SimpleNamespace,config(),allocation_arm=arm,device='cpu')
            for step in (1,15001):
                inputs.set_step(step)
                batch,_,plans=inputs.make([4,0,2,1],777)
                offset=0
                for j,p in enumerate(plans):
                    vid=p['view_id'];y,x=p['y'],p['x'];n=y.numel()
                    self.assertEqual(p['start'],offset);self.assertEqual(p['end'],offset+n)
                    self.assertTrue(p['point_mode']);self.assertIsNone(p['weights'])
                    torch.testing.assert_close(p['point_coords'],torch.stack((y,x),-1),atol=0,rtol=0)
                    torch.testing.assert_close(batch.rgb_gt[j],(ds.rgb[vid][:,y,x]/255.).reshape(1,n,3),atol=0,rtol=0)
                    torch.testing.assert_close(batch.rays_dir[j],ds.rd[:,y,x].reshape(1,n,3),atol=0,rtol=0)
                    expected=ds.ro[:,y,x]+ds.poses[vid][0,:3,3]
                    torch.testing.assert_close(batch.rays_ori[j],expected.reshape(1,n,3),atol=0,rtol=0)
                    offset+=n
                self.assertEqual(offset,batch.rgb_gt.numel()//3)

    def test_ema_once_per_block_observed_only_duplicate_aware_and_detached(self):
        ds=Dataset();inputs=AllocationInputs(ds,SimpleNamespace,config(),allocation_arm='combined',device='cpu')
        inputs.set_step(5)
        # Force bonus concentration, leaving some 8x8 score blocks unobserved.
        for score in inputs.score_maps.values(): score.zero_();score[0]=1
        batch,_,plans=inputs.make([4,0,2,1],777)
        before={v:s.clone() for v,s in inputs.score_maps.items()}
        delta=torch.arange(batch.rgb_gt.numel(),dtype=torch.float32).reshape_as(batch.rgb_gt)/batch.rgb_gt.numel()
        pred=(batch.rgb_gt+delta).requires_grad_()
        error=(pred.detach()-batch.rgb_gt).abs().reshape(-1,3).mean(-1)
        inputs.update_residual(pred,batch.rgb_gt,plans)
        self.assertIsNone(pred.grad)
        for p in plans:
            v=p['view_id'];ids=p['score_blocks'];r=error[p['start']:p['end']]
            expected=before[v].clone()
            for block in ids.unique():
                expected[block]=.9*before[v][block]+.1*r[ids==block].mean()
                self.assertEqual(inputs.update_counts[v][block],1)
                self.assertEqual(inputs.last_steps[v][block],5)
                self.assertEqual(inputs.observation_counts[v][block],int((ids==block).sum()))
            torch.testing.assert_close(inputs.score_maps[v],expected,atol=2e-7,rtol=1e-6)
            self.assertFalse(inputs.score_maps[v].requires_grad)
        # The unselected view never updates.
        torch.testing.assert_close(inputs.score_maps[3],before[3],atol=0,rtol=0)
        self.assertEqual(inputs.observation_counts[3].sum(),0)
        with self.assertRaises(ValueError): inputs.update_residual(pred,batch.rgb_gt,plans)

    def test_state_roundtrip_replays_sampling_and_rejects_incompatible_state(self):
        ds=Dataset()
        a=AllocationInputs(ds,SimpleNamespace,config(),allocation_arm='residual',device='cpu')
        batch,_,plans=a.make([0,1,2,3],777)
        a.update_residual(batch.rgb_gt+.3,batch.rgb_gt,plans)
        state=a.state_dict()
        b=AllocationInputs(ds,SimpleNamespace,config(),allocation_arm='residual',device='cpu')
        b.load_state_dict(state)
        for inputs in (a,b): inputs.set_step(2)
        aa,_,pa=a.make([4,0,2,1],778);bb,_,pb=b.make([4,0,2,1],778)
        torch.testing.assert_close(aa.rgb_gt,bb.rgb_gt,atol=0,rtol=0)
        for x,y in zip(pa,pb):
            torch.testing.assert_close(x['point_coords'],y['point_coords'],atol=0,rtol=0)
        bad=copy.deepcopy(state);bad['arm']='combined'
        with self.assertRaises(ValueError): b.load_state_dict(bad)
        bad=copy.deepcopy(state);bad['score_maps'][0][0]=float('nan')
        with self.assertRaises(ValueError): b.load_state_dict(bad)


if __name__=='__main__': unittest.main()
