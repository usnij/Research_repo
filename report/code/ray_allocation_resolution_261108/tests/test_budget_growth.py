import math
from pathlib import Path
import struct
import tempfile
import unittest

from sparse_grt.budget import continuous_count, short_track_fraction
from sparse_grt.growth import GrowthPlan


class BudgetGrowth(unittest.TestCase):
    def test_frozen_formula_and_missing_tracks(self):
        rounded, raw = continuous_count(.22370188455380408, .1858328089079537)
        self.assertEqual(rounded, 1229851)
        self.assertAlmostEqual(raw, 1229850.8651177606, places=7)
        with self.assertRaises(ValueError): continuous_count(.2, None)
        with self.assertRaises(ValueError): continuous_count(0., .2)

    def test_training_only_incidence_and_duplicate_observations(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'images.bin'
            with p.open('wb') as f:
                f.write(struct.pack('<Q',4))
                for name,ids in [('a.jpg',[1,1,2]),('b.jpg',[1,3]),('c.jpg',[1,3]),
                                 ('test.jpg',[2,3,99])]:
                    f.write(bytes(64));f.write(name.encode()+b'\x00');f.write(struct.pack('<Q',len(ids)))
                    for i in ids:f.write(struct.pack('<ddq',0.,0.,i))
            # TRAIN degrees are {1:3, 2:1, 3:2}. Test-only point99 is excluded.
            self.assertEqual(short_track_fraction(p,['a.jpg','b.jpg','c.jpg']),2/3)

    def test_horizon_and_resume_are_initial_count_independent(self):
        for n,cap in [(155767,1229851),(54275,3021939),(32049,2824473),(9,10),(10,10)]:
            plan=GrowthPlan(n,cap,600,5000)
            counts=[plan.count_after(t) for t in range(30000)]
            self.assertEqual(counts[599],n)
            self.assertEqual(counts[5000],cap)
            self.assertEqual(counts[-1],cap)
            self.assertEqual(sorted(counts),counts)
            self.assertEqual(sum(r['added'] for r in plan.record()['events']),cap-n)
            for t in [699,700,1500,4999,5000,15000]:
                resumed=GrowthPlan(**vars(plan))
                self.assertEqual(resumed.count_after(t),counts[t])
            rate=plan.record()['unrounded_relative_add']
            self.assertAlmostEqual(n*(1+rate)**len(plan.events),cap,delta=1e-7*cap)
        with self.assertRaises(ValueError):GrowthPlan(20,10,600,5000)
        with self.assertRaises(ValueError):GrowthPlan(10,20,600,5050)


if __name__=='__main__':unittest.main()
