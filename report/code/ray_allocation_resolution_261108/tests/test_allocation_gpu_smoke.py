"""CPU-only guard tests: importing this diagnostic must never start CUDA."""
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from tools.ray_allocation_261108.smoke_gpu import checkpoint_steps, parser, validate_paths


class SmokeGuards(unittest.TestCase):
    def test_bounded_checkpoints_cross_first_native_growth(self):
        self.assertEqual(checkpoint_steps(1), {1})
        self.assertEqual(checkpoint_steps(610), {1, 601, 610})
        for n in (0, 600, 30000):
            with self.assertRaises(ValueError):
                checkpoint_steps(n)

    def test_default_is_diagnostic_610_not_changed_method_horizon(self):
        a = parser().parse_args(['--root', '/x', '--scene', '/s', '--config', '/c',
                                 '--downsample', '4', '--arm', 'residual', '--output', '/o'])
        self.assertEqual(a.steps, 610)
        self.assertEqual(a.cache_mib, 1200)

    def test_never_write_campaign_results_or_input_data_or_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); scene = root/'data/garden'; scene.mkdir(parents=True)
            config = root/'method.json'; config.write_text('{}')
            def args(out):
                return SimpleNamespace(root=root, scene=scene, config=config, output=out,
                                       downsample=4, cache_mib=1200, _phase=None)
            for output in (root/'runs', root/'runs/run_a', scene/'oops'):
                with self.assertRaises(ValueError):
                    validate_paths(args(output))
            existing = root/'smoke_old'; existing.mkdir()
            with self.assertRaises(FileExistsError):
                validate_paths(args(existing))
            self.assertEqual(validate_paths(args(root/'smoke/residual')).output, root/'smoke/residual')


if __name__ == '__main__':
    unittest.main()
