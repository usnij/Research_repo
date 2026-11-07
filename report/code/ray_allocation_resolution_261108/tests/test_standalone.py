import json
from pathlib import Path
import sys
import unittest

from sparse_grt.config import MethodConfig
from sparse_grt import standalone_backend as B


class Standalone(unittest.TestCase):
    def test_resolved_scientific_config_matches_frozen_reference(self):
        root=Path(__file__).resolve().parents[1]
        release=root/'releases/paper_v1_20261102'
        method=MethodConfig.read(release/'configs/counter_structure.json')
        expected=json.loads((release/'resolved_reference.json').read_text())
        cfg=B.configure(B.load(),'/tmp/example-scene',2,method)
        actual=B.OmegaConf.to_container(cfg,resolve=True)
        actual['path']=expected['path']
        self.assertEqual(actual,expected)

    def test_headless_import_has_no_legacy_modules(self):
        forbidden=('threedgrut','threedgrt_tracer','area_tracer512','lib3dgrt','lib_mcmc')
        self.assertFalse([n for n in sys.modules if n.startswith(forbidden)])


if __name__=='__main__':
    unittest.main()
