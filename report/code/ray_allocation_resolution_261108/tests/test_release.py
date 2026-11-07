import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from sparse_grt.release import PaperRelease, difference


class Release(unittest.TestCase):
    def test_changed_deleted_sources_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'model.py'
            source.write_text('original')
            (root/'protocol.json').write_text(json.dumps({'release_id': 'test'}))
            (root/'source_lock.json').write_text(json.dumps(dict(
                workspace_files={'model.py': hashlib.sha256(source.read_bytes()).hexdigest()},
                external_files={}, environment={}, versions={}, runtime_modules={})))
            release = PaperRelease(root, root)
            self.assertTrue(release.verify()['passed'])
            source.write_text('changed')
            with self.assertRaises(RuntimeError): release.verify()
            source.unlink()
            with self.assertRaises(RuntimeError): release.verify()

    def test_changed_or_added_config_is_not_ignored(self):
        a = dict(render=dict(degree=4, min_T=.001), growth=5000)
        b = dict(render=dict(degree=2, min_T=.001, extra=True), growth=5000)
        self.assertEqual(difference(a,b), ['render.degree','render.extra'])
