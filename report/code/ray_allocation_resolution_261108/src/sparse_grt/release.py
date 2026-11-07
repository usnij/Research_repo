"""Fail-closed guards for a frozen local research runtime.

The lock is an integrity record, not an independent-build/public-release claim.
It never rewrites a lock, installs a dependency, or changes training settings.
"""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def difference(expected, actual, prefix=''):
    """Return changed/missing/extra fields, including nested config keys."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        rows = []
        for key in sorted(set(expected) | set(actual)):
            name = f'{prefix}.{key}' if prefix else key
            if key not in expected or key not in actual:
                rows.append(name)
            else:
                rows.extend(difference(expected[key], actual[key], name))
        return rows
    return [] if expected == actual else [prefix]


class PaperRelease:
    def __init__(self, folder, workspace):
        self.folder, self.workspace = Path(folder).resolve(), Path(workspace).resolve()
        self.lock_path = self.folder / 'source_lock.json'
        self.lock = json.loads(self.lock_path.read_text())
        self.protocol = json.loads((self.folder / 'protocol.json').read_text())
        self.report = None

    def verify(self, environment=True):
        errors = []
        for name, expected in self.lock['workspace_files'].items():
            p = self.workspace / name
            if not p.is_file() or digest(p) != expected:
                errors.append(f'source: {name}')
        for name, expected in self.lock['external_files'].items():
            p = Path(name)
            if not p.is_file() or digest(p) != expected:
                errors.append(f'external binary: {name}')
        if environment:
            for key, value in self.lock['environment'].items():
                if os.environ.get(key) != value:
                    errors.append(f'environment: {key} expected {value!r}')
            for key, value in self.lock['versions'].items():
                try:
                    actual = importlib.metadata.version(key)
                except importlib.metadata.PackageNotFoundError:
                    actual = None
                if actual != value:
                    errors.append(f'package: {key} expected {value!r}, got {actual!r}')
        if errors:
            raise RuntimeError('Paper release mismatch; do not start training:\n' + '\n'.join(errors))
        self.report = dict(release_id=self.protocol['release_id'],
                           lock_sha256=digest(self.lock_path),
                           files_checked=len(self.lock['workspace_files']) + len(self.lock['external_files']),
                           environment_checked=environment, passed=True)
        return self.report

    def validate_method(self, method, preset):
        expected = json.loads((self.folder / 'configs' / f'{preset}.json').read_text())
        changed = difference(expected, vars(method))
        if changed:
            raise RuntimeError(f'Paper method differs from {preset}: {changed}')

    def validate_runtime(self, runtime, cfg, method, scene, downsample):
        expected = json.loads((self.folder / 'resolved_reference.json').read_text())
        expected['path'] = str(Path(scene).resolve())
        expected['dataset']['downsample_factor'] = downsample
        expected['strategy']['add']['max_n_gaussians'] = method.cap
        expected['loss']['lambda_l1'] = method.l1_weight
        expected['loss']['lambda_ssim'] = method.structure_weight
        actual = runtime.S.OmegaConf.to_container(cfg, resolve=True)
        changed = difference(expected, actual)
        if changed:
            raise RuntimeError(f'Paper backend configuration changed: {changed}')
        # Check dynamically imported modules too, including the extension
        # selected by TORCH_EXTENSIONS_DIR or a changed sys.path.
        for name, expected in self.lock['runtime_modules'].items():
            module = sys.modules.get(name)
            path = getattr(module, '__file__', None)
            if path is None or digest(path) != expected:
                raise RuntimeError(f'Loaded backend differs from lock: {name} ({path})')
        return dict(passed=True, modules_checked=len(self.lock['runtime_modules']),
                    resolved_config_matches=True)
