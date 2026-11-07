"""Run the standalone CLI with explicit forbidden source roots and report imports.

This detects Python reads of old experiment files. It does not replace a clean
machine test: the host's installed Python dependencies and GPU driver remain.
"""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deny-root', type=Path, action='append', default=[])
    p.add_argument('--allow-root', type=Path, action='append', default=[])
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('training_args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    denied = [x.resolve() for x in a.deny_root]
    allowed = [x.resolve() for x in a.allow_root]
    blocked = []

    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(d) for d in denied) and not any(path.is_relative_to(d) for d in allowed):
            blocked.append(str(path))
            raise RuntimeError(f'Forbidden dependency read: {path}')

    sys.addaudithook(audit)
    args = a.training_args
    if args[:1] == ['--']:
        args = args[1:]
    sys.argv = ['sparse-grt-train', *args]
    from sparse_grt.training import cli
    cli()
    from sparse_grt.native_build import BUILD_RECORDS
    modules = {name: str(Path(module.__file__).resolve()) for name, module in sys.modules.items()
               if name.startswith(('sparse_grt', '_sparse_grt')) and getattr(module, '__file__', None)}
    forbidden_modules = [n for n in sys.modules if n.startswith(
        ('threedgrut', 'threedgrt_tracer', 'area_tracer512', 'lib3dgrt', 'lib_mcmc'))]
    result = dict(passed=not blocked and not forbidden_modules, blocked_reads=blocked,
                  forbidden_modules=forbidden_modules, modules=modules, native_builds=BUILD_RECORDS,
                  denied_roots=list(map(str, denied)), allowed_roots=list(map(str, allowed)),
                  scope='Source isolation on existing local Python environment; not a fresh dependency installation')
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2)+'\n')
    if not result['passed']:
        raise SystemExit('Isolation check failed')


if __name__ == '__main__':
    main()
