"""Compare two from-scratch smoke runs, including optimizer and RNG states."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def equal(x, y):
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and x.dtype == y.dtype and torch.equal(x,y)
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and x.dtype == y.dtype and np.array_equal(x,y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(equal(x[k],y[k]) for k in x)
    if isinstance(x, (tuple,list)):
        return type(x) == type(y) and len(x) == len(y) and all(equal(a,b) for a,b in zip(x,y))
    return x == y


def main(a):
    torch.set_num_threads(2)
    if a.output.exists():
        raise FileExistsError(a.output)
    paths=[a.original,a.current]
    states=[torch.load(str(p/'checkpoint_latest.pt'), map_location='cpu', mmap=True,
                       weights_only=False) for p in paths]
    keys=['positions','rotation','scale','density','a_raw','features_albedo','features_specular',
          'optimizer','torch_rng','cuda_rng','numpy_rng','python_rng','global_step',
          'manifest_hash','paired_step','scene_extent']
    checks={k:bool(equal(states[0][k],states[1][k])) for k in keys}
    inputs={f:json.loads((paths[0]/f).read_text()) == json.loads((paths[1]/f).read_text()) for f in
            ['initial.json','first_input.json','training_manifest.json','growth_plan.json','ray_geometry.json']}
    rows=[[json.loads(x) for x in (p/'steps.jsonl').read_text().splitlines()] for p in paths]
    step_fields_equal=len(rows[0])==len(rows[1])==610 and all(
        {k:v for k,v in x.items() if k!='seconds'} == {k:v for k,v in y.items() if k!='seconds'}
        for x,y in zip(*rows))
    result=dict(passed=all(checks.values()) and all(inputs.values()) and step_fields_equal,
                checkpoint_bit_equal=checks, input_and_schedule_equal=inputs,
                all_610_non_timing_step_fields_identical=step_fields_equal,
                initial_N=json.loads((paths[0]/'initial.json').read_text())['N'],
                final_N=len(states[0]['positions']),
                growth_events=json.loads((paths[0]/'complete.json').read_text())['growth_events'],
                runs=[str(p.resolve()) for p in paths],
                scope='610 updates compared by checkpoint, inputs, schedule and step values, including first ADD and relocation; not a full-training quality/speed measurement.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:
        raise SystemExit('Smoke parity failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--current',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
