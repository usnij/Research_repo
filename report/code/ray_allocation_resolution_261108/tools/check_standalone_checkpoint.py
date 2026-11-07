"""Reload a trusted checkpoint with the standalone backend and exercise evaluation."""
import argparse
import json
from pathlib import Path
import torch
from sparse_grt import standalone_backend as B


def main(a):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.75)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(str(a.checkpoint), map_location='cpu', mmap=True, weights_only=False)
    cfg = checkpoint['config']
    if a.scene:
        cfg.path = str(a.scene.resolve())
    parameter_hash = B.digest({k:checkpoint[k] for k in B.KEYS})
    optimizer_hash = B.digest(checkpoint['optimizer'])
    for k in B.KEYS:
        checkpoint[k] = torch.nn.Parameter(checkpoint[k].cuda())
    model = B.MixtureOfGaussians(cfg)
    model.init_from_checkpoint(checkpoint, setup_optimizer=True)
    model.cuda()
    model.build_acc(rebuild=True)
    runner = B.Runner(model)
    model.eval()
    assert B.digest({k:getattr(model,k) for k in B.KEYS}) == parameter_hash
    assert B.digest(model.optimizer.state_dict()) == optimizer_hash
    runtime = B.load()
    train = B.dataset(runtime, cfg, 'train')
    batch = train.get_gpu_batch_with_intrinsics(torch.utils.data.default_collate([train[0]]))
    with torch.no_grad():
        first = model(batch, train=False)['pred_rgb']
        repeated = model(batch, train=False)['pred_rgb']
    assert torch.isfinite(first).all() and torch.equal(first, repeated)
    render_hash = B.digest(first)
    del batch,first,repeated,train
    torch.cuda.empty_cache()
    metrics = B.evaluate(runtime, model, cfg)
    result = dict(passed=True, step=int(checkpoint['global_step']), N=model.num_gaussians,
                  parameter_hash_preserved=True, optimizer_hash_preserved=True,
                  repeated_full_ray_render_bit_equal=True, train_view_render_hash=render_hash,
                  evaluation=metrics, peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                  scope='Checkpoint reload and evaluation plumbing; not final training quality')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='evaluation'},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--scene', type=Path)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
