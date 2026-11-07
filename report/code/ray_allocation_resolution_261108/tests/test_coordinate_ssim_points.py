"""Numerical contracts for arbitrary-point SSIM.

CPU tests always run. Set RUN_COORDINATE_POINTS_CUDA=1 to compile/run the GPU
tests on an explicitly assigned GPU; ordinary discovery never starts CUDA.
"""
import os
import unittest
import torch

from sparse_grt.losses import coordinate_ssim_all as grid_ssim
from sparse_grt.losses import coordinate_ssim_points as points
from sparse_grt.sampling import StratifiedGrid


def direct_score(pred, gt, weight):
    """Independent SSIM formula, rather than residual algebra."""
    mp, mt = weight @ pred, weight @ gt
    dp, dt = pred[None]-mp[:, None], gt[None]-mt[:, None]
    w = weight[..., None]
    vp, vt, cov = (w*dp.square()).sum(1), (w*dt.square()).sum(1), (w*dp*dt).sum(1)
    return (1-(2*mp*mt+.0001)*(2*cov+.0009)/
            ((mp.square()+mt.square()+.0001)*(vp+vt+.0009))).mean()


class PointSSIMCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_duplicate_and_boundary_against_direct_formula(self):
        y = torch.tensor([0, 0, 0, 1, 2, 20, 23, 23])
        x = torch.tensor([0, 0, 2, 1, 0, 22, 24, 24])
        g = torch.Generator().manual_seed(17)
        p = torch.rand(8, 3, generator=g, dtype=torch.float64, requires_grad=True)
        t = torch.rand(8, 3, generator=g, dtype=torch.float64)
        for sigma in (1.5, 6., 12.):
            delta = torch.stack((y, x), -1).double()
            delta = delta[:, None]-delta[None]
            w = torch.exp(-delta.square().sum(-1)/(2*sigma*sigma))
            w *= delta.abs().amax(-1) <= 3*sigma
            w /= w.sum(1, keepdim=True)
            a = points.loss(p, t, y, x, sigma, 24, 25)
            b = direct_score(p, t, w)
            ga, = torch.autograd.grad(a, p, retain_graph=True)
            gb, = torch.autograd.grad(b, p, retain_graph=True)
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
            torch.testing.assert_close(ga, gb, atol=1e-12, rtol=1e-10)

    def test_uniform_grid_value_and_gradient_parity(self):
        for height, width, block, sigma in [(11, 17, 6, 6.), (37, 43, 8, 6.), (41, 49, 16, 12.)]:
            g = torch.Generator().manual_seed(113)
            sampler = StratifiedGrid(height, width, block)
            sample = sampler.sample(g)
            gh, gw = sampler.shape
            p = torch.rand(gh*gw, 3, generator=g, dtype=torch.float64, requires_grad=True)
            t = torch.rand(gh*gw, 3, generator=g, dtype=torch.float64)
            weights, k = grid_ssim.reference_weights(sample.y.reshape(gh, gw).double(),
                                                     sample.x.reshape(gh, gw).double(), block, sigma)
            a = points.loss(p, t, sample.y, sample.x, sigma, height, width)
            b = grid_ssim.reference_loss(p.reshape(gh, gw, 3), t.reshape(gh, gw, 3), weights, k)
            ga, = torch.autograd.grad(a, p, retain_graph=True)
            gb, = torch.autograd.grad(b, p)
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
            torch.testing.assert_close(ga, gb, atol=1e-12, rtol=1e-10)

    def test_gradient_check_repeated_pixel_distinct_values(self):
        y = torch.tensor([0, 0, 0, 3, 3])
        x = torch.tensor([0, 0, 2, 3, 3])
        g = torch.Generator().manual_seed(4)
        p = torch.rand(5, 3, generator=g, dtype=torch.float64, requires_grad=True)
        t = torch.rand(5, 3, generator=g, dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda q: points.loss(q, t, y, x, 6., 4, 4),
                                                (p,), eps=1e-6, atol=1e-5, rtol=1e-4))

    def test_permutation_keeps_value_and_reorders_gradient(self):
        y = torch.tensor([0, 4, 0, 6, 4, 2])
        x = torch.tensor([0, 4, 0, 6, 4, 3])
        perm = torch.tensor([5, 3, 1, 4, 0, 2])
        g = torch.Generator().manual_seed(77)
        p = torch.rand(6, 3, generator=g, dtype=torch.float64, requires_grad=True)
        t = torch.rand(6, 3, generator=g, dtype=torch.float64)
        a = points.loss(p, t, y, x, 6., 7, 7)
        pp = p[perm].detach().requires_grad_()
        b = points.loss(pp, t[perm], y[perm], x[perm], 6., 7, 7)
        ga, = torch.autograd.grad(a, p)
        gb, = torch.autograd.grad(b, pp)
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(ga[perm], gb, atol=1e-12, rtol=1e-10)

    def test_multiplicity_is_not_deduplicated(self):
        p = torch.tensor([[.1, .2, .3], [.9, .8, .7]], dtype=torch.float64)
        t = torch.tensor([[.2, .3, .4], [.4, .5, .6]], dtype=torch.float64)
        y, x = torch.tensor([0, 0]), torch.tensor([0, 1])
        once = points.loss(p, t, y, x, 6., 1, 2)
        index = torch.tensor([0, 0, 0, 1])
        repeated = points.loss(p[index], t[index], y[index], x[index], 6., 1, 2)
        self.assertGreater(abs(float(once-repeated)), 1e-3)

    def test_single_sample_and_identical_images(self):
        p = torch.tensor([[.1, .5, .9]], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[.2, .6, .8]], dtype=torch.float64)
        y = x = torch.zeros(1, dtype=torch.long)
        actual = points.loss(p, t, y, x, 6., 1, 1)
        expected = ((p-t).square()/(p.square()+t.square()+.0001)).mean()
        torch.testing.assert_close(actual, expected)
        same = points.loss(p, p.detach(), y, x, 6., 1, 1)
        grad, = torch.autograd.grad(same, p)
        torch.testing.assert_close(same, torch.zeros_like(same))
        torch.testing.assert_close(grad, torch.zeros_like(grad))

    def test_square_cutoff_includes_corner_and_excludes_outside(self):
        y = torch.tensor([0, 18, 0])
        x = torch.tensor([0, 18, 19])
        p = torch.tensor([[.1]*3, [.6]*3, [.9]*3], dtype=torch.float64)
        t = torch.tensor([[.2]*3, [.4]*3, [.5]*3], dtype=torch.float64)
        dy = (y[:, None]-y[None]).double(); dx = (x[:, None]-x[None]).double()
        w = torch.exp(-(dy.square()+dx.square())/72.)*((dy.abs()<=18)&(dx.abs()<=18))
        self.assertGreater(float(w[0, 1]), 0.)
        self.assertEqual(float(w[0, 2]), 0.)
        w /= w.sum(1, keepdim=True)
        torch.testing.assert_close(points.loss(p, t, y, x, 6., 19, 20), direct_score(p, t, w),
                                   atol=1e-12, rtol=1e-10)

    def test_contract_rejects_invalid_coordinates_and_differentiable_gt(self):
        p = torch.zeros(1, 3, dtype=torch.float64)
        with self.assertRaises(ValueError):
            points.loss(p, p, torch.tensor([1]), torch.tensor([0]), 6., 1, 1)
        with self.assertRaises(ValueError):
            points.loss(p, p.clone().requires_grad_(), torch.tensor([0]), torch.tensor([0]), 6., 1, 1)


@unittest.skipUnless(os.environ.get('RUN_COORDINATE_POINTS_CUDA') == '1', 'GPU tests require explicit opt-in')
class PointSSIMCUDA(unittest.TestCase):
    def test_cuda_arbitrary_duplicates_and_boundaries(self):
        g = torch.Generator().manual_seed(23)
        y = torch.randint(25, (80,), generator=g)
        x = torch.randint(29, (80,), generator=g)
        y[:20], x[:20] = 0, 0
        p0, t = torch.rand(80, 3, generator=g), torch.rand(80, 3, generator=g)
        for sigma in (1.5, 6., 12.):
            p = p0.double().requires_grad_()
            ref = points.reference_loss(p, t.double(), y, x, sigma, 25, 29)
            gr, = torch.autograd.grad(ref, p)
            pc = p0.cuda().requires_grad_()
            actual = points.loss(pc, t.cuda(), y.cuda(), x.cuda(), sigma, 25, 29)
            gc, = torch.autograd.grad(actual, pc)
            torch.testing.assert_close(actual.cpu().double(), ref, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(gc.cpu().double(), gr, atol=3e-6, rtol=2e-4)

    def test_cuda_uniform_grid_matches_existing_kernel(self):
        for height, width, block, sigma in [(77, 91, 6, 6.), (83, 97, 16, 12.)]:
            g = torch.Generator(device='cuda').manual_seed(99)
            sampler = StratifiedGrid(height, width, block, device='cuda')
            sample = sampler.sample(g)
            gh, gw = sampler.shape
            p = torch.rand(gh*gw, 3, generator=g, device='cuda', requires_grad=True)
            t = torch.rand(gh*gw, 3, generator=g, device='cuda')
            w, k = grid_ssim.weights(sample.y.reshape(gh, gw), sample.x.reshape(gh, gw), block, sigma)
            a = points.loss(p, t, sample.y, sample.x, sigma, height, width)
            b = grid_ssim.loss(p.reshape(gh, gw, 3), t.reshape(gh, gw, 3), w, k)
            ga, = torch.autograd.grad(a, p, retain_graph=True)
            gb, = torch.autograd.grad(b, p)
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(ga, gb, atol=3e-6, rtol=2e-4)

    def test_cuda_concentrated_duplicates_have_no_quadratic_neighbour_list(self):
        n = 8192
        g = torch.Generator().manual_seed(908)
        p = torch.rand(n, 3, generator=g, dtype=torch.float64, requires_grad=True)
        t = torch.rand(n, 3, generator=g, dtype=torch.float64)
        mp, mt = p.mean(0), t.mean(0)
        dp, dt = p-mp, t-mt
        a = (mp-mt).square()/(mp.square()+mt.square()+.0001)
        b = (dp-dt).square().mean(0)/(dp.square().mean(0)+dt.square().mean(0)+.0009)
        ref = (a+b-a*b).mean()
        gr, = torch.autograd.grad(ref, p)
        pc = p.detach().float().cuda().requires_grad_()
        y = torch.zeros(n, dtype=torch.int32, device='cuda')
        actual = points.loss(pc, t.float().cuda(), y, y, 6., 1, 1)
        gc, = torch.autograd.grad(actual, pc)
        torch.testing.assert_close(actual.cpu().double(), ref, atol=2e-5, rtol=3e-5)
        torch.testing.assert_close(gc.cpu().double(), gr, atol=2e-7, rtol=3e-4)


if __name__ == '__main__':
    unittest.main()
