import unittest
import torch
from sparse_grt.sampling import StratifiedGrid
from sparse_grt.losses import coordinate_ssim_all as C


class AllCentres(unittest.TestCase):
    def test_matches_all_pairs_with_incomplete_edge_blocks(self):
        torch.set_num_threads(2)
        for height,width,block in [(1,1,8),(11,17,8),(31,29,6)]:
            grid=StratifiedGrid(height,width,block)
            s=grid.sample(torch.Generator().manual_seed(19))
            h,w=grid.shape
            weight,k=C.reference_weights(s.y.reshape(h,w).double(),s.x.reshape(h,w).double(),block,6.)
            p=torch.stack((s.y,s.x),-1).double()
            delta=p[:,None]-p[None]
            pair=torch.exp(-delta.square().sum(-1)/72.)*(delta.abs().amax(-1)<=18)
            pair=pair/pair.sum(1,keepdim=True)
            gt=torch.rand(h,w,3,dtype=torch.float64)
            pred=torch.rand_like(gt,requires_grad=True)
            mx=pair@pred.reshape(-1,3);my=pair@gt.reshape(-1,3)
            vx=pair@pred.reshape(-1,3).square()-mx.square()
            vy=pair@gt.reshape(-1,3).square()-my.square()
            cov=pair@(pred*gt).reshape(-1,3)-mx*my
            brute=(1-(2*mx*my+.0001)*(2*cov+.0009)/((mx.square()+my.square()+.0001)*(vx+vy+.0009))).mean()
            actual=C.reference_loss(pred,gt,weight,k)
            ga,=torch.autograd.grad(actual,pred,retain_graph=True)
            gb,=torch.autograd.grad(brute,pred)
            torch.testing.assert_close(actual,brute,atol=1e-12,rtol=1e-10)
            torch.testing.assert_close(ga,gb,atol=1e-12,rtol=1e-10)
