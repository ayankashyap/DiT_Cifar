import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random
from model import DiT
from tqdm.auto import tqdm


def normalize_to_neg1_1(x):
    return x * 2 - 1

def unnormalize_to_0_1(x):
    return (x + 1) * 0.5

class CFM(nn.Module):

    def __init__(self,
                 model,
                 device,
                 method='euler',
                 class_drop_prob=0.1,
                 cfg_scale=5.0,
                 ):
        super().__init__()

        self.model = model.to(device)
        self.num_classes = model.num_classes
        self.channels = model.in_channels
        self.input_size = model.input_size
        self.dim = model.dim
        self.device = device
        self.method = method

        # for cfg
        self.class_drop_prob = class_drop_prob
        self.cfg_scale = cfg_scale


    def forward(self, x, c):
        
        B, C, H, W = x.shape
        dtype = x.dtype
        x = normalize_to_neg1_1(x) # x:(0,1)->(-1,1)
        x1 = x
        x0 = torch.randn_like(x1)

        
        t = torch.rand(B, dtype=dtype, device=self.device)
        t_ = t[:, None, None, None]

        # euler method
        x_t = (1 - t_) * x0 + t_ * x1
        flow = x1 - x0

        # cfg training with class dropout
        self.drop_class = False
        if random() < self.class_drop_prob:
            self.drop_class = True

        pred = self.model(x_t=x_t, t=t, c=c, drop_class=self.drop_class)
        loss = F.mse_loss(pred, flow)
        return loss, pred

    
    @torch.no_grad()
    def sample(self, num_imgs, c=None, steps=50, cfg_scale=5.0, return_traj=False):
        self.eval()
        if c is not None:
            assert c.shape[0] == num_imgs
            c = c.to(self.device)
        else:
            c = torch.randint(0, self.num_classes, (num_imgs, ), device=self.device)


        z = torch.randn((num_imgs, self.channels, self.input_size, self.input_size), device=self.device)
        ts = torch.linspace(0, 1, steps, device=self.device)
        imgs = [z]
        for t in tqdm(ts):
            t = t.repeat(num_imgs)
            pred = self.model(x_t=z, t=t, c=c, drop_class=False)
            null_pred = self.model(x_t=z, t=t, c=c, drop_class=True)
            pred = pred + (pred - null_pred) * cfg_scale # cfg

            # euler
            z = z + (pred / steps) # z_{t+1} = z_t + dz/dt * step_size 
            if return_traj:
                imgs.append(z)

        z = unnormalize_to_0_1(z.clip(-1, 1))
        if return_traj:
            return z, imgs
        return z


if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32).to("cuda")
    c = torch.randint(0, 10, (2,)).to("cuda")
    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
        hidden_scale=4,
        pos_encoding="rope",
    )
    # model = model.to("cuda")
    cfm = CFM(model=model, device="cuda")
    cfm = cfm.to("cuda")
    # print(cfm)
    # loss, pred = cfm(x, c)
    # print(loss)
    # loss.backward()
    # print(pred.shape)
    # # print(cfm)
    # z = cfm.sample(2, c=c, steps=5)
    # print(z.shape) # torch.Size([2, 3, 32, 32])
    # z = cfm.sample(2, steps=5)
    # print(z.shape)
    # # sample every class
    # c =  torch.arange(10)
    # z = cfm.sample(10, c=c, steps=5)
    # from torch.nn.utils import parameters_to_vector
    # print("Total parameters", sum(p.numel() for p in cfm.parameters()))
    # print("Total parameters", sum(p.numel() for p in parameters_to_vector(cfm.parameters())))
    for i, (name, param) in enumerate(cfm.named_parameters()):
        print(i, name, param.shape)

