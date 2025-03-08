class CFM:
    def __init__(self, model, device, method='euler', 
                 class_drop_prob=0.1, cfg_scale=5.0):
        # Initialize model parameters
        self.model = model
        self.device = device
        self.method = method
        self.class_drop_prob = class_drop_prob
        self.cfg_scale = cfg_scale

    def forward(self, x, c):
        ...

    @torch.no_grad()
    def sample(self, num_imgs, c, steps=50, cfg_scale=5.0, return_traj=False):
        self.model.eval()
        c = c.to(self.device) 
        z = torch.randn((num_imgs, self.model.in_channels, self.model.input_size, self.model.input_size), device=self.device)
        ts = torch.linspace(0, 1, steps, device=self.device)
        imgs = [z] if return_traj else None

        for t in ts:
            t = t.repeat(num_imgs)
            pred = self.model(x_t=z, t=t, c=c, drop_class=False)
            null_pred = self.model(x_t=z, t=t, c=c, drop_class=True)
            pred = pred + (pred - null_pred) * cfg_scale  # Classifier-free guidance
            z = z + pred / steps  # Euler method step
            if return_traj:
                imgs.append(z)

        z = unnormalize_to_0_1(z.clamp(-1, 1))
        return (z, imgs) if return_traj else z

    

import torch.nn as nn
import torch.nn.functional as F
from dit_block import *
from cfm import *