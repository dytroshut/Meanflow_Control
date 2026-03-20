import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

def adaptive_l2_loss(error, p=1.0, c=1e-3):
    delta_sq = torch.sum(error ** 2, dim=1)
    w = 1.0 / (delta_sq + c).pow(p)
    return (w.detach() * delta_sq).mean()

class MeanFlow:
    def __init__(self, dim=3, omega=0.0, warmup_steps=5000, **kwargs):
        self.dim = dim
        self.omega = omega
        self.warmup_steps = warmup_steps
        self.x1_data = None
        # kwargs absorbs flow_ratio and endpoint_mix from your runner seamlessly

    def set_prior_data(self, data):
        self.x1_data = data

    def _rotate_z(self, v, theta):
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)
        x, y, z = v[:, 0:1], v[:, 1:2], v[:, 2:3]
        return torch.cat([cos_t * x - sin_t * y, sin_t * x + cos_t * y, z], dim=1)

    def expmA_vec(self, t, v):
        return self._rotate_z(v, self.omega * t)

    def expm_minusA_vec(self, t, v):
        return self._rotate_z(v, -self.omega * t)

    def loss(self, model, x0, step):
        B, device = x0.shape[0], x0.device
        
        # Sample target Torus
        idx = torch.randint(0, self.x1_data.shape[0], (B,), device=device)
        x1 = self.x1_data[idx]
        
        # Sample time intervals
        t = torch.rand(B, 1, device=device)
        r = t + (1.0 - t) * torch.rand(B, 1, device=device)
        
        # Bridge Kinematics
        x1_target = self.expm_minusA_vec(torch.ones_like(t), x1)
        y_t = (1.0 - t) * x0 + t * x1_target
        z_t = self.expmA_vec(t, y_t)
        v_t = self.expmA_vec(t, x1_target - x0)
        
        # Physics Drift: Az
        az = torch.stack([-self.omega * z_t[:, 1], self.omega * z_t[:, 0], torch.zeros_like(z_t[:, 2])], dim=1)
        z_dot = az + v_t

        # 1. Primal evaluation (Calculated normally so gradients flow to model parameters)
        u = model(z_t, t, r)

        # 2. Directional derivative (Stop-gradient: detached params so it acts strictly as a target)
        params_detached = {k: v.detach() for k, v in model.named_parameters()}
        
        def fn_detached(z_in, t_in):
            return functional_call(model, params_detached, (z_in, t_in, r))
        
        with torch.no_grad():
            _, dot_u = jvp(fn_detached, (z_t, t), (z_dot, torch.ones_like(t)))

        # 3. Objective Loss
        if step < self.warmup_steps:
            loss = F.mse_loss(u, v_t)
        else:
            residual = (r - t) * dot_u - u + v_t
            loss = adaptive_l2_loss(residual)
            
        return loss, F.mse_loss(u, v_t)