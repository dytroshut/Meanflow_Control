import os
import time
import json
import copy
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from tqdm import tqdm

# ----- CLEAN IMPORTS -----
# Note: Ensure your model.py and meanflow3d.py are in the same directory.
from model import ResMLP3D 
from meanflow3d import MeanFlow

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    print("GPU:", torch.cuda.get_device_name(0))
    return "cuda"

# ----- EMA UTILITY (Exponential Moving Average) -----
class EMA:
    """Keeps a smoothed running average of model weights for pristine generation."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].sub_((1.0 - self.decay) * (self.shadow[name] - param.data))

    def apply_shadow(self, model):
        """Swap the model weights with the EMA weights for evaluation."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        """Restore the original training weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

# ----- 3D DATASET GENERATORS (SEPARATED) -----
def make_pyramid_dataset(n=20000, noise=0.01, seed=0):
    """Generates a 3D hollow Pyramid (Source) - Shifted Left"""
    rng = np.random.default_rng(seed)
    apex = np.array([0.0, 0.0, 0.5])
    v0, v1, v2, v3 = np.array([-0.5,-0.5,-0.5]), np.array([0.5,-0.5,-0.5]), \
                     np.array([0.5,0.5,-0.5]), np.array([-0.5,0.5,-0.5])
    
    faces = [
        (v0, v1, apex), (v1, v2, apex), (v2, v3, apex), (v3, v0, apex), 
        (v0, v1, v2), (v0, v2, v3) 
    ]
    
    pts = []
    pts_per_face = n // len(faces)
    for tri in faces:
        A, B, C = tri
        u, v = rng.uniform(0, 1, pts_per_face), rng.uniform(0, 1, pts_per_face)
        mask = (u + v) > 1
        u[mask], v[mask] = 1 - u[mask], 1 - v[mask]
        pts.append(A + u[:, None] * (B - A) + v[:, None] * (C - A))
        
    pts = np.vstack(pts)
    pts += noise * rng.normal(size=pts.shape)
    pts[:, 0] -= 1.5 
    return pts.astype(np.float32)

def make_torus_dataset(n=20000, noise=0.01, seed=0, R=0.6, r=0.2):
    """Generates a 3D hollow Torus with TRUE uniform surface density"""
    rng = np.random.default_rng(seed)
    pts = []
    while len(pts) < n:
        u = rng.uniform(0, 2 * np.pi, n)
        v = rng.uniform(0, 2 * np.pi, n)
        p = (R + r * torch.cos(torch.tensor(v))) / (R + r)
        accept = rng.uniform(0, 1, n) < p.numpy()
        u_acc, v_acc = u[accept], v[accept]
        x = (R + r * np.cos(v_acc)) * np.cos(u_acc)
        y = (R + r * np.cos(v_acc)) * np.sin(u_acc)
        z = r * np.sin(v_acc)
        pts.extend(np.column_stack([x, y, z]))
    pts = np.array(pts[:n])
    pts += noise * rng.normal(size=pts.shape)
    pts[:, 0] += 1.5 
    return pts.astype(np.float32)

def build_3D_dataset(RUN, device):
    P_raw = make_pyramid_dataset(n=RUN["n_data"], noise=RUN["shape_noise"], seed=RUN["seed"])
    T_raw = make_torus_dataset(n=RUN["n_data"], noise=RUN["shape_noise"], seed=RUN["seed"] + 1)
    x0 = torch.tensor(P_raw, dtype=torch.float32, device=device)
    x1 = torch.tensor(T_raw, dtype=torch.float32, device=device)
    return x0, x1

@torch.no_grad()
def sample_vis_subset(x, n):
    idx = torch.randperm(x.shape[0], device=x.device)[:n]
    return x[idx]

# ----- 3D ROTATION & FORWARD INFERENCE LOGIC -----
@torch.no_grad()
def expA_vec(dt: float, x: torch.Tensor, omega: float) -> torch.Tensor:
    if abs(float(omega)) < 1e-12: return x
    ang = omega * dt
    c, s = torch.cos(torch.tensor(ang, device=x.device)), torch.sin(torch.tensor(ang, device=x.device))
    x1, x2, x3 = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([c * x1 - s * x2, s * x1 + c * x2, x3], dim=1)

@torch.no_grad()
def gammaA_u(dt: float, u: torch.Tensor, omega: float) -> torch.Tensor:
    if abs(float(omega)) < 1e-12: return dt * u
    ang = omega * dt
    a = torch.sin(torch.tensor(ang, device=u.device)) / omega
    b = (1.0 - torch.cos(torch.tensor(ang, device=u.device))) / omega
    u1, u2, u3 = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    return torch.cat([a * u1 - b * u2, b * u1 + a * u2, dt * u3], dim=1)

@torch.no_grad()
def compute_trajectories(model, x0, omega, steps=80):
    model.eval()
    z = x0.clone()
    traj = [z.detach().cpu().numpy()]
    for k in range(steps):
        t, r = k / steps, (k + 1) / steps
        dt = r - t
        u_bar = model(z, torch.full((z.shape[0],), t, device=z.device), 
                         torch.full((z.shape[0],), r, device=z.device))
        z = expA_vec(dt, z, omega) + gammaA_u(dt, u_bar, omega)
        traj.append(z.detach().cpu().numpy())
    return np.stack(traj, axis=1)

@torch.no_grad()
def compute_snapshots_from_trajectories(traj, t_list):
    n_steps = traj.shape[1] - 1
    snaps = {float(t): traj[:, max(0, min(int(round(float(t) * n_steps)), n_steps)), :] for t in t_list}
    return snaps

class RunLogger:
    def __init__(self, root, dataset, method, config):
        self.run_name = f"{dataset}_{method}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.root = os.path.join(root, self.run_name)
        self.sample_dir = os.path.join(self.root, "samples")
        self.ckpt_dir = os.path.join(self.root, "checkpoints")
        self.log_dir = os.path.join(self.root, "logs")
        for d in [self.root, self.sample_dir, self.ckpt_dir, self.log_dir]: Path(d).mkdir(parents=True, exist_ok=True)
        self.log_txt = os.path.join(self.log_dir, "train_log.txt")
        with open(os.path.join(self.root, "run_config.json"), "w") as f: json.dump(config, f, indent=2)

    def log_train(self, step, loss, mse, lr):
        msg = f"[step {step:7d}] loss={loss:.8f}  mse={mse:.8f}  lr={lr:.3e}"
        print(msg); 
        with open(self.log_txt, "a") as f: f.write(msg + "\n")

    def save_checkpoint(self, step, model, ema, opt):
        torch.save({"step": step, "model": model.state_dict(), "ema_shadow": ema.shadow, "opt": opt.state_dict()}, 
                   os.path.join(self.ckpt_dir, f"ckpt_{step:07d}.pt"))

    def _set_3d_limits(self, ax, pts):
        xmin, ymin, zmin = pts.min(axis=0); xmax, ymax, zmax = pts.max(axis=0)
        cx, cy, cz = 0.5*(xmin+xmax), 0.5*(ymin+ymax), 0.5*(zmin+zmax)
        max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
        half = 0.42 * max_range 
        ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half); ax.set_zlim(cz - half, cz + half)
        ax.set_box_aspect([1, 1, 1]); ax.axis("off")

    def save_data_preview(self, x0, x1, step=0):
        P, T = x0.detach().cpu().numpy(), x1.detach().cpu().numpy()
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(P[:, 0], P[:, 1], P[:, 2], s=2, alpha=0.3, color="#FF1493")
        self._set_3d_limits(ax1, P)
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(T[:, 0], T[:, 1], T[:, 2], s=2, alpha=0.3, color="#00BFFF")
        self._set_3d_limits(ax2, T)
        plt.savefig(os.path.join(self.sample_dir, f"data_preview_{step:07d}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def save_snapshot_row(self, x0, x1, snaps, traj_lines, t_list, step, colors_snap, colors_lines):
        fig = plt.figure(figsize=(3.5 * len(t_list), 3.5))
        all_pts = np.concatenate([x0, x1], axis=0)
        for i, t in enumerate(t_list):
            ax = fig.add_subplot(1, len(t_list), i+1, projection='3d')
            ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], s=1, alpha=0.02, color="#FF1493")
            ax.scatter(x1[:, 0], x1[:, 1], x1[:, 2], s=1, alpha=0.02, color="#00BFFF")
            idx = max(0, min(int(round(float(t) * (traj_lines.shape[1]-1))), traj_lines.shape[1]-1))
            if idx > 0:
                for j in range(traj_lines.shape[0]):
                    ax.plot(traj_lines[j, :idx+1, 0], traj_lines[j, :idx+1, 1], traj_lines[j, :idx+1, 2], 
                            color=colors_lines[j], alpha=0.25, lw=0.6)
            z_np = snaps[float(t)]
            ax.scatter(z_np[:, 0], z_np[:, 1], z_np[:, 2], s=3, alpha=0.8, c=colors_snap, edgecolors='none')
            ax.set_title(f"t={t:g}", fontsize=16, pad=-15)
            self._set_3d_limits(ax, all_pts)
        plt.savefig(os.path.join(self.sample_dir, f"snapshots_{step:07d}.png"), dpi=300, bbox_inches="tight")
        plt.close()

def get_gradient_colors(pts_np):
    z_norm = (pts_np[:, 2] - pts_np[:, 2].min()) / (pts_np[:, 2].max() - pts_np[:, 2].min() + 1e-8)
    return plt.cm.plasma(z_norm)

def save_all_samples(logger, model, ema, x0, x1, RUN, step_tag):
    ema.apply_shadow(model)
    x0_l, x0_s = sample_vis_subset(x0, RUN["traj_line_num"]), sample_vis_subset(x0, RUN["snap_num_samples"])
    c_l, c_s = get_gradient_colors(x0_l.cpu().numpy()), get_gradient_colors(x0_s.cpu().numpy())
    traj_lines = compute_trajectories(model, x0_l, omega=RUN["omega"], steps=RUN["traj_steps"])
    traj_snap = compute_trajectories(model, x0_s, omega=RUN["omega"], steps=RUN["traj_steps"])
    snaps = compute_snapshots_from_trajectories(traj_snap, RUN["snap_t_list"])
    logger.save_snapshot_row(x0[:2500].cpu().numpy(), x1[:2500].cpu().numpy(), snaps, traj_lines, RUN["snap_t_list"], step_tag, c_s, c_l)
    ema.restore(model)

def main():
    RUN = {
        "dataset": "Pyramid_to_Torus_3D", 
        "method": "Forward_MeanFlow_ZOH", 
        "seed": 42, "dim": 3,
        "n_data": 4096, 
        "batch_size": 512, 
        "lr": 2e-4, 
        "warmup_steps": 5000, 
        "shape_noise": 0.015,
        "width": 512, 
        "depth": 8, 
        "train_steps": 150000, 
        "omega": -np.pi / 0.75, #np.pi / 2, 
        "run_root": "runs_3d",
        "traj_line_num": 150, 
        "traj_steps": 64, 
        "snap_num_samples": 4096, 
        "snap_t_list": [0.0, 0.25, 0.5, 0.75, 1.0],
        "log_step": 200, 
        "sample_step": 5000, 
        "ckpt_step": 10000,
    }
    set_seed(RUN["seed"]); device = get_device()
    x0, x1 = build_3D_dataset(RUN, device)
    model = ResMLP3D(dim=3, width=RUN["width"], depth=RUN["depth"]).to(device)
    ema, opt = EMA(model), torch.optim.AdamW(model.parameters(), lr=RUN["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s/5000 if s<5000 else 0.5*(1+np.cos(np.pi*(s-5000)/(145000))))
    mf = MeanFlow(dim=3, omega=RUN["omega"]); mf.set_prior_data(x1)
    logger = RunLogger(RUN["run_root"], RUN["dataset"], RUN["method"], RUN)
    logger.save_data_preview(x0, x1, step=0)
    pbar = tqdm(range(RUN["train_steps"]))
    for step in pbar:
        idx = torch.randint(0, x0.shape[0], (RUN["batch_size"],), device=device)
        loss, mse = mf.loss(model, x0[idx], step=step)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); scheduler.step(); ema.update(model)
        if step % RUN["log_step"] == 0: logger.log_train(step, float(loss.item()), float(mse.item()), opt.param_groups[0]["lr"])
        if step > 0 and step % RUN["sample_step"] == 0: save_all_samples(logger, model, ema, x0, x1, RUN, step)
        if step > 0 and step % RUN["ckpt_step"] == 0: logger.save_checkpoint(step, model, ema, opt)

if __name__ == "__main__": main()