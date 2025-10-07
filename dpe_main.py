# dpe_main.py
import math, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# ---------- Time embedding + conditional denoiser ----------
class TimeMLP(nn.Module):
    def __init__(self, d_t=16, d_hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_t, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU()
        )
    def forward(self, t_emb):
        return self.net(t_emb)

class CondDenoiser(nn.Module):
    def __init__(self, dim_x, dim_p, d_hidden=256, d_t=16):
        super().__init__()
        # all three branches output d_hidden to make concat = 3*d_hidden
        self.tproj = TimeMLP(d_t=d_t, d_hidden=d_hidden)   # <-- fixed to 256
        self.embed_t = nn.Linear(1, d_t)

        self.xproj = nn.Sequential(nn.Linear(dim_x, d_hidden), nn.SiLU())
        self.pproj = nn.Sequential(nn.Linear(dim_p, d_hidden), nn.SiLU())

        self.core  = nn.Sequential(
            nn.Linear(d_hidden*3, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, d_hidden), nn.SiLU(),
            nn.Linear(d_hidden, dim_p)
        )

    def forward(self, x, p_noisy, t_scalar01):
        # t_scalar01: [B,1] in [0,1]
        t_emb = self.embed_t(t_scalar01)     # [B, d_t]
        h_t = self.tproj(t_emb)              # [B, d_hidden]
        h_x = self.xproj(x)                  # [B, d_hidden]
        h_p = self.pproj(p_noisy)            # [B, d_hidden]
        h = torch.cat([h_x, h_p, h_t], dim=-1)  # [B, 3*d_hidden]
        return self.core(h)                  # predicts noise ε_hat (shape = p)

# ---------- Diffusion schedule (cosine) ----------
def cosine_beta_schedule(T=200, s=0.008):
    steps = torch.arange(T+1, dtype=torch.float32)
    alphas_cumprod = torch.cos(((steps/T)+s)/(1+s) * math.pi/2)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)

class Diffusion1D:
    def __init__(self, T=200):
        self.T = T
        self.betas = cosine_beta_schedule(T).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def sample_t(self, B):
        return torch.randint(0, self.T, (B,), device=device)

    def q_sample(self, p0, t, noise=None):
        if noise is None: noise = torch.randn_like(p0)
        sqrt_ab = self.alphas_cumprod[t].sqrt().view(-1,1)
        sqrt_1m = (1 - self.alphas_cumprod[t]).sqrt().view(-1,1)
        return sqrt_ab*p0 + sqrt_1m*noise, noise

# ---------- Reward: context-shifted Rosenbrock (differentiable) ----------
# We optimize p ∈ R^2 to minimize Rosenbrock centered by context x ∈ R^2.
def rosenbrock_reward(x, p):
    # map context to parameters and a small shift
    a = 1.0 + 0.5*torch.tanh(x[:, :1])                 # ~[0.5,1.5]
    b = 100.0 * torch.exp(torch.tanh(x[:, 1:2]))       # ~[36,271]
    p_shift = p + 0.25 * torch.cat([x[:, :1], x[:, 1:2]], dim=-1)
    x1, x2 = p_shift[:, :1], p_shift[:, 1:2]
    val = (a - x1)**2 + b*(x2 - x1**2)**2              # smaller = better
    R = -val.squeeze(-1)                                # reward = -loss
    return R

# ---------- Synthetic dataset for warm-up ----------
def make_synthetic_dataset(N=4096):
    x = torch.randn(N, 2)
    a = 1.0 + 0.5*torch.tanh(x[:, :1])
    p_star = torch.cat([a, a**2], dim=-1) - 0.25*torch.cat([x[:, :1], x[:, 1:2]], dim=-1)
    p_star += 0.05*torch.randn_like(p_star)
    return x, p_star

# ---------- Sampler (with optional tiny MPPI reweighting) ----------
@torch.no_grad()
def ddpm_sample(model, diff, x, dim_p=2, mppi_K=0, lam=0.5):
    B = x.size(0)
    p = torch.randn(B, dim_p, device=device)
    for t in reversed(range(diff.T)):
        t01 = torch.full((B,1), float(t)/diff.T, device=device)
        eps_hat = model(x, p, t01)
        alpha = diff.alphas[t]; beta = diff.betas[t]; ab = diff.alphas_cumprod[t]
        mean = (1/math.sqrt(alpha))*(p - (beta/math.sqrt(1-ab))*eps_hat)

        if mppi_K > 0:
            cands, scores = [mean], [rosenbrock_reward(x, mean)]
            for _ in range(mppi_K):
                jitter = 0.05*torch.randn_like(mean)
                pj = mean + jitter
                cands.append(pj)
                scores.append(rosenbrock_reward(x, pj))
            S = torch.stack(scores, 0)                  # [K+1,B]
            W = torch.softmax(S/lam, dim=0).unsqueeze(-1)
            P = torch.stack(cands, 0)                   # [K+1,B,dim_p]
            mean = (W*P).sum(0)

        if t > 0:
            p = mean + math.sqrt(beta) * torch.randn_like(mean)
        else:
            p = mean
    return p

# ---------- Training ----------
def train(args):
    dim_x, dim_p = 2, 2
    model = CondDenoiser(dim_x, dim_p, d_hidden=256, d_t=16).to(device)
    diff = Diffusion1D(T=args.T)

    x_all, p_star = make_synthetic_dataset(N=args.N)
    ds = TensorDataset(x_all, p_star)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(args.epochs):
        for x, p0 in dl:
            x, p0 = x.to(device), p0.to(device)

            # diffusion forward (noising)
            t = diff.sample_t(x.size(0))
            p_t, noise = diff.q_sample(p0, t)
            t01 = (t.float()/diff.T).unsqueeze(-1)

            # denoising prediction
            eps_hat = model(x, p_t, t01)
            loss_denoise = F.mse_loss(eps_hat, noise)

            # reconstruct p0 from p_t and eps_hat
            ab = diff.alphas_cumprod[t].view(-1,1)
            p0_pred = (p_t - (1-ab).sqrt()*eps_hat) / ab.sqrt()

            # reward-guided steering (differentiable)
            R = rosenbrock_reward(x, p0_pred)
            loss_reward = - args.lambda_R * R.mean()

            loss = loss_denoise + loss_reward
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # simple validation / logging
        with torch.no_grad():
            x_val = torch.randn(256, dim_x, device=device)
            p_gen = ddpm_sample(model, diff, x_val, dim_p, mppi_K=args.mppi_K)
            R_val = rosenbrock_reward(x_val, p_gen).mean().item()
        print(f"[epoch {epoch+1:03d}] loss={loss.item():.4f}  R_val≈{R_val:.3f}")

    # final small demo
    with torch.no_grad():
        x_demo = torch.tensor([[0.0, 0.0],
                               [1.0,-1.0],
                               [-0.5,0.5]], device=device, dtype=torch.float32)
        p_out = ddpm_sample(model, diff, x_demo, dim_p, mppi_K=args.mppi_K)
        R_out = rosenbrock_reward(x_demo, p_out)
        print("\nDemo contexts x:\n", x_demo.cpu().numpy())
        print("Generated p:\n", p_out.cpu().numpy())
        print("Rewards:\n", R_out.cpu().numpy())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--lambda_R", type=float, default=0.05)
    ap.add_argument("--mppi_K", type=int, default=4, help="0=off; >0 enables tiny MPPI at inference")
    ap.add_argument("--log_every", type=int, default=1)
    args = ap.parse_args()
    train(args)
