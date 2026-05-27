"""
SSM / SHRED / LSSL / S4D model definitions and training utilities.

Models:
    - SHREDSeq2Seq   : GRU/LSTM encoder + shallow MLP decoder
    - LSSLStack       : Linear State-Space Layer stack (HiPPO-LegS, bilinear) + optional shallow MLP decoder
    - SensorToPODS4D  : Paper-aligned S4D (NPLR + Woodbury + FFT convolution)

Training:
    - train_model     : generic training loop with early stopping
    - predict         : batched inference returning numpy arrays
    - compute_metrics : MAE / MSE / MRE summary dict (numpy)
    - mae, mse, mre   : element-wise metric functions (numpy + torch)
"""
from __future__ import annotations

import math
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ════════════════════════════════════════════════════════════════
#  Metric helpers (work with both torch tensors and numpy arrays)
# ════════════════════════════════════════════════════════════════

def _is_torch(x):
    return isinstance(x, torch.Tensor)


def mae(datatrue, datapred):
    diff = datatrue - datapred
    if _is_torch(diff):
        return diff.abs().mean()
    return np.abs(diff).mean()


def mse(datatrue, datapred):
    diff = datatrue - datapred
    if _is_torch(diff):
        return diff.pow(2).sum(dim=-1).mean()
    return np.square(diff).sum(axis=-1).mean()


def mre(datatrue, datapred, eps=1e-12):
    diff = datatrue - datapred
    if _is_torch(diff):
        num = diff.pow(2).sum(dim=-1).sqrt()
        den = datatrue.pow(2).sum(dim=-1).sqrt().clamp_min(eps)
        return (num / den).mean()
    num = np.sqrt(np.square(diff).sum(axis=-1))
    den = np.sqrt(np.square(datatrue).sum(axis=-1))
    den = np.clip(den, eps, None)
    return (num / den).mean()


# ════════════════════════════════════════════════════════════════
#  LSSL  (Linear State-Space Layer — HiPPO-LegS, bilinear disc.)
# ════════════════════════════════════════════════════════════════

def _lssl_hippo_diag(N: int, device=None):
    cpu = torch.device("cpu")
    n = torch.arange(N, dtype=torch.float32, device=cpu)
    r = torch.sqrt(2.0 * n + 1.0)
    A = torch.zeros(N, N, dtype=torch.float32, device=cpu)
    ii, jj = torch.meshgrid(n.long(), n.long(), indexing="ij")
    A[ii > jj] = -(r[ii[ii > jj]] * r[jj[ii > jj]])
    A.diagonal().copy_(-(n + 1.0))
    p = torch.sqrt(n + 0.5)
    S = A + torch.outer(p, p)
    lam, _ = torch.linalg.eig(S)
    return lam[lam.imag.argsort()].to(device)


class LSSLKernel(nn.Module):
    def __init__(
        self,
        d_model,
        state_dim,
        l_max,
        dt_min=1e-3,
        dt_max=1e-1,
        device=None,
        a_init="diagonal",
        learn_cd=False,
    ):
        super().__init__()
        self.a_init = a_init
        self.learn_cd = learn_cd
        if a_init == "nplr":
            lam, P0, _ = make_nplr_legs(state_dim, device=device)
            neg_re = -lam.real
            theta_init = torch.log(torch.expm1(neg_re))
            self.Lambda_re = nn.Parameter(theta_init.unsqueeze(0).repeat(d_model, 1))
            self.Lambda_im = nn.Parameter(lam.imag.unsqueeze(0).repeat(d_model, 1))
            self.P_re = nn.Parameter(P0.real.unsqueeze(0).repeat(d_model, 1))
            self.P_im = nn.Parameter(P0.imag.unsqueeze(0).repeat(d_model, 1))
        else:  # "diagonal"
            lam = _lssl_hippo_diag(state_dim, device=device)
            # lam.real is negative (HiPPO eigenvalues are stable).
            # We want -softplus(theta) = lam.real, i.e. softplus(theta) = -lam.real.
            # softplus^{-1}(y) = log(exp(y) - 1) = log(expm1(y)) for y > 0.
            neg_re = -lam.real  # positive
            theta_init = torch.log(torch.expm1(neg_re))  # inverse softplus
            self.Lambda_re = nn.Parameter(
                theta_init.unsqueeze(0).repeat(d_model, 1)
            )
            self.Lambda_im = nn.Parameter(
                lam.imag.unsqueeze(0).repeat(d_model, 1)
            )
        scale = 1.0 / math.sqrt(state_dim)
        self.B_re = nn.Parameter(torch.randn(d_model, state_dim, device=device) * scale)
        self.B_im = nn.Parameter(torch.zeros(d_model, state_dim, device=device))
        if learn_cd:
            self.C_re = nn.Parameter(torch.randn(d_model, state_dim, device=device) * scale)
            self.C_im = nn.Parameter(torch.zeros(d_model, state_dim, device=device))
            self.D = nn.Parameter(torch.ones(d_model, device=device))
        else:
            # Lean SSM: tie C to B and remove direct feedthrough term D.
            self.register_buffer("D", torch.zeros(d_model, device=device))
        self.log_dt = nn.Parameter(
            torch.empty(d_model, device=device).uniform_(math.log(dt_min), math.log(dt_max))
        )

    def _bilinear(self):
        lam = torch.complex(-F.softplus(self.Lambda_re), self.Lambda_im)
        B = torch.complex(self.B_re, self.B_im)
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        half = 0.5 * dt * lam
        denom = 1.0 - half
        a_bar = (1.0 + half) / denom
        b_bar = (dt / denom) * B
        return a_bar, b_bar

    def kernel(self, L):
        if self.a_init == "nplr":
            return self._kernel_nplr(L)
        a_bar, b_bar = self._bilinear()
        if self.learn_cd:
            C = torch.complex(self.C_re, self.C_im)
        else:
            C = torch.complex(self.B_re, self.B_im)
        k = torch.arange(L, device=a_bar.device)
        a_pow = torch.exp(torch.log(a_bar).unsqueeze(-1) * k.to(a_bar.dtype))
        return ((C * b_bar).unsqueeze(-1) * a_pow).sum(dim=1).real

    def _kernel_nplr(self, L):
        """NPLR kernel via Woodbury+Cauchy evaluated at roots of unity, then IFFT."""
        Lambda = torch.complex(-F.softplus(self.Lambda_re), self.Lambda_im)  # (d, N)
        P = torch.complex(self.P_re, self.P_im)                              # (d, N)
        B = torch.complex(self.B_re, self.B_im)                              # (d, N)
        dt = torch.exp(self.log_dt)                                          # (d,)

        # Roots of unity for IFFT recovery of the kernel
        k = torch.arange(L, device=Lambda.device)
        z = torch.exp(-2j * math.pi * k / L)  # (L,) — z_k = e^{-2πi k/L}

        # Bilinear transform: map evaluation points from z-domain to s-domain
        # g = (2/dt) * (1-z)/(1+z)  — Möbius / bilinear change of variable
        eps = torch.finfo(dt.dtype).eps
        denom_z = 1.0 + z  # (L,)
        denom_z = torch.where(
            denom_z.abs() < eps,
            torch.complex(torch.full_like(denom_z.real, eps), torch.zeros_like(denom_z.real)),
            denom_z,
        )
        g = (2.0 / dt[:, None]) * ((1.0 - z[None, :]) / denom_z[None, :])  # (d, L)

        # Cauchy resolvent:  R_{d,l,n} = 1 / (g_{d,l} - Lambda_{d,n})
        R = 1.0 / (g[:, :, None] - Lambda[:, None, :])  # (d, L, N)

        # Rank-1 Woodbury correction terms  (d, L)
        PRB = (P.conj()[:, None, :] * R * B[:, None, :]).sum(dim=-1)
        PRP = (P.conj()[:, None, :] * R * P[:, None, :]).sum(dim=-1)

        if self.learn_cd:
            C = torch.complex(self.C_re, self.C_im)                          # (d, N)
            CRB = (C.conj()[:, None, :] * R * B[:, None, :]).sum(dim=-1)
            CRP = (C.conj()[:, None, :] * R * P[:, None, :]).sum(dim=-1)
            K_hat_core = CRB - CRP * PRB / (1.0 + PRP)
        else:
            # Lean SSM: C is tied to B, so remove explicit C algebra.
            BRB = (B.conj()[:, None, :] * R * B[:, None, :]).sum(dim=-1)
            BRP = (B.conj()[:, None, :] * R * P[:, None, :]).sum(dim=-1)
            K_hat_core = BRB - BRP * PRB / (1.0 + PRP)

        # Full NPLR kernel in frequency domain
        K_hat = (2.0 / denom_z)[None, :] * K_hat_core  # (d, L)
        return torch.fft.ifft(K_hat, n=L, dim=-1).real  # (d, L)

    def forward(self, L):
        return self.kernel(L), self.D


class LSSLLayer(nn.Module):
    def __init__(
        self,
        d_model,
        state_dim=64,
        l_max=2048,
        dt_min=1e-3,
        dt_max=1e-1,
        dropout=0.0,
        prenorm=True,
        device=None,
        activation=nn.GLU,
        a_init="diagonal",
        learn_cd=False,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.norm = nn.LayerNorm(d_model)
        self.kernel = LSSLKernel(
            d_model,
            state_dim,
            l_max,
            dt_min,
            dt_max,
            device,
            a_init=a_init,
            learn_cd=learn_cd,
        )
        self.dropout = nn.Dropout(dropout)
        # Accept both a class (nn.Tanh) and an instance (nn.Tanh()) for activation
        act_cls = activation if isinstance(activation, type) else type(activation)
        if act_cls is nn.GLU:
            self.output_linear = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.GLU(dim=-1),
            )
        else:
            self.output_linear = nn.Sequential(
                nn.Linear(d_model, d_model),
                act_cls(),
            )

    def _fft_conv(self, u, k):
        L = u.shape[-1]
        n = 1 << (2 * L - 1).bit_length()  # next power of 2 >= 2*L
        uf = torch.fft.rfft(u.contiguous(), n=n, dim=-1)
        kf = torch.fft.rfft(k.contiguous(), n=n, dim=-1)
        return torch.fft.irfft(uf * kf.unsqueeze(0), n=n, dim=-1)[..., :L]

    def forward(self, x):
        residual = x
        if self.prenorm:
            x = self.norm(x)
        _, L, _ = x.shape
        K, D = self.kernel(L)
        u = x.transpose(1, 2)
        y = self._fft_conv(u, K) + u * D.unsqueeze(-1)
        y = y.transpose(1, 2)
        y = self.dropout(y)
        y = self.output_linear(y)
        
        y = y + residual
        if not self.prenorm:
            y = self.norm(y)
        return y


class LSSLStack(nn.Module):
    """Linear State-Space Layer stack with an optional shallow MLP decoder.

    Parameters
    ----------
    decoder_sizes : list[int] | None
        Hidden layer widths for the MLP decoder, e.g. ``[128, 256]``.
        ``None`` (default) uses a single linear projection ``d_model → d_output``,
        preserving backward-compatibility with existing checkpoints.
    decoder_act : nn.Module | None
        Activation used between decoder layers.  Defaults to ``nn.SiLU()``.
    learn_cd : bool
        If ``True``, learn SSM output/readout ``C`` and skip ``D`` terms.
        If ``False`` (default), use a lean SSM with ``C=B`` and fixed ``D=0``
        to reduce parameter count when a shallow decoder is used.
    """

    def __init__(
        self,
        d_input,
        d_model,
        d_output,
        n_layers,
        state_dim=64,
        l_max=2048,
        dt_min=1e-3,
        dt_max=1e-1,
        dropout=0.0,
        prenorm=True,
        device=None,
        activation=nn.SiLU,
        a_init="diagonal",
        decoder_sizes=[350, 500],
        decoder_act=None,
        learn_cd=False,
    ):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([
            LSSLLayer(d_model, state_dim, l_max, dt_min, dt_max, dropout,
                      prenorm, device, activation, a_init=a_init, learn_cd=learn_cd)
            for _ in range(n_layers)
        ])

        # ── Decoder: linear (default) or shallow MLP ─────────────────────────
        if decoder_act is None:
            decoder_act = nn.SiLU()

        if decoder_sizes:
            sizes = [d_model] + list(decoder_sizes) + [d_output]
            self.decoder = nn.ModuleList()
            for i in range(len(sizes) - 1):
                self.decoder.append(nn.Linear(sizes[i], sizes[i + 1]))
                if i < len(sizes) - 2:           # no activation after last layer
                    self.decoder.append(nn.Dropout(dropout))
                    self.decoder.append(type(decoder_act)())
        else:
            # Single linear layer kept as ModuleList for a uniform _decode path
            self.decoder = nn.ModuleList([nn.Linear(d_model, d_output)])

    def _decode(self, x):
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        return self._decode(x)


# ════════════════════════════════════════════════════════════════
#  S4D  (Paper-aligned: NPLR + Woodbury + bilinear + FFT)
# ════════════════════════════════════════════════════════════════

def _as_complex(re: torch.Tensor, im: torch.Tensor) -> torch.Tensor:
    return torch.complex(re, im)


def _conj(x: torch.Tensor) -> torch.Tensor:
    return torch.conj(x)


def hippo_legs_matrix(N: int, device=None, dtype=torch.float32):
    """HiPPO-LegS matrix A and rank-1 correction vector p."""
    n = torch.arange(N, device=device, dtype=dtype)
    r = torch.sqrt(2.0 * n + 1.0)

    A = torch.zeros(N, N, device=device, dtype=dtype)
    i = n[:, None]
    j = n[None, :]

    lower = i > j
    A[lower] = -(r[:, None] * r[None, :])[lower]
    A[torch.arange(N), torch.arange(N)] = -(n + 1.0)

    p = torch.sqrt(n + 0.5)
    return A, p


def make_nplr_legs(N: int, device=None, dtype=torch.float32):
    """NPLR parameterization from HiPPO-LegS (diagonalisation at init)."""
    cpu = torch.device("cpu")
    A, p = hippo_legs_matrix(N, device=cpu, dtype=dtype)

    S = A + torch.outer(p, p)
    Lambda, V = torch.linalg.eig(S)

    B = torch.sqrt(2.0 * torch.arange(N, dtype=dtype) + 1.0)
    Vh = V.conj().transpose(-1, -2)
    P = Vh @ p.to(torch.cfloat)
    B = Vh @ B.to(torch.cfloat)

    return (
        Lambda.to(device=device),
        P.to(device=device),
        B.to(device=device),
    )

def train_model(
    model: nn.Module,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_vl: np.ndarray,
    Y_vl: np.ndarray,
    *,
    device: torch.device | str = "cpu",
    loss_fun=None,
    epochs: int = 400,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 20,
    lr_patience: int = 10,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    label: str = "model",
):
    """Generic training loop with early stopping and LR scheduling."""
    if loss_fun is None:
        loss_fun = mse

    model = model.to(device)
    start = time.time()

    train_dl = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(Y_tr, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_dl = DataLoader(
        TensorDataset(
            torch.tensor(X_vl, dtype=torch.float32),
            torch.tensor(Y_vl, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=lr_patience, factor=0.5)

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    wait = 0
    tr_losses, vl_losses = [], []
    epoch_times = []

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()
        tr_sum = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fun(pred, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_sum += loss.item()

        tr_loss = tr_sum / max(len(train_dl), 1)

        model.eval()
        vl_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                vl_sum += loss_fun(model(xb), yb).item()
        vl_loss = vl_sum / max(len(val_dl), 1)

        epoch_times.append(time.time() - t_epoch_start)
        tr_losses.append(tr_loss)
        vl_losses.append(vl_loss)
        sched.step(vl_loss)

        if vl_loss < best_val:
            best_val = vl_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        print(
            f"[{label}] epoch={epoch:4d}/{epochs}  train={tr_loss:.4e}  "
            f"val={vl_loss:.4e}  best={best_val:.4e}  "
            f"lr={opt.param_groups[0]['lr']:.1e}  wait={wait}/{patience}",
            end="\r",
        )

        if wait >= patience:
            print(f"\n[{label}] Early stop at epoch {epoch}")
            break

    elapsed = time.time() - start
    ep = np.array(epoch_times)
    print(
        f"\n[{label}] Training completed in {elapsed:.2f}s  "
        f"({len(ep)} epochs)  "
        f"per-epoch: mean={ep.mean():.3f}s  "
        f"std={ep.std():.3f}s  "
        f"min={ep.min():.3f}s  "
        f"max={ep.max():.3f}s"
    )
    model.load_state_dict(best_state)
    model.to(device)
    return tr_losses, vl_losses


def predict(
    model: nn.Module,
    X: np.ndarray,
    *,
    device: torch.device | str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Batched model inference, returns numpy array."""
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), batch_size):
            out.append(model(Xt[i : i + batch_size]).cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, MSE, MRE (%) as formatted strings."""
    _mae = np.mean(np.abs(y_true - y_pred))
    _mse = np.mean((y_true - y_pred) ** 2)
    num = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=-1))
    den = np.sqrt(np.sum(y_true ** 2, axis=-1)) + 1e-12
    _mre = np.mean(num / den)
    return {"mae": _mae, "mse": _mse, "mre": 100 * _mre}