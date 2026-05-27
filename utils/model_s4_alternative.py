"""
S4D / SHRED model definitions and training utilities.

Models:
    - S4DStack        : Diagonal State-Space (S4D) layer stack
                        with HiPPO-LegS or S4D-Lin initialisation,
                        bilinear discretisation, and FFT convolution.
                        Includes an optional shallow MLP decoder.

Training:
    - train_model     : generic training loop with early stopping
    - predict         : batched inference returning numpy arrays
    - compute_metrics : MAE / MSE / MRE summary dict (numpy)
    - mae, mse, mre   : element-wise metric functions (numpy + torch)
"""
from __future__ import annotations

import math
import time

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
#  S4D — Diagonal State-Space Layer (canonical S4D)
#  Two initialisations supported:
#    • "legs": HiPPO-LegS diagonal,  λ_j = -(j + 1/2)
#    • "lin":  S4D-Lin,              λ_j = -1/2 + i π j
#  Both are closed-form; no matrix diagonalisation is required.
# ════════════════════════════════════════════════════════════════

def _s4d_init(N: int, kind: str = "lin", device=None) -> torch.Tensor:
    """Closed-form S4D eigenvalue initialisations.

    Parameters
    ----------
    N : int
        State dimension per channel.
    kind : {"lin", "legs"}
        - "lin"  : λ_j = -1/2 + i π j   (S4D-Lin, Gu et al. 2022)
        - "legs" : λ_j = -(j + 1/2)     (S4D-LegS diagonal)
    device : torch.device | str | None
        Target device. Eigenvalues are constructed on CPU, then moved.

    Returns
    -------
    lam : (N,) complex tensor on `device`, sorted by Im(λ).
    """
    n = torch.arange(N, dtype=torch.float32)
    if kind == "lin":
        lam_re = -0.5 * torch.ones(N)
        lam_im = math.pi * n
    elif kind == "legs":
        lam_re = -(n + 0.5)
        lam_im = torch.zeros(N)
    else:
        raise ValueError(f"Unknown S4D init: {kind!r}. Use 'lin' or 'legs'.")
    lam = torch.complex(lam_re, lam_im)
    return lam[lam.imag.argsort()].to(device) if device is not None else lam


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse softplus: y + log(-expm1(-y)) for y > 0."""
    return y + torch.log(-torch.expm1(-y))


class S4DKernel(nn.Module):
    """Diagonal S4D kernel with bilinear (Tustin) discretisation.

    The continuous-time dynamics  ẋ = Λ x + B u  are discretised with
    the bilinear transform (α = 1/2), giving discrete (Ā, B̄). The state
    matrix is parameterised in pre-softplus form,
        Λ_re = -softplus(θ),
    so that Re(λ) < 0 holds by construction throughout training. θ is
    initialised to the inverse softplus of the canonical S4D-Lin or
    S4D-LegS spectrum, so the effective Λ at step 0 matches the paper.
    """

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        device=None,
        init: str = "lin",
        learn_cd: bool = False,
    ):
        super().__init__()
        self.learn_cd = learn_cd

        # ── Eigenvalue init ──────────────────────────────────────────
        # Constructed on CPU; moved to device. Stored as inverse-softplus
        # pre-activations so -softplus(Λ_re) == Re(λ_init) exactly at t=0.
        lam = _s4d_init(state_dim, kind=init)               # (N,) complex (cpu)
        neg_re = -lam.real                                   # (N,) positive
        neg_re = neg_re.clamp_min(1e-6)                      # guard against zero
        theta_init = _inv_softplus(neg_re)                   # (N,) real

        Lambda_re = theta_init.unsqueeze(0).repeat(d_model, 1)
        Lambda_im = lam.imag.unsqueeze(0).repeat(d_model, 1)
        self.Lambda_re = nn.Parameter(Lambda_re.to(device) if device else Lambda_re)
        self.Lambda_im = nn.Parameter(Lambda_im.to(device) if device else Lambda_im)

        # ── B initialisation ────────────────────────────────────────
        # Canonical S4D: B = 1 (all ones) in the diagonal basis. With C
        # tied to B (lean mode) this is initialised at ones and learned;
        # with C learned independently, B = 1 is optimal up to a constant.
        B_re = torch.ones(d_model, state_dim)
        B_im = torch.zeros(d_model, state_dim)
        self.B_re = nn.Parameter(B_re.to(device) if device else B_re)
        self.B_im = nn.Parameter(B_im.to(device) if device else B_im)

        # ── C (and D) ───────────────────────────────────────────────
        if learn_cd:
            scale = 1.0 / math.sqrt(state_dim)
            C_re = torch.randn(d_model, state_dim) * scale
            C_im = torch.randn(d_model, state_dim) * scale
            self.C_re = nn.Parameter(C_re.to(device) if device else C_re)
            self.C_im = nn.Parameter(C_im.to(device) if device else C_im)
            D = torch.ones(d_model)
            self.D = nn.Parameter(D.to(device) if device else D)
        else:
            # Lean SSM: tie C = B, fix D = 0.
            D_buf = torch.zeros(d_model)
            self.register_buffer("D", D_buf.to(device) if device else D_buf)

        # ── Per-channel learnable Δt ─────────────────────────────────
        log_dt = torch.empty(d_model).uniform_(math.log(dt_min), math.log(dt_max))
        self.log_dt = nn.Parameter(log_dt.to(device) if device else log_dt)

    def _bilinear(self):
        """Apply bilinear (Tustin) transform to obtain discrete Ā, B̄."""
        lam = torch.complex(-F.softplus(self.Lambda_re), self.Lambda_im)  # (d, N)
        B = torch.complex(self.B_re, self.B_im)                            # (d, N)
        dt = torch.exp(self.log_dt).unsqueeze(-1)                          # (d, 1)
        half = 0.5 * dt * lam
        denom = 1.0 - half
        a_bar = (1.0 + half) / denom
        b_bar = (dt / denom) * B
        return a_bar, b_bar

    def kernel(self, L: int) -> torch.Tensor:
        """Return the convolution kernel K of shape (d_model, L)."""
        a_bar, b_bar = self._bilinear()                                    # (d, N)
        if self.learn_cd:
            C = torch.complex(self.C_re, self.C_im)
        else:
            C = torch.complex(self.B_re, self.B_im)                        # tied: C = B
        k = torch.arange(L, device=a_bar.device, dtype=a_bar.real.dtype)
        # a_bar^k via complex exponentiation: exp(k · log a_bar)
        a_pow = torch.exp(torch.log(a_bar).unsqueeze(-1) * k)               # (d, N, L)
        return ((C * b_bar).unsqueeze(-1) * a_pow).sum(dim=1).real          # (d, L)

    def forward(self, L: int):
        return self.kernel(L), self.D


class S4DLayer(nn.Module):
    """One S4D block: pre-LayerNorm → kernel → FFT conv → dropout → linear+SiLU → residual."""

    def __init__(
        self,
        d_model: int,
        state_dim: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dropout: float = 0.0,
        prenorm: bool = True,
        device=None,
        activation=nn.SiLU,
        init: str = "lin",
        learn_cd: bool = False,
    ):
        super().__init__()
        self.prenorm = prenorm
        self.norm = nn.LayerNorm(d_model)
        self.kernel = S4DKernel(
            d_model, state_dim, dt_min, dt_max, device,
            init=init, learn_cd=learn_cd,
        )
        self.dropout = nn.Dropout(dropout)
        act_cls = activation if isinstance(activation, type) else type(activation)
        self.output_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            act_cls(),
        )

    def _fft_conv(self, u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Causal convolution of u (B, d, L) with kernel k (d, L) via FFT."""
        L = u.shape[-1]
        n = 1 << (2 * L - 1).bit_length()       # next pow2 >= 2L (no wrap-around)
        uf = torch.fft.rfft(u.contiguous(), n=n, dim=-1)
        kf = torch.fft.rfft(k.contiguous(), n=n, dim=-1)
        return torch.fft.irfft(uf * kf.unsqueeze(0), n=n, dim=-1)[..., :L]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        if self.prenorm:
            x = self.norm(x)
        _, L, _ = x.shape
        K, D = self.kernel(L)
        u = x.transpose(1, 2)                              # (B, d, L)
        y = self._fft_conv(u, K) + u * D.unsqueeze(-1)
        y = y.transpose(1, 2)                              # (B, L, d)
        y = self.dropout(y)
        y = self.output_linear(y)
        y = y + residual
        if not self.prenorm:
            y = self.norm(y)
        return y


class S4DStack(nn.Module):
    """Encoder → stack of S4D layers → shallow MLP decoder.

    Parameters
    ----------
    d_input, d_model, d_output : int
        Input, latent, and output channel dimensions.
    n_layers : int
        Number of stacked S4D blocks.
    state_dim : int
        State dimension N of each diagonal SSM per channel.
    dt_min, dt_max : float
        Bounds for the log-uniform initialisation of Δt.
    dropout : float
        Dropout probability inside each block and the decoder.
    prenorm : bool
        If True, LayerNorm is applied before the kernel (pre-norm).
    activation : nn.Module class or instance
        Activation used in the post-kernel linear projection.
    init : {"lin", "legs"}
        Eigenvalue initialisation. "lin" gives oscillatory eigenvalues
        (multi-frequency inductive bias); "legs" gives purely real
        eigenvalues (pure decay, no oscillation at init).
    decoder_sizes : list[int] | None
        Hidden layer widths for the MLP decoder, e.g. [350, 400].
        None uses a single linear projection d_model → d_output.
    decoder_act : nn.Module instance | None
        Activation used between decoder layers. Default: nn.SiLU().
    learn_cd : bool
        If True, learn SSM output/readout C and feedthrough D.
        If False (default), use a lean SSM with C = B and D = 0.
    """

    def __init__(
        self,
        d_input: int,
        d_model: int,
        d_output: int,
        n_layers: int,
        state_dim: int = 64,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dropout: float = 0.0,
        prenorm: bool = True,
        device=None,
        activation=nn.SiLU,
        init: str = "lin",
        decoder_sizes=None,
        decoder_act=None,
        learn_cd: bool = False,
    ):
        super().__init__()
        self.encoder = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([
            S4DLayer(d_model, state_dim, dt_min, dt_max, dropout,
                     prenorm, device, activation,
                     init=init, learn_cd=learn_cd)
            for _ in range(n_layers)
        ])

        if decoder_act is None:
            decoder_act = nn.SiLU()

        if decoder_sizes:
            sizes = [d_model] + list(decoder_sizes) + [d_output]
            self.decoder = nn.ModuleList()
            for i in range(len(sizes) - 1):
                self.decoder.append(nn.Linear(sizes[i], sizes[i + 1]))
                if i < len(sizes) - 2:                       # no act after last layer
                    self.decoder.append(nn.Dropout(dropout))
                    self.decoder.append(type(decoder_act)())
        else:
            self.decoder = nn.ModuleList([nn.Linear(d_model, d_output)])

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        return self._decode(x)


# ════════════════════════════════════════════════════════════════
#  Training / inference utilities
# ════════════════════════════════════════════════════════════════

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
    """Generic training loop with early stopping and LR scheduling.

    Compatible with MPS, CUDA, and CPU devices. Model parameters are
    moved to `device` before training begins; the returned `best_state`
    is also restored on `device`.
    """
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
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=lr_patience, factor=0.5
    )

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
    return {"mae": f"{_mae:.4e}", "mse": f"{_mse:.4e}", "mre": f"{100 * _mre:.2f}%"}