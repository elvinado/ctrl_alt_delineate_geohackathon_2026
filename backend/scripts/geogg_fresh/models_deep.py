"""Deep challengers for stage 07: models that read the seismic PATCH itself (PyTorch).

Both models are multi-task: one shared network, three heads.

* ``lithology`` - logit of P(sand) (binary cross-entropy, masked where the label is missing,
  ``pos_weight`` = shale/sand ratio of the training rows);
* ``vsh`` and ``phit`` - BOUNDED regression heads: sigmoid x upper bound (1.0 and 0.476), so a
  prediction can never leave its physical range (masked mean-squared error, each target
  divided by its training standard deviation).

Models
------
tcn              temporal CNN on the centre 3 x 3 traces (averaged) of the five stacks:
                 input [5 stacks, 21 samples]; dilated 1-D convolutions (receptive field
                 covers the 80-ms window); features = centre sample + window mean.
cnn3d_multitask  3-D CNN on the full patch [5 stacks, 5 IL, 5 XL, 21 samples]; two 3x3x5
                 convolutions; features = centre column (+/-2 samples) + global mean.

Amplitude scaling (fitted on the training wells only): ``common`` divides every stack by ONE
median / IQR pooled over the five stacks, so relative near/far amplitude (AVO behaviour) is
kept; ``per_channel`` scales each stack separately (an ablation).

Training: Adam (lr 1e-3, weight decay 1e-4), mini-batches, early stopping on the inner-
validation wells; the refit on all training wells of a fold uses the median best epoch.
PyTorch is optional: :func:`available` says whether these models can run.

References: Bai et al. 2018 (temporal convolutional networks); Das & Mukerji 2020 and Biswas
et al. 2019 (CNNs from seismic to petrophysical properties); Caruana 1997 (multi-task learning).
"""

from __future__ import annotations

import copy
import time

import numpy as np

from .metrics import PHI_MAX

HEADS = ("lithology", "vsh", "phit")
UPPER = {"vsh": 1.0, "phit": PHI_MAX}


def available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False, "PyTorch is not installed in this environment"
    return True, ""


def device_name(pref: str = "auto") -> str:
    import torch

    if pref == "cpu":
        return "cpu"
    if pref == "cuda" or (pref == "auto" and torch.cuda.is_available()):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


# =============================================================================== data scaling (numpy)
def fit_patch_scaling(P: np.ndarray, mode: str = "common") -> dict:
    """Centre/scale of stack patches P [N, C, IL, XL, T] (training rows only)."""
    out = {"mode": mode}
    if mode == "common":
        v = P[np.isfinite(P)]
        med = float(np.median(v)) if v.size else 0.0
        q25, q75 = (np.quantile(v, [0.25, 0.75]) if v.size else (0.0, 1.0))
        out["centre"] = [med] * P.shape[1]
        out["scale"] = [float(q75 - q25) or 1.0] * P.shape[1]
    else:
        out["centre"], out["scale"] = [], []
        for c in range(P.shape[1]):
            v = P[:, c][np.isfinite(P[:, c])]
            q25, q50, q75 = (np.quantile(v, [0.25, 0.5, 0.75]) if v.size else (0.0, 0.0, 1.0))
            out["centre"].append(float(q50))
            out["scale"].append(float(q75 - q25) or 1.0)
    return out


def apply_patch_scaling(P: np.ndarray, sc: dict, clip: float = 10.0) -> np.ndarray:
    """(P - centre) / scale per channel, NaN -> 0 (= the training median), clipped to +/- clip."""
    c = np.asarray(sc["centre"], np.float32)[None, :, None, None, None]
    s = np.asarray(sc["scale"], np.float32)[None, :, None, None, None]
    X = (np.asarray(P, np.float32) - c) / s
    X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
    return np.clip(X, -clip, clip)


def make_targets(df) -> dict[str, np.ndarray]:
    """Targets and masks for the heads (NaN-free arrays; masks say which entries count)."""
    lith = df["y_lithology"].to_numpy(float)
    out = {"lithology": np.where(lith >= 0, lith, 0.0).astype(np.float32),
           "m_lithology": (lith >= 0)}
    for h in ("vsh", "phit"):
        y = df[f"y_{h}"].to_numpy(float)
        m = df[f"mask_{h}"].to_numpy(bool) & np.isfinite(y)
        out[h] = np.where(m, y, 0.0).astype(np.float32)
        out[f"m_{h}"] = m
    return out


# =============================================================================== networks
def _networks():
    import torch
    from torch import nn

    class MultiHead(nn.Module):
        def __init__(self, d_in: int, dropout: float):
            super().__init__()
            self.trunk = nn.Sequential(nn.Linear(d_in, 32), nn.ReLU(), nn.Dropout(dropout))
            self.lith = nn.Linear(32, 1)
            self.reg = nn.Linear(32, 2)
            self.register_buffer("upper", torch.tensor([UPPER["vsh"], UPPER["phit"]], dtype=torch.float32))

        def forward(self, f):
            z = self.trunk(f)
            return self.lith(z).squeeze(-1), torch.sigmoid(self.reg(z)) * self.upper

    class TCN(nn.Module):
        def __init__(self, c_in: int, width: int = 32, dropout: float = 0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c_in, width, 5, padding=2), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(width, width, 5, padding=4, dilation=2), nn.BatchNorm1d(width), nn.ReLU(),
                nn.Conv1d(width, width, 3, padding=4, dilation=4), nn.BatchNorm1d(width), nn.ReLU())
            self.head = MultiHead(2 * width, dropout)

        def forward(self, x):                                  # x [B, C, IL, XL, T]
            ci, cj = x.shape[2] // 2, x.shape[3] // 2
            seq = x[:, :, max(0, ci - 1):ci + 2, max(0, cj - 1):cj + 2, :].mean(dim=(2, 3))   # [B, C, T]
            h = self.net(seq)                                  # [B, W, T]
            ck = h.shape[-1] // 2
            return self.head(torch.cat([h[:, :, ck], h.mean(dim=-1)], dim=1))

    class CNN3D(nn.Module):
        def __init__(self, c_in: int, dropout: float = 0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(c_in, 16, (3, 3, 5), padding=(1, 1, 2)), nn.BatchNorm3d(16), nn.ReLU(),
                nn.Dropout3d(dropout),
                nn.Conv3d(16, 32, (3, 3, 5), padding=(1, 1, 2)), nn.BatchNorm3d(32), nn.ReLU())
            self.head = MultiHead(64, dropout)

        def forward(self, x):                                  # x [B, C, IL, XL, T]
            h = self.net(x)                                    # [B, 32, IL, XL, T]
            ci, cj, ck = h.shape[2] // 2, h.shape[3] // 2, h.shape[4] // 2
            centre = h[:, :, ci, cj, max(0, ck - 2):ck + 3].mean(dim=-1)
            return self.head(torch.cat([centre, h.mean(dim=(2, 3, 4))], dim=1))

    return {"tcn": TCN, "cnn3d_multitask": CNN3D}


def build(name: str, c_in: int, dropout: float = 0.2):
    nets = _networks()
    if name not in nets:
        raise ValueError(f"unknown deep model {name}")
    return nets[name](c_in, dropout=dropout)


def describe(model) -> dict:
    """Architecture text and parameter count (saved next to the weights)."""
    n = sum(p.numel() for p in model.parameters())
    return {"n_parameters": int(n), "architecture": str(model)}


# =============================================================================== training
def _loss(out, batch, pos_weight, reg_scale):
    import torch.nn.functional as F

    logit, reg = out
    total = 0.0 * (logit.sum() + reg.sum())                     # zero that stays on the graph (no-label batch)
    ml = batch["m_lithology"]                                    # bool masks
    if bool(ml.any()):
        w = ml.to(logit.dtype)
        bce = F.binary_cross_entropy_with_logits(logit, batch["lithology"], pos_weight=pos_weight, reduction="none")
        total = total + (bce * w).sum() / w.sum()
    for k, h in enumerate(("vsh", "phit")):
        m = batch[f"m_{h}"]
        if bool(m.any()):
            w = m.to(reg.dtype)
            se = ((reg[:, k] - batch[h]) / reg_scale[k]) ** 2          # targets are NaN-free (filled with 0)
            total = total + (se * w).sum() / w.sum()
    return total


def _to_tensors(X, T, idx, device):
    import torch

    b = {"x": torch.as_tensor(np.ascontiguousarray(X[idx]), dtype=torch.float32, device=device)}
    for k, v in T.items():
        a = np.ascontiguousarray(v[idx])
        b[k] = torch.as_tensor(a, dtype=torch.bool if k.startswith("m_") else torch.float32, device=device)
    return b


def train(name: str, X_tr: np.ndarray, T_tr: dict, X_va: np.ndarray | None = None, T_va: dict | None = None,
          epochs: int = 60, batch_size: int = 128, patience: int = 10, lr: float = 1e-3, weight_decay: float = 1e-4,
          dropout: float = 0.2, seed: int = 2026, device: str = "cpu") -> dict:
    """Train one multi-task network.  With validation data: early stopping; returns the best state.

    Returns {"model", "history" (list of dicts), "best_epoch", "seconds", "pos_weight", "reg_scale"}.
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)
    model = build(name, X_tr.shape[1], dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ml = T_tr["m_lithology"]
    n_pos = float(T_tr["lithology"][ml].sum())
    n_neg = float(ml.sum() - n_pos)
    pos_weight = torch.tensor(n_neg / n_pos if n_pos > 0 else 1.0, device=device)
    reg_scale = []
    for h in ("vsh", "phit"):
        v = T_tr[h][T_tr[f"m_{h}"]]
        reg_scale.append(float(v.std()) if v.size > 1 and v.std() > 0 else 1.0)
    reg_scale_t = torch.tensor(reg_scale, device=device)
    n = X_tr.shape[0]
    history, best, best_state, best_epoch, bad = [], np.inf, None, epochs, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        tot, cnt = 0.0, 0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            if idx.size < 2:                                   # BatchNorm needs >= 2 samples
                continue
            b = _to_tensors(X_tr, T_tr, idx, device)
            loss = _loss(model(b["x"]), b, pos_weight, reg_scale_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * idx.size
            cnt += idx.size
        rec = {"epoch": ep, "train_loss": tot / max(1, cnt)}
        if X_va is not None and X_va.shape[0]:
            rec["val_loss"] = evaluate_loss(model, X_va, T_va, pos_weight, reg_scale_t, device, batch_size)
            if rec["val_loss"] < best - 1e-6:
                best, best_epoch, bad = rec["val_loss"], ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
        history.append(rec)
        if X_va is not None and bad >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    elif X_va is None:
        best_epoch = epochs
    return {"model": model, "history": history, "best_epoch": int(best_epoch), "seconds": time.time() - t0,
            "pos_weight": float(pos_weight), "reg_scale": reg_scale}


def evaluate_loss(model, X, T, pos_weight, reg_scale_t, device, batch_size=512) -> float:
    import torch

    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for s in range(0, X.shape[0], batch_size):
            idx = np.arange(s, min(s + batch_size, X.shape[0]))
            b = _to_tensors(X, T, idx, device)
            tot += float(_loss(model(b["x"]), b, pos_weight, reg_scale_t)) * idx.size
            cnt += idx.size
    return tot / max(1, cnt)


def predict(model, X: np.ndarray, device: str = "cpu", batch_size: int = 512,
            *, pin_memory: bool = False, precision: str = "fp32") -> dict[str, np.ndarray]:
    """Batched inference with optional pinned H2D transfer and CUDA autocast.

    FP32 remains the default to preserve the released model's numerical path.
    Pinned source buffers are retained until the queued non-blocking copies finish.
    """
    import torch

    precision = str(precision).casefold()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16 or bf16")
    is_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
    if precision != "fp32" and not is_cuda:
        raise ValueError("fp16/bf16 inference precision requires a CUDA device")
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    logits, vsh, phit = [], [], []
    pinned_sources = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, X.shape[0], max(1, int(batch_size))):
            source_np = np.ascontiguousarray(X[start:start + batch_size], dtype=np.float32)
            source = torch.from_numpy(source_np)
            if is_cuda and pin_memory:
                pinned = torch.empty(source.shape, dtype=source.dtype, pin_memory=True)
                pinned.copy_(source)
                pinned_sources.append(pinned)
                xb = pinned.to(device, non_blocking=True)
            else:
                xb = source.to(device, non_blocking=is_cuda)
            if is_cuda and precision != "fp32":
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    logit, reg = model(xb)
            else:
                logit, reg = model(xb)
            logits.append(torch.sigmoid(logit))
            vsh.append(reg[:, 0])
            phit.append(reg[:, 1])
    if is_cuda:
        torch.cuda.synchronize(device)
    def to_numpy(parts):
        return torch.cat(parts).float().cpu().numpy() if parts else np.zeros(0, dtype=np.float32)
    # Keep pinned_sources alive through the synchronization above.
    return {"lithology": to_numpy(logits), "vsh": to_numpy(vsh), "phit": to_numpy(phit)}


def state_for_saving(model) -> dict:
    return {k: v.detach().cpu() for k, v in copy.deepcopy(model.state_dict()).items()}
