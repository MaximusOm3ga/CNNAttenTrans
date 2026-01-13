import torch


def make_forecast_pairs(X: torch.Tensor, Y: torch.Tensor, horizon: int):
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if X.dim() != 3 or Y.dim() != 2:
        raise ValueError(f"Expected X (N,T,F) and Y (N,T). Got {tuple(X.shape)} and {tuple(Y.shape)}")
    if X.size(0) != Y.size(0) or X.size(1) != Y.size(1):
        raise ValueError(f"Time/N mismatch: X {tuple(X.shape)} vs Y {tuple(Y.shape)}")
    T = int(X.size(1))
    if T <= horizon:
        raise ValueError(f"Not enough timesteps T={T} for horizon={horizon}")
    return X[:, : T - horizon, :], Y[:, horizon:]


def reduce_to_patches(y: torch.Tensor, patch_size: int, reduction: str = "mean") -> torch.Tensor:
    if y.dim() != 2:
        raise ValueError(f"Expected y to be 2D (N,T), got {tuple(y.shape)}")
    N, T = y.shape
    P = T // patch_size
    if P <= 0:
        raise ValueError(f"Sequence too short T={T} for patch_size={patch_size}")
    T_eff = P * patch_size
    y_eff = y[:, :T_eff].contiguous().view(N, P, patch_size)
    if reduction == "mean":
        return y_eff.mean(dim=-1)
    if reduction == "last":
        return y_eff[:, :, -1]
    raise ValueError(f"Unknown reduction={reduction}")


def reduce_preds_to_patches(preds: torch.Tensor, patch_size: int, reduction: str = "mean") -> torch.Tensor:
    if preds.dim() != 2:
        raise ValueError(f"Expected preds to be 2D (N,T), got {tuple(preds.shape)}")
    return reduce_to_patches(preds, patch_size=patch_size, reduction=reduction)


def make_patch_forecast_pairs(x_patch: torch.Tensor, y_patch: torch.Tensor, horizon_patches: int):
    if horizon_patches <= 0:
        raise ValueError(f"horizon_patches must be > 0, got {horizon_patches}")
    if x_patch.dim() != 2 or y_patch.dim() != 2:
        raise ValueError(f"Expected x_patch (N,P) and y_patch (N,P), got {tuple(x_patch.shape)} and {tuple(y_patch.shape)}")
    if x_patch.shape != y_patch.shape:
        raise ValueError(f"Shape mismatch: x_patch {tuple(x_patch.shape)} vs y_patch {tuple(y_patch.shape)}")
    P = int(x_patch.size(1))
    if P <= horizon_patches:
        raise ValueError(f"Not enough patches P={P} for horizon_patches={horizon_patches}")
    return x_patch[:, : P - horizon_patches], y_patch[:, horizon_patches:]
