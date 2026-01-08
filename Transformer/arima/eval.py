import numpy as np
import torch

from model import ARIMABaseline, ARIMAConfig

DATA_PATH = "../dataset/paper_new_test.pt"


def eval_arima_rolling_one_step(
    order=(5, 1, 1),
    trend="n",
    warmup: int = 10,
    max_train_len: int | None = None,
) -> None:
    data = torch.load(DATA_PATH, map_location="cpu")
    Y = data["Y"].float().numpy()

    if Y.ndim != 2:
        raise ValueError(f"Expected Y to be 2D (N,T), got {Y.shape}")

    N, T = Y.shape
    if not (1 <= warmup < T):
        raise ValueError(f"warmup must be in [1, T-1], got warmup={warmup}, T={T}")

    model = ARIMABaseline(config=ARIMAConfig(order=order, trend=trend), feature_index=0)

    preds = np.full((N, T), np.nan, dtype=np.float64)

    for i in range(N):
        y = Y[i].astype(np.float64, copy=False)

        for t in range(warmup, T):
            start = 0
            if max_train_len is not None and max_train_len > 0:
                start = max(0, t - int(max_train_len))

            hist = y[start:t]
            x3d = hist[None, :, None]

            fc = model.predict(x3d, steps=1)
            preds[i, t] = float(fc[0, 0])

    y_true = Y[:, warmup:]
    y_pred = preds[:, warmup:]

    if np.isnan(y_pred).any():
        raise RuntimeError("NaNs in predictions; check warmup and data validity.")

    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))

    print("ARIMA baseline (univariate on Y, rolling 1-step)")
    print(f"order: {order}, trend: {trend}, warmup: {warmup}/{T}")
    if max_train_len is not None:
        print(f"max_train_len: {int(max_train_len)}")
    print(f"MSE :  {mse:.6f}")
    print(f"MAE :  {mae:.6f}")


if __name__ == "__main__":
    eval_arima_rolling_one_step(order=(5, 1, 1), trend="n", warmup=10, max_train_len=24)
