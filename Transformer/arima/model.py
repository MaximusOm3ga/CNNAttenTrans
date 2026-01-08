from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ARIMAConfig:
    order: Tuple[int, int, int] = (10, 1, 1)
    trend: str = "c"
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False
    maxiter: int = 50
    method: str = "statespace"
    suppress_warnings: bool = True


class ARIMABaseline:
    def __init__(
        self,
        config: ARIMAConfig | None = None,
        feature_index: int = 0,
    ) -> None:
        self.config = config or ARIMAConfig()
        self.feature_index = feature_index

    def _fit_and_forecast_1d(self, x_1d: np.ndarray, steps: int) -> np.ndarray:
        try:
            from statsmodels.tsa.arima.model import ARIMA  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "statsmodels is required for ARIMA. Install with: pip install statsmodels"
            ) from e

        x_1d = np.asarray(x_1d, dtype=np.float64)

        if x_1d.size < max(10, sum(self.config.order) + 1) or np.allclose(x_1d, x_1d[0]):
            return np.full(steps, float(x_1d[-1]), dtype=np.float64)

        try:
            if self.config.suppress_warnings:
                import warnings
                from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    model = ARIMA(
                        x_1d,
                        order=self.config.order,
                        trend=self.config.trend,
                        enforce_stationarity=self.config.enforce_stationarity,
                        enforce_invertibility=self.config.enforce_invertibility,
                    )
                    res = model.fit(method=self.config.method, maxiter=self.config.maxiter)
            else:
                model = ARIMA(
                    x_1d,
                    order=self.config.order,
                    trend=self.config.trend,
                    enforce_stationarity=self.config.enforce_stationarity,
                    enforce_invertibility=self.config.enforce_invertibility,
                )
                res = model.fit(method=self.config.method, maxiter=self.config.maxiter)

            fc = res.forecast(steps=steps)
            return np.asarray(fc, dtype=np.float64)
        except Exception:
            return np.full(steps, float(x_1d[-1]), dtype=np.float64)

    def predict(
        self,
        X: np.ndarray,
        steps: Optional[int] = None,
    ) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected X to be 3D (N,T,F), got {X.shape}")

        N, T, F = X.shape
        if not (0 <= self.feature_index < F):
            raise ValueError(f"feature_index {self.feature_index} out of range for F={F}")

        H = int(steps if steps is not None else T)
        preds = np.empty((N, H), dtype=np.float64)

        for i in range(N):
            series = X[i, :, self.feature_index]
            preds[i] = self._fit_and_forecast_1d(series, steps=H)

        return preds
