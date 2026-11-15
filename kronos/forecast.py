"""Kronos forecasts as the Desk chart draws them.

Kronos's own `predict` averages its sampled paths into one line. To show a
range, the same series goes through `predict_batch` once per sample, so each
row of the batch is an independent path; the summary keeps the mean path, the
min-max band across paths and the shares of paths that end up or swing more.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["open", "high", "low", "close"]
OPEN, HIGH, LOW, CLOSE, VOLUME = range(5)


@dataclass(frozen=True)
class Bar:
    time: int  # ms since epoch, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


class BatchPredictor(Protocol):
    def predict_batch(  # noqa: PLR0913 - mirrors KronosPredictor.predict_batch
        self,
        df_list: list[pd.DataFrame],
        x_timestamp_list: list[pd.Series],
        y_timestamp_list: list[pd.Series],
        pred_len: int,
        T: float = 1.0,  # noqa: N803 - upstream argument name
        top_k: int = 0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = True,
    ) -> list[pd.DataFrame]: ...


def has_volume(bars: Sequence[Bar]) -> bool:
    """Volume counts only when every bar has it and some of it is non-zero."""
    return all(bar.volume is not None for bar in bars) and any(
        (bar.volume or 0) > 0 for bar in bars
    )


def _timestamps(times: Sequence[int]) -> pd.Series:
    return pd.Series(pd.to_datetime(list(times), unit="ms", utc=True))


def frame(bars: Sequence[Bar]) -> pd.DataFrame:
    data: dict[str, list[float]] = {
        "open": [bar.open for bar in bars],
        "high": [bar.high for bar in bars],
        "low": [bar.low for bar in bars],
        "close": [bar.close for bar in bars],
    }
    if has_volume(bars):
        data["volume"] = [float(bar.volume or 0) for bar in bars]
    return pd.DataFrame(data)


def sample_paths(  # noqa: PLR0913
    predictor: BatchPredictor,
    bars: Sequence[Bar],
    future_times: Sequence[int],
    samples: int,
    temperature: float,
    top_p: float,
) -> np.ndarray:
    """Independent sampled paths, shape (samples, steps, 5): OHLC and volume."""
    df = frame(bars)
    x_time = _timestamps([bar.time for bar in bars])
    y_time = _timestamps(future_times)
    results = predictor.predict_batch(
        [df] * samples,
        [x_time] * samples,
        [y_time] * samples,
        pred_len=len(future_times),
        T=temperature,
        top_k=0,
        top_p=top_p,
        sample_count=1,
        verbose=False,
    )
    columns = [*PRICE_COLUMNS, "volume"]
    return np.stack([result[columns].to_numpy(dtype=np.float64) for result in results])


def path_rows(paths: np.ndarray) -> list[list[list[float]]]:
    """
    Each sampled path as OHLC rows, made coherent (high at or above the open,
    close and low; low at or below them): a joint trajectory per sample, for
    callers that need first-passage probabilities rather than the summary.
    """
    out = []
    for path in paths:
        rows = []
        for step in path:
            o, h, lo, c = (float(v) for v in step[:4])
            rows.append([round(o, 6), round(max(o, h, lo, c), 6), round(min(o, h, lo, c), 6), round(c, 6)])
        out.append(rows)
    return out


def _log_return_std(closes: np.ndarray) -> float:
    closes = closes[closes > 0]
    if closes.size < 3:
        return float("nan")
    return float(np.std(np.diff(np.log(closes)), ddof=1))


# A bar spacing this many times the usual one is a market closure (a weekend).
CLOSURE_GAP = 3.0


def _recent_volatility(bars: Sequence[Bar], count: int) -> float:
    """
    Close-to-close volatility of the most recent `count` returns, leaving out
    returns across a closure: a weekend's jump is not the market's pace.
    """
    times = np.array([bar.time for bar in bars], dtype=np.float64)
    closes = np.array([bar.close for bar in bars], dtype=np.float64)
    if closes.size < 3 or (closes <= 0).any():
        return float("nan")
    gaps = np.diff(times)
    returns = np.diff(np.log(closes))
    usual = float(np.median(gaps))
    within = returns[gaps <= usual * CLOSURE_GAP][-count:]
    if within.size < 2:
        return float("nan")
    return float(np.std(within, ddof=1))


def _pct(value: float, base: float) -> float:
    return (value / base - 1.0) * 100.0


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def summarize(
    paths: np.ndarray,
    bars: Sequence[Bar],
    future_times: Sequence[int],
) -> dict[str, Any]:
    """
    The chart's figures from the sampled paths.

    - Each step: the mean OHLC across paths, the lowest and highest close
      (the band), and the mean volume when the candles carry volume.
    - upsideProbability: the share of paths whose last close is above the
      current close.
    - volatilityAmplificationProbability: the share of paths whose close-to-
      close volatility over the horizon exceeds that of the same number of
      most recent real returns (as Kronos's own demo measures it), leaving out
      returns across a market closure.
    - meanChangePct, minChangePct, maxChangePct: the mean, lowest and highest
      last close against the current close.
    """
    if paths.ndim != 3 or paths.shape[1] != len(future_times):
        raise ValueError("paths must be (samples, steps, 5) with one step per time")
    last_close = bars[-1].close
    steps = len(future_times)
    closes = paths[:, :, CLOSE]
    mean = paths.mean(axis=0)
    with_volume = has_volume(bars)

    step_rows = []
    for index, time in enumerate(future_times):
        row: dict[str, Any] = {
            "time": int(time),
            "open": float(mean[index, OPEN]),
            "high": float(mean[index, HIGH]),
            "low": float(mean[index, LOW]),
            "close": float(mean[index, CLOSE]),
            "min": float(closes[:, index].min()),
            "max": float(closes[:, index].max()),
            "volume": max(0.0, float(mean[index, VOLUME])) if with_volume else None,
        }
        step_rows.append(row)

    final = closes[:, -1]
    recent_vol = _recent_volatility(bars, steps)
    path_vols = np.array(
        [_log_return_std(np.concatenate([[last_close], path])) for path in closes]
    )
    if math.isfinite(recent_vol) and recent_vol > 0 and np.isfinite(path_vols).all():
        amplification: float | None = float(np.mean(path_vols > recent_vol))
    else:
        amplification = None

    return {
        "lastClose": float(last_close),
        "lastTime": int(bars[-1].time),
        "samples": int(paths.shape[0]),
        "steps": step_rows,
        "stats": {
            "upsideProbability": float(np.mean(final > last_close)),
            "volatilityAmplificationProbability": amplification,
            "meanChangePct": _finite_or_none(_pct(float(final.mean()), last_close)),
            "minChangePct": _finite_or_none(_pct(float(final.min()), last_close)),
            "maxChangePct": _finite_or_none(_pct(float(final.max()), last_close)),
            "recentVolatility": _finite_or_none(recent_vol),
            "forecastVolatility": _finite_or_none(float(np.nanmean(path_vols))),
        },
    }
