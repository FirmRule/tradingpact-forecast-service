"""KiT sampling for the Desk chart (local preview only; KiT is AGPL-3.0, see vendor/KiT/LICENSE).

Turns the chart's candles into KiT's inputs the way the 14 November 2026 gold H1
test did (scratchpad kit_run.py): KiT's round-the-clock "crypto" market template,
unknown sector and instrument, price scales fitted on the candles sent (the test
fitted them on a year of earlier gold), bar close times, 32 Euler steps, fp32.
Differences from the test, all said in the README: scales from the request's own
candles, no tick volume when the chart has none, and a context of up to the
candles the chart sends (KiT's encoding is relative, so a shorter context is
built without padding as long as it is at least KiT's minimum for that scale).
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "vendor", "KiT"))

from kit_v1.config import Config  # noqa: E402
from kit_v1.data.calendar import MARKET_IDS, TIMESCALE_IDS, attach_window_calendar, bars_per_day  # noqa: E402
from kit_v1.data.features import FEATURE_NAMES, compute_features, compute_volume_ema, ema_alpha  # noqa: E402
from kit_v1.data.normalize import fit_mad_scale, normalize  # noqa: E402
from kit_v1.infer import denormalize_paths  # noqa: E402
from kit_v1.infer.sampler import _COND_KEYS, _chunked_forward  # noqa: E402
from kit_v1.infer.postprocess import paths_to_ohlcv  # noqa: E402
from kit_v1.model import KlineDiT  # noqa: E402

MARKET = "crypto"  # the only round-the-clock template KiT has
MINUTE = 60_000
SCALES = {MINUTE: "1m", 5 * MINUTE: "5m", 15 * MINUTE: "15m", 30 * MINUTE: "30m",
          60 * MINUTE: "1h", 120 * MINUTE: "2h", 1440 * MINUTE: "1d"}


class UnsupportedTimeframe(ValueError):
    pass


class TooFewCandles(ValueError):
    pass


class Cancelled(RuntimeError):
    """The caller went away; the forecast stopped part way."""


@dataclass(frozen=True)
class Candle:
    time: int  # bar open, ms since epoch, UTC
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None


def timeframe_ms(times: Sequence[int]) -> int:
    gaps = np.diff(np.asarray(times, dtype=np.int64))
    return int(np.median(gaps)) if gaps.size else 0


class KitEngine:
    def __init__(self, ckpt_path: str, device: str = "mps", euler_steps: int = 32):
        started = time.time()
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # tensors and plain data only
        self.cfg = Config.from_dict(ck["config"])
        model = KlineDiT(self.cfg.model)
        state = dict(model.state_dict())
        state.update({k: v.to(torch.float32) for k, v in ck["ema"]["shadow"].items()})
        model.load_state_dict(state)
        del ck
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.euler_steps = euler_steps
        self.load_seconds = time.time() - started

    def scale_for(self, candles: Sequence[Candle]) -> tuple[str, int]:
        tf = timeframe_ms([c.time for c in candles])
        scale = SCALES.get(tf)
        if scale is None:
            raise UnsupportedTimeframe(f"KiT has no {tf // MINUTE}-minute timescale")
        return scale, tf

    def sample(self, candles: Sequence[Candle], future_times: Sequence[int], samples: int,
               on_draft: Callable[[int, int, np.ndarray], None] | None = None,
               draft_every: int = 4,
               should_stop: Callable[[], bool] | None = None) -> tuple[np.ndarray, dict]:
        """Sampled paths, shape (samples, steps, 5) as open, high, low, close, volume.

        KiT starts every path as noise and refines all of them over its Euler
        steps. With on_draft, every draft_every steps it is called with
        (step, steps, paths): each path's current clean estimate (z - t*v),
        blurry early and converging on the final paths."""
        scale, tf = self.scale_for(candles)
        lc, lh = self.cfg.data.window_for(scale)
        tmin = self.cfg.data.trunc_min_for(scale)
        n_ctx = min(lc, len(candles) - 1)  # one bar before the context dates its first gap
        if n_ctx < tmin:
            raise TooFewCandles(f"KiT needs at least {tmin + 1} candles on {scale}")

        o, h, l, c = (np.array([getattr(x, k) for x in candles], dtype=np.float64) for k in ("open", "high", "low", "close"))
        has_volume = all(x.volume is not None for x in candles) and any((x.volume or 0) > 0 for x in candles)
        v = np.array([x.volume or 0.0 for x in candles], dtype=np.float64) if has_volume else np.ones(len(candles))
        hl = 5.0 * bars_per_day(MARKET, scale)
        feats = compute_features(o, h, l, c, v, half_life_bars=hl)
        stats = {f"{MARKET}|{scale}|{n}": fit_mad_scale(feats[n][1:]) for n in FEATURE_NAMES}
        clip = self.cfg.data.soft_clip_c
        xn = np.stack([normalize(feats[n], stats[f"{MARKET}|{scale}|{n}"], c=clip) for n in FEATURE_NAMES], 1).astype(np.float32)

        # KiT dates a bar by its close; the chart by its open.
        closes = pd.DatetimeIndex(pd.to_datetime([x.time + tf for x in candles], unit="ms"))
        future = [t + tf for t in future_times[:lh]]
        while len(future) < lh:  # KiT forecasts its own horizon; extra bars are dropped below
            future.append(future[-1] + tf)
        window = closes[-n_ctx:].append(pd.DatetimeIndex(pd.to_datetime(future, unit="ms")))
        cal = attach_window_calendar(window, MARKET, scale, n_ctx=n_ctx, prev_ts=closes[-n_ctx - 1])
        cond = {k: torch.from_numpy(cal[k]).unsqueeze(0) for k in ("clock", "sess", "dow", "event", "month", "yday")}
        cond.update({"market": torch.tensor([MARKET_IDS[MARKET]]), "sector": torch.tensor([0]), "inst": torch.tensor([0]),
                     "scale": torch.tensor([TIMESCALE_IDS[scale]]),
                     "pad_mask": torch.ones(1, n_ctx + lh, dtype=torch.bool)})
        x_ctx = torch.from_numpy(xn[-n_ctx:]).unsqueeze(0)
        seed = int.from_bytes(hashlib.sha256(f"{candles[-1].time}:{len(candles)}:{c[-1]}".encode()).digest()[:8], "big") >> 1
        ema = compute_volume_ema(v, hl)
        alpha = ema_alpha(hl)
        prev_ema = float((1 - alpha) * ema[-1] + alpha * v[-1])
        steps = min(len(future_times), lh)

        def to_ohlcv(z_norm: torch.Tensor) -> np.ndarray:
            fe = denormalize_paths(z_norm.view(1, samples, lh, -1), MARKET, scale, stats, c=clip)[0]
            ohlcv = paths_to_ohlcv(fe, prev_close=float(c[-1]), prev_ema=prev_ema, half_life_bars=hl)
            out = np.stack([ohlcv[k] for k in ("open", "high", "low", "close", "volume")], axis=-1)[:, :steps, :]
            if not has_volume:
                out[:, :, 4] = 0.0
            return out

        # KiT's flow_sample (Euler, no guidance, fp32), unrolled to report drafts.
        dev = self.device
        x_rep = x_ctx.to(dev).repeat_interleave(samples, dim=0)
        cond_rep = {k: cond[k].to(dev).repeat_interleave(samples, dim=0) for k in _COND_KEYS if k in cond}
        z = torch.randn(samples, lh, x_ctx.shape[-1], generator=torch.Generator().manual_seed(seed),
                        dtype=x_rep.dtype).to(dev)
        grid = torch.linspace(1.0, 0.0, self.euler_steps + 1)
        with torch.no_grad():
            for i in range(self.euler_steps):
                if should_stop is not None and should_stop():
                    raise Cancelled()
                t_cur, t_next = float(grid[i]), float(grid[i + 1])
                t_vec = torch.full((samples,), t_cur, device=dev, dtype=x_rep.dtype)
                vel = _chunked_forward(self.model, torch.cat([x_rep, z], dim=1), t_vec, cond_rep, lh, 64).float()
                if on_draft is not None and (i + 1) % draft_every == 0 and i + 1 < self.euler_steps:
                    on_draft(i + 1, self.euler_steps, to_ohlcv(z - t_cur * vel))
                z = z - (t_cur - t_next) * vel
        out = to_ohlcv(z)
        return out, {"scale": scale, "context": n_ctx, "horizon": steps, "nativeHorizon": lh, "volume": has_volume}
