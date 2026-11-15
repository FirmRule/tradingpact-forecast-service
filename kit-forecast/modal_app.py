"""KiT forecast service on Modal (an NVIDIA L4 GPU, scales to zero).

Same FastAPI app as the local preview (main.py), same contract as apps/kronos:
POST /forecast, POST /forecast/stream, GET /health, bearer token.

KiT is AGPL-3.0 (vendor/KiT/LICENSE). Offering this to traders over a network
obliges publishing this service's source; until the owner decides, only the
local app points at it.

One-time setup, from apps/kit-forecast after ./setup.sh:
    .venv/bin/modal token new
    .venv/bin/modal volume create kit-weights
    .venv/bin/modal volume put kit-weights ckpt/step_market_284M.pt /step_market_284M.pt
    .venv/bin/modal secret create kit-api-token --from-dotenv .env.local
Deploy, then time it:
    .venv/bin/modal deploy modal_app.py
    .venv/bin/python bench.py https://<workspace>--tradingpact-kit-forecast-web.modal.run 3
Full context: docs/plans/2026-11-15-kit-modal-handoff.md
"""

from __future__ import annotations

import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
KRONOS_FORECAST = os.path.join(HERE, "..", "kronos", "forecast.py")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # The versions the local preview runs and was checked with.
        "torch==2.15.0",
        "numpy==2.5.4",
        "pandas==3.1.0",
        "fastapi==0.163.0",
        "pydantic==2.14.3",
    )
    .env(
        {
            "KIT_DEVICE": "cuda",
            "KIT_CKPT": "/weights/step_market_284M.pt",
            "PYTHONPATH": "/root/app",
        }
    )
    # The service, the vendored KiT (with its licence) and the card figures it
    # shares with apps/kronos; main.py finds forecast.py at ../kronos.
    .add_local_file(os.path.join(HERE, "main.py"), "/root/app/main.py")
    .add_local_file(os.path.join(HERE, "kit_engine.py"), "/root/app/kit_engine.py")
    .add_local_dir(
        os.path.join(HERE, "vendor"),
        "/root/app/vendor",
        ignore=["**/__pycache__/**"],
    )
    .add_local_file(KRONOS_FORECAST, "/root/kronos/forecast.py")
)

app = modal.App("tradingpact-kit-forecast", image=image)
weights = modal.Volume.from_name("kit-weights")


@app.function(
    gpu="L4",
    volumes={"/weights": weights},
    secrets=[modal.Secret.from_name("kit-api-token")],
    # Stay warm two minutes after the last forecast, then scale to zero.
    scaledown_window=120,
    timeout=600,
    max_containers=2,
)
@modal.asgi_app()
def web():
    import sys

    sys.path.insert(0, "/root/app")
    from main import app as fastapi_app

    return fastapi_app
