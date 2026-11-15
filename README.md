# TradingPact forecast service

The source of the AI model service that draws the sampled price paths on TradingPact's Forecast card. It is published under the GNU Affero General Public License v3.0 because it includes KiT, which is AGPL-3.0.

- `kit-forecast/main.py`: the HTTP service (FastAPI).
- `kit-forecast/kit_engine.py`: turns a chart's candles into KiT's inputs and samples future candle paths.
- `kit-forecast/modal_app.py`: runs the service on a Modal L4 GPU, which is how it is deployed.
- `kit-forecast/vendor/KiT/`: KiT (https://github.com/Luciferbobo/KiT) at the commit in `COMMIT`, unmodified, with its licence and README.
- `kronos/forecast.py`: the summary figures the service returns (the band across paths, the mean path, the share finishing higher).

The model weights are KiT's published `step_market_284M.pt` (Hugging Face `Lucifer744/KiT_model`). `setup.sh` downloads them and checks their SHA-256; they are not in this repository.

## Run it

Needs `uv`, `curl`, `openssl` and about 4 GB free.

```bash
cd kit-forecast
./setup.sh                          # weights, a Python 3.12 venv, a bearer token in .env.local
set -a; . ./.env.local; set +a
KIT_DEVICE=cpu .venv/bin/python -I -m uvicorn main:app --host 127.0.0.1 --port 4007 --app-dir .   # or cuda / mps
```

- `GET /health`: whether the model has loaded.
- `POST /forecast`: candles in (`bars`, at least 64, in time order), sampled paths summarised out.
- `POST /forecast/stream`: the same as NDJSON, with drafts of the paths while KiT refines them, then the final forecast.

Every request except `/health` needs `Authorization: Bearer $KIT_API_TOKEN`. Timeframes: 1m, 5m, 15m, 30m, 1h and 1D (KiT's own; others get a 501).

To deploy on Modal, see the steps at the top of `modal_app.py`.

## Licence

GNU Affero General Public License v3.0 (`LICENSE`). KiT is the work of its authors, under the same licence (`kit-forecast/vendor/KiT/LICENSE`). The TradingPact app calls this service over HTTP; the app's own source is not part of this repository.

The paths are display only: they never set a rule, place an order or change a position. On held-out gold 1-hour candles the range of 20 paths held the close 89% of the time against about 90% implied, no better than a volatility range, and its direction did not beat chance. Other instruments and timeframes are not validated.
