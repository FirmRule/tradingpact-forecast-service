#!/usr/bin/env bash
# Sets up the KiT forecast service on a fresh machine (macOS or Linux):
# KiT's source at the tested commit, its weights (checked by SHA-256), a
# Python 3.12 venv, and a bearer token in .env.local. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

KIT_REPO=https://github.com/Luciferbobo/KiT.git
KIT_COMMIT=78acec5f9de5787672f0f9cc201f73c6e3526c43
CKPT_URL=https://huggingface.co/Lucifer744/KiT_model/resolve/main/step_market_284M/step_market_284M.pt
CKPT_SHA256=7637d62c76ab9cce78de04a439f952b25c2a427cd46d4292d4ad9511633ed0b7

if [ ! -f vendor/KiT/COMMIT ]; then
  tmp=$(mktemp -d)
  git clone --quiet "$KIT_REPO" "$tmp/KiT"
  git -C "$tmp/KiT" checkout --quiet "$KIT_COMMIT"
  mkdir -p vendor/KiT
  cp -R "$tmp/KiT/kit_v1" "$tmp/KiT/LICENSE" "$tmp/KiT/README.md" vendor/KiT/
  echo "$KIT_COMMIT" > vendor/KiT/COMMIT
  rm -rf "$tmp"
  echo "KiT source at $KIT_COMMIT (AGPL-3.0, vendor/KiT/LICENSE)"
fi

mkdir -p ckpt
if [ ! -f ckpt/step_market_284M.pt ]; then
  echo "Downloading weights (1.1 GB)…"
  curl -fL --retry 3 -o ckpt/step_market_284M.pt.part "$CKPT_URL"
  mv ckpt/step_market_284M.pt.part ckpt/step_market_284M.pt
fi
actual=$(shasum -a 256 ckpt/step_market_284M.pt 2>/dev/null || sha256sum ckpt/step_market_284M.pt)
if [ "${actual%% *}" != "$CKPT_SHA256" ]; then
  echo "Weights checksum mismatch: ${actual%% *}" >&2
  exit 1
fi
echo "Weights verified."

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.12 .venv
fi
uv pip install --python .venv/bin/python -r requirements.txt

if [ ! -f .env.local ]; then
  printf 'KIT_API_TOKEN=%s\n' "$(openssl rand -base64 48 | tr -d '/+=' | head -c 40)" > .env.local
  chmod 600 .env.local
  echo "Created .env.local with a new KIT_API_TOKEN."
fi
echo "Ready. Next: see docs/plans/2026-11-15-kit-modal-handoff.md"
