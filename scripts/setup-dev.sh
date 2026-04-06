#!/usr/bin/env bash
set -euo pipefail

echo "==> Installing STATE in dev mode..."
uv sync --group dev

echo "==> Verifying installation..."
uv run state --help > /dev/null 2>&1 && echo "    state CLI: OK" || echo "    state CLI: FAILED"

echo "==> Running ruff check..."
uv run ruff check src/ --statistics || true

echo ""
echo "==> Dev environment ready!"
echo "    Run tests:  uv run pytest"
echo "    Run CLI:    uv run state --help"
echo "    Format:     uv run ruff format src/"
