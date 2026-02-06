# Suggested Commands

## Development
- `pip install -e .` — Install project in editable mode
- `cd api && uvicorn main:app --reload --port 8000` — Run backend
- `cd webapp && npm run dev` — Run frontend
- `docker compose up -d` — Run full stack via Docker

## Testing
- `pytest` — Run tests (testpaths: tests/, asyncio_mode: auto)
- `pytest --cov` — Run tests with coverage

## Linting & Formatting
- `black --line-length 100 src/ api/` — Format code
- `ruff check src/ api/` — Lint code
- `ruff check --fix src/ api/` — Auto-fix lint issues
- `mypy src/` — Type checking (strict mode)

## CLI Entry Points
- `run-backtest` — Run historical backtesting
- `run-paper` — Run paper trading
- `train-rl` — Train reinforcement learning models

## Git
- Main branch: `dev`
- Current branch: `dev-next`
