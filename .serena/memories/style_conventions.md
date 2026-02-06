# Code Style & Conventions

## Python
- **Line length:** 100 (black + ruff)
- **Target version:** Python 3.11
- **Formatter:** black
- **Linter:** ruff (rules: E, F, I, N, W, UP)
- **Type checker:** mypy (strict mode)
- **Type hints:** Required (mypy strict)
- **Test framework:** pytest with pytest-asyncio (auto mode)

## Frontend (webapp/)
- Next.js + TypeScript
- Tailwind CSS
- ESLint

## Patterns
- Multi-agent system using CrewAI framework
- Agent base class in `src/agents/base.py`
- Config-driven agents via `configs/agents.yaml` and `configs/tasks.yaml`
- FastAPI routers in `api/routers/`
- Pydantic models for validation
- structlog for logging
