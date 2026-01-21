"""Agent configuration API endpoints."""

from pathlib import Path
from fastapi import APIRouter, Request
from pydantic import BaseModel
import structlog
import yaml

logger = structlog.get_logger()
router = APIRouter()

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"


class AgentConfig(BaseModel):
    """Agent configuration - matches CrewAI-like structure."""
    name: str
    role: str
    goal: str
    backstory: str = ""
    system_prompt: str = ""
    model: str = "gpt-5-mini-2025-08-07"
    temperature: float = 0.5
    enabled: bool = True


@router.get("/")
async def list_agents(request: Request):
    """List all agent configurations."""
    state = request.app.state.app_state
    return {
        "agents": [
            {"id": agent_id, **config}
            for agent_id, config in state.agent_configs.items()
        ]
    }


@router.get("/{agent_id}")
async def get_agent(request: Request, agent_id: str):
    """Get specific agent configuration."""
    state = request.app.state.app_state
    
    config = state.agent_configs.get(agent_id)
    if not config:
        return {"error": f"Agent {agent_id} not found"}
    
    return {"id": agent_id, **config}


@router.put("/{agent_id}")
async def update_agent(request: Request, agent_id: str, config: AgentConfig):
    """Update agent configuration."""
    state = request.app.state.app_state
    
    if agent_id not in state.agent_configs:
        return {"error": f"Agent {agent_id} not found"}
    
    state.agent_configs[agent_id] = config.model_dump()
    state.save_agent_configs()
    
    logger.info("Updated agent config", agent_id=agent_id)
    
    state.log_activity({
        "type": "config",
        "action": "UPDATED",
        "symbol": "-",
        "details": f"Updated {agent_id} agent configuration",
    })
    
    return {"status": "success", "id": agent_id, **state.agent_configs[agent_id]}


@router.post("/")
async def create_agent(request: Request, agent_id: str, config: AgentConfig):
    """Create new agent configuration."""
    state = request.app.state.app_state
    
    if agent_id in state.agent_configs:
        return {"error": f"Agent {agent_id} already exists"}
    
    state.agent_configs[agent_id] = config.model_dump()
    state.save_agent_configs()
    
    logger.info("Created agent config", agent_id=agent_id)
    
    return {"status": "success", "id": agent_id, **state.agent_configs[agent_id]}


@router.delete("/{agent_id}")
async def delete_agent(request: Request, agent_id: str):
    """Delete agent configuration."""
    state = request.app.state.app_state
    
    if agent_id not in state.agent_configs:
        return {"error": f"Agent {agent_id} not found"}
    
    # Don't allow deleting core agents
    if agent_id in ["technical", "fundamental", "executive"]:
        return {"error": "Cannot delete core agents"}
    
    del state.agent_configs[agent_id]
    state.save_agent_configs()
    
    logger.info("Deleted agent config", agent_id=agent_id)
    
    return {"status": "success", "message": f"Agent {agent_id} deleted"}


@router.get("/configs/full")
async def get_full_configs():
    """Get full agent and task configurations from YAML files."""
    agents_config = {}
    tasks_config = {}
    
    agents_path = CONFIGS_DIR / "agents.yaml"
    tasks_path = CONFIGS_DIR / "tasks.yaml"
    
    if agents_path.exists():
        with open(agents_path) as f:
            agents_config = yaml.safe_load(f) or {}
    
    if tasks_path.exists():
        with open(tasks_path) as f:
            tasks_config = yaml.safe_load(f) or {}
    
    return {
        "agents": agents_config,
        "tasks": tasks_config,
        "files": {
            "agents": str(agents_path),
            "tasks": str(tasks_path),
        }
    }


@router.get("/tasks")
async def list_tasks():
    """List all task configurations."""
    tasks_path = CONFIGS_DIR / "tasks.yaml"
    
    if tasks_path.exists():
        with open(tasks_path) as f:
            tasks = yaml.safe_load(f) or {}
        return {"tasks": [{"id": k, **v} for k, v in tasks.items()]}
    
    return {"tasks": []}
