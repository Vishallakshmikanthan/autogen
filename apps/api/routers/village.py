"""Village Society & Emergent Economy API router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from apps.api.schemas.common import APIResponseEnvelope
from autogen.agents.village.controller import VillageSimulationController
from autogen.research.village_experiments import run_all_village_experiments
from autogen.simulation.environment.village.village_grid import VillageWorldConfig

router = APIRouter(prefix="/api/v1/village", tags=["village"])

_VILLAGE_CONTROLLER: VillageSimulationController | None = None


def get_village_controller() -> VillageSimulationController:
    """Returns or initializes the singleton VillageSimulationController."""
    global _VILLAGE_CONTROLLER
    if _VILLAGE_CONTROLLER is None:
        from autogen.intelligence.village.persona.llm_bridge import _read_dotenv_configs

        cfg = _read_dotenv_configs()
        has_key = bool(
            cfg.get("NEMOTRON_API_KEY")
            or cfg.get("NVIDIA_API_KEY")
            or cfg.get("OPENAI_API_KEY")
            or cfg.get("GEMINI_API_KEY")
        )
        _VILLAGE_CONTROLLER = VillageSimulationController(
            world_config=VillageWorldConfig(width=40, height=40, seed=42),
            enable_llm_persona=has_key,
        )
    return _VILLAGE_CONTROLLER


class VillageResetRequest(BaseModel):
    """Request payload for resetting or reconfiguring the Village Society simulation."""

    model_config = ConfigDict(extra="forbid")

    seed: int = Field(default=42, ge=0)
    width: int = Field(default=40, ge=20, le=80)
    height: int = Field(default=40, ge=20, le=80)
    drought_active: bool = False
    enable_llm_persona: bool = False


class VillageStepRequest(BaseModel):
    """Request payload for stepping the Village Society simulation."""

    model_config = ConfigDict(extra="forbid")

    steps: int = Field(default=1, ge=1, le=50)
    time_of_day_override: float | None = Field(default=None, ge=0.0, lt=24.0)


@router.get("/state", response_model=APIResponseEnvelope[dict[str, Any]])
def get_village_state() -> Any:
    """Returns the complete current state of AUTOGEN Village."""
    ctrl = get_village_controller()
    return APIResponseEnvelope(
        success=True,
        data=ctrl.get_full_state_payload(),
        message="Village state retrieved",
    )


@router.post("/step", response_model=APIResponseEnvelope[dict[str, Any]])
def step_village_simulation(req: VillageStepRequest | None = None) -> Any:
    """Advances the AUTOGEN Village simulation by one or more ticks."""
    ctrl = get_village_controller()
    n_steps = req.steps if req is not None else 1
    if req is not None and req.time_of_day_override is not None:
        ctrl.world.time_of_day = req.time_of_day_override

    state = ctrl.get_full_state_payload()
    for _ in range(n_steps):
        state = ctrl.step()
    return APIResponseEnvelope(
        success=True,
        data=state,
        message=f"Village stepped {n_steps} tick(s)",
    )


@router.post("/reset", response_model=APIResponseEnvelope[dict[str, Any]])
def reset_village_simulation(req: VillageResetRequest | None = None) -> Any:
    """Resets the AUTOGEN Village simulation with new seed or environmental regime."""
    global _VILLAGE_CONTROLLER
    cfg = req or VillageResetRequest()
    _VILLAGE_CONTROLLER = VillageSimulationController(
        world_config=VillageWorldConfig(
            width=cfg.width,
            height=cfg.height,
            seed=cfg.seed,
            drought_active=cfg.drought_active,
        ),
        enable_llm_persona=cfg.enable_llm_persona,
    )
    return APIResponseEnvelope(
        success=True,
        data=_VILLAGE_CONTROLLER.get_full_state_payload(),
        message="Village simulation reset",
    )


@router.post("/drought", response_model=APIResponseEnvelope[dict[str, Any]])
def toggle_village_drought(active: bool = True) -> Any:
    """Toggles drought/famine scarcity pressure in the active village."""
    ctrl = get_village_controller()
    ctrl.world.drought_active = active
    ctrl.world.well = ctrl.world.well.model_copy(
        update={"regen_per_step": 2.5 if active else 10.0}
    )
    return APIResponseEnvelope(
        success=True,
        data=ctrl.get_full_state_payload(),
        message=f"Drought regime set to {active}",
    )


@router.get("/experiments", response_model=APIResponseEnvelope[list[dict[str, Any]]])
def get_village_experiments() -> Any:
    """Runs and returns the 3 reproducible Phase 6 Village Society empirical scenarios."""
    results = run_all_village_experiments()
    return APIResponseEnvelope(
        success=True,
        data=results,
        message="Village empirical scenarios completed",
    )


@router.get("/export", response_model=APIResponseEnvelope[dict[str, Any]])
def export_village_world() -> Any:
    """Exports the village_world.json specification."""
    ctrl = get_village_controller()
    return APIResponseEnvelope(
        success=True,
        data=ctrl.world.export_village_world_json(),
        message="Village world exported",
    )

