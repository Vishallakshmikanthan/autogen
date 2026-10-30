"""World-linked symbolic grounding and deductive safety filtering (SYS-07, Module M08)."""

from __future__ import annotations

from autogen.intelligence.logic.ast import Formula
from autogen.intelligence.logic.knowledge_base import KnowledgeBase
from autogen.intelligence.logic.parser import parse_formula
from autogen.intelligence.logic.provenance import ProofTree
from autogen.simulation.core.contracts import Observation
from autogen.simulation.state.types import ActionType, DIRECTION_DISPLACEMENTS, Position, TerrainType


class SafetyRuleLayer:
    """Logical reasoning filter validating candidate agent actions against axiomatic safety rules."""

    def __init__(self, energy_threshold: float = 30.0) -> None:
        self.energy_threshold = energy_threshold
        self.kb = KnowledgeBase()
        self._init_safety_rules()

    def _init_safety_rules(self) -> None:
        """Loads domain-level safety inference axioms into the Knowledge Base."""
        rules = [
            "Obstacle => ~Safe",
            "Water => ~Safe",
            "DeepHazard => ~Safe",
            "PredatorNear & Track => Hazard",
            "Hazard => ~Safe",
        ]
        for r in rules:
            self.kb.tell(r)

    def ground_perceptions(self, observation: Observation) -> None:
        """Grounds sensory cell and agent observations into propositional knowledge base facts."""
        pos = observation.position
        # Clear dynamic transient facts from previous step
        # Re-initialize baseline rules
        self.kb = KnowledgeBase()
        self._init_safety_rules()

        # Check for nearby visible agents/predators
        if observation.visible_agents:
            self.kb.tell("PredatorNear")

        for cell in observation.visible_cells:
            cpos = cell.position
            # Ground obstacles
            if cell.terrain == TerrainType.OBSTACLE or cell.terrain == TerrainType.MOUNTAIN:
                self.kb.tell(f"Obstacle_{cpos.x}_{cpos.y}")
                self.kb.tell(f"Obstacle_{cpos.x}_{cpos.y} => ~Safe_{cpos.x}_{cpos.y}")
            elif cell.terrain == TerrainType.WATER:
                self.kb.tell(f"Water_{cpos.x}_{cpos.y}")
                self.kb.tell(f"Water_{cpos.x}_{cpos.y} => ~Safe_{cpos.x}_{cpos.y}")

    def is_action_provably_unsafe(
        self,
        current_pos: Position,
        action: ActionType,
    ) -> tuple[bool, ProofTree | None]:
        """Tests whether moving along candidate action is provably unsafe (KB |= ~Safe_x_y)."""
        if action not in DIRECTION_DISPLACEMENTS:
            return False, None

        dx, dy = DIRECTION_DISPLACEMENTS[action]
        target = Position(current_pos.x + dx, current_pos.y + dy)
        query = f"~Safe_{target.x}_{target.y}"

        # Ask KB if destination is provably unsafe
        unsafe, proof = self.kb.ask(query)
        return unsafe, proof
