"""Visualization package: replay encoders, rendering, and studio communication."""

from autogen.visualization.adapter import ViewerDataAdapter
from autogen.visualization.contracts import (
    ViewerAgentMindCell,
    ViewerAgentState,
    ViewerCellState,
    ViewerOverlayData,
    ViewerState,
    ViewerTimeMachineState,
)
from autogen.visualization.pygame_viewer import PygameViewer
from autogen.visualization.renderer import (
    TERRAIN_PALETTE,
    UI_PALETTE,
    GridCoordinateProjector,
)
from autogen.visualization.time_machine import TimeMachineController

__all__ = [
    "GridCoordinateProjector",
    "PygameViewer",
    "TERRAIN_PALETTE",
    "TimeMachineController",
    "UI_PALETTE",
    "ViewerAgentMindCell",
    "ViewerAgentState",
    "ViewerCellState",
    "ViewerDataAdapter",
    "ViewerOverlayData",
    "ViewerState",
    "ViewerTimeMachineState",
]
