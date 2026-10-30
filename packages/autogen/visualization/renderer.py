"""Pure grid coordinate projection and color palettes for renderers."""

from __future__ import annotations

from autogen.simulation.state.types import TerrainType

TERRAIN_PALETTE: dict[str, tuple[int, int, int]] = {
    TerrainType.GRASSLAND.value: (76, 175, 80),
    TerrainType.FOREST.value: (46, 125, 50),
    TerrainType.DESERT.value: (238, 203, 137),
    TerrainType.WATER.value: (33, 150, 243),
    TerrainType.MOUNTAIN.value: (120, 144, 156),
    TerrainType.OBSTACLE.value: (66, 66, 66),
}

UI_PALETTE = {
    "bg_dark": (26, 26, 36),
    "panel_bg": (36, 36, 48),
    "border": (60, 60, 80),
    "text_main": (240, 240, 250),
    "text_muted": (160, 160, 180),
    "accent_amber": (255, 183, 3),
    "neon_cyan": (33, 158, 188),
    "danger_red": (224, 49, 49),
    "fog_shroud": (18, 18, 24),
    "food_gold": (255, 215, 0),
}


class GridCoordinateProjector:
    """Calculates scaling, offset, and bounds for rendering 2D simulation grids."""

    def __init__(
        self,
        grid_width: int,
        grid_height: int,
        viewport_width: int,
        viewport_height: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> None:
        self.gw = grid_width
        self.gh = grid_height
        self.vw = viewport_width
        self.vh = viewport_height
        self.ox = offset_x
        self.oy = offset_y

        cell_w = viewport_width / max(1, grid_width)
        cell_h = viewport_height / max(1, grid_height)
        self.cell_size = max(2, int(min(cell_w, cell_h)))

        # Center grid inside viewport
        self.start_x = offset_x + int((viewport_width - (self.cell_size * grid_width)) / 2)
        self.start_y = offset_y + int((viewport_height - (self.cell_size * grid_height)) / 2)

    def grid_to_pixel(self, gx: int, gy: int) -> tuple[int, int, int, int]:
        """Converts grid coordinate (gx, gy) to pixel rect (px, py, width, height)."""
        px = self.start_x + (gx * self.cell_size)
        py = self.start_y + (gy * self.cell_size)
        return (px, py, self.cell_size, self.cell_size)

    def pixel_to_grid(self, px: int, py: int) -> tuple[int, int] | None:
        """Converts mouse pixel click (px, py) to grid coordinate (gx, gy) if within bounds."""
        rel_x = px - self.start_x
        rel_y = py - self.start_y
        if rel_x < 0 or rel_y < 0:
            return None
        gx = rel_x // self.cell_size
        gy = rel_y // self.cell_size
        if 0 <= gx < self.gw and 0 <= gy < self.gh:
            return (int(gx), int(gy))
        return None
