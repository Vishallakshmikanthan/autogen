"""Pygame simulation viewer with split-screen, decision trace inspector, and time machine."""

from __future__ import annotations

import os
import sys
from typing import Any

from autogen.visualization.adapter import ViewerDataAdapter
from autogen.visualization.contracts import ViewerState
from autogen.visualization.renderer import (
    TERRAIN_PALETTE,
    UI_PALETTE,
    GridCoordinateProjector,
)
from autogen.visualization.time_machine import TimeMachineController


class PygameViewer:
    """Interactive GUI consumer rendering simulation state, decision traces, and time machine."""

    def __init__(
        self,
        adapter: ViewerDataAdapter,
        time_machine: TimeMachineController,
        window_width: int = 1280,
        window_height: int = 720,
        headless: bool = False,
    ) -> None:
        self.adapter = adapter
        self.time_machine = time_machine
        self.window_width = window_width
        self.window_height = window_height
        self.headless = headless

        # Interactive selection and toggles
        self.selected_agent_id: str | None = None
        self.show_fog = True
        self.show_trails = True
        self.show_heatmaps = False
        self.show_collisions = True

        self._pygame: Any = None
        self._screen: Any = None
        self._font_small: Any = None
        self._font_main: Any = None
        self._font_title: Any = None

        if not self.headless:
            self._init_pygame()

    def _init_pygame(self) -> None:
        try:
            import pygame

            self._pygame = pygame
        except ImportError as exc:
            raise ImportError(
                "Pygame is required for GUI rendering. Install it via 'uv pip install pygame'."
            ) from exc

        # Initialize pygame display
        if "SDL_VIDEODRIVER" not in os.environ and sys.platform.startswith("linux"):
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        self._pygame.init()
        self._pygame.font.init()
        try:
            self._screen = self._pygame.display.set_mode((self.window_width, self.window_height))
            self._pygame.display.set_caption("AUTOGEN Research Studio — Simulation Viewer (M05)")
        except Exception:
            # Fallback to dummy driver if display open fails
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self._pygame.display.init()
            self._screen = self._pygame.display.set_mode((self.window_width, self.window_height))

        self._font_small = self._pygame.font.SysFont("monospace", 12)
        self._font_main = self._pygame.font.SysFont("monospace", 14)
        self._font_title = self._pygame.font.SysFont("monospace", 18, bold=True)

    def step(self) -> ViewerState:
        """Processes user input, updates time machine, and renders current frame."""
        state = self.adapter.generate_viewer_state(
            selected_agent_id=self.selected_agent_id,
            time_machine_state=self.time_machine.get_state(),
        )

        if not self.headless and self._screen is not None:
            self._handle_events(state)
            if not self.time_machine.is_paused:
                self.time_machine.step_forward()
                state = self.adapter.generate_viewer_state(
                    selected_agent_id=self.selected_agent_id,
                    time_machine_state=self.time_machine.get_state(),
                )
            self._render_frame(state)
            self._pygame.display.flip()

        return state

    def _handle_events(self, state: ViewerState) -> None:
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                self._pygame.quit()
                sys.exit(0)

            elif event.type == self._pygame.KEYDOWN:
                if event.key == self._pygame.K_SPACE:
                    self.time_machine.toggle_play_pause()
                elif event.key == self._pygame.K_RIGHT:
                    self.time_machine.step_forward()
                elif event.key == self._pygame.K_LEFT:
                    self.time_machine.step_backward()
                elif event.key == self._pygame.K_UP:
                    self.time_machine.set_speed(min(10.0, self.time_machine.playback_speed * 2.0))
                elif event.key == self._pygame.K_DOWN:
                    self.time_machine.set_speed(max(0.25, self.time_machine.playback_speed / 2.0))
                elif event.key == self._pygame.K_f:
                    self.show_fog = not self.show_fog
                elif event.key == self._pygame.K_t:
                    self.show_trails = not self.show_trails
                elif event.key == self._pygame.K_h:
                    self.show_heatmaps = not self.show_heatmaps
                elif event.key == self._pygame.K_c:
                    self.show_collisions = not self.show_collisions
                elif event.key == self._pygame.K_TAB:
                    # Cycle through available agent IDs
                    if state.agents:
                        agent_ids = [a.agent_id for a in state.agents]
                        if self.selected_agent_id not in agent_ids:
                            self.selected_agent_id = agent_ids[0]
                        else:
                            curr_idx = agent_ids.index(self.selected_agent_id)
                            self.selected_agent_id = agent_ids[(curr_idx + 1) % len(agent_ids)]

            elif event.type == self._pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Handle selection click on ground truth viewport
                mx, my = event.pos
                projector = GridCoordinateProjector(
                    state.width, state.height, 460, 460, offset_x=20, offset_y=60
                )
                clicked_grid = projector.pixel_to_grid(mx, my)
                if clicked_grid:
                    for a in state.agents:
                        if (a.x, a.y) == clicked_grid:
                            self.selected_agent_id = a.agent_id
                            break

    def _render_frame(self, state: ViewerState) -> None:
        """Renders ground truth, agent view, decision trace panel, and scrubber."""
        self._screen.fill(UI_PALETTE["bg_dark"])

        # Title bar
        title_surf = self._font_title.render(
            f"AUTOGEN RESEARCH STUDIO | Step {state.step} | Hash {state.state_hash[:8]}",
            True,
            UI_PALETTE["text_main"],
        )
        self._screen.blit(title_surf, (20, 15))

        # Viewport 1: Ground Truth World (Left)
        gt_projector = GridCoordinateProjector(
            state.width, state.height, 440, 440, offset_x=20, offset_y=60
        )
        self._render_grid_viewport(state, gt_projector, is_agent_view=False)

        # Viewport 2: Agent Mind / Fog of War (Right)
        mind_projector = GridCoordinateProjector(
            state.width, state.height, 440, 440, offset_x=480, offset_y=60
        )
        self._render_grid_viewport(state, mind_projector, is_agent_view=True)

        # Panel 3: Decision Trace Inspector (Far Right)
        self._render_decision_trace_panel(state, offset_x=940, offset_y=60, width=320, height=440)

        # Bottom Bar: Time Machine Scrubber & Key Controls
        self._render_scrubber_bar(state, offset_x=20, offset_y=520, width=1240, height=180)

    def _render_grid_viewport(
        self, state: ViewerState, proj: GridCoordinateProjector, is_agent_view: bool
    ) -> None:
        pg = self._pygame
        panel_title = (
            f"AGENT VIEW ({self.selected_agent_id or 'None'})" if is_agent_view else "GROUND TRUTH"
        )
        t_surf = self._font_main.render(panel_title, True, UI_PALETTE["accent_amber"])
        self._screen.blit(t_surf, (proj.ox, proj.oy - 20))

        # Build agent mind lookup if in agent view
        mind_lookup = {}
        if is_agent_view and state.agent_mind:
            mind_lookup = {(m.x, m.y): m for m in state.agent_mind}

        fog_set = set(state.overlays.fog_of_war_visible)

        # Render terrain cells
        for c in state.grid_cells:
            px, py, cw, ch = proj.grid_to_pixel(c.x, c.y)
            rect = pg.Rect(px, py, cw, ch)

            if is_agent_view:
                if (c.x, c.y) in mind_lookup:
                    m_cell = mind_lookup[(c.x, c.y)]
                    base_color = TERRAIN_PALETTE.get(m_cell.terrain, (40, 40, 40))
                    # Modulate brightness by confidence
                    conf = m_cell.confidence
                    color = tuple(int(ch_v * (0.3 + 0.7 * conf)) for ch_v in base_color)
                elif (c.x, c.y) in fog_set:
                    color = TERRAIN_PALETTE.get(c.terrain, (40, 40, 40))
                else:
                    # Unexplored fog of war
                    color = UI_PALETTE["fog_shroud"]
            else:
                color = TERRAIN_PALETTE.get(c.terrain, (40, 40, 40))

            pg.draw.rect(self._screen, color, rect)

            # Food resource indicator
            food_val = (
                mind_lookup[(c.x, c.y)].food
                if (is_agent_view and (c.x, c.y) in mind_lookup)
                else c.food
            )
            if food_val > 0.0 and (
                not is_agent_view or (c.x, c.y) in mind_lookup or (c.x, c.y) in fog_set
            ):
                food_rad = max(2, int(min(cw, ch) * 0.3 * min(1.0, food_val / 10.0)))
                pg.draw.circle(
                    self._screen,
                    UI_PALETTE["food_gold"],
                    (px + cw // 2, py + ch // 2),
                    food_rad,
                )

        # Movement trails overlay
        if self.show_trails and not is_agent_view:
            for _aid, trail in state.overlays.movement_trails.items():
                if len(trail) > 1:
                    points = [
                        (
                            proj.grid_to_pixel(x, y)[0] + proj.cell_size // 2,
                            proj.grid_to_pixel(x, y)[1] + proj.cell_size // 2,
                        )
                        for x, y in trail
                    ]
                    pg.draw.lines(self._screen, UI_PALETTE["neon_cyan"], False, points, 2)

        # Collision markers overlay
        if self.show_collisions and not is_agent_view:
            for cx, cy in state.overlays.collision_markers:
                px, py, cw, ch = proj.grid_to_pixel(cx, cy)
                pg.draw.line(
                    self._screen,
                    UI_PALETTE["danger_red"],
                    (px, py),
                    (px + cw, py + ch),
                    2,
                )
                pg.draw.line(
                    self._screen,
                    UI_PALETTE["danger_red"],
                    (px + cw, py),
                    (px, py + ch),
                    2,
                )

        # Render agents
        for a in state.agents:
            if is_agent_view and (a.x, a.y) not in fog_set and (a.x, a.y) not in mind_lookup:
                continue
            apx, apy, cw, ch = proj.grid_to_pixel(a.x, a.y)
            agent_color = (
                UI_PALETTE["accent_amber"]
                if a.agent_id == self.selected_agent_id
                else UI_PALETTE["neon_cyan"]
            )
            pg.draw.circle(
                self._screen,
                agent_color,
                (apx + cw // 2, apy + ch // 2),
                max(3, cw // 2 - 1),
            )

    def _render_decision_trace_panel(
        self, state: ViewerState, offset_x: int, offset_y: int, width: int, height: int
    ) -> None:
        pg = self._pygame
        panel_rect = pg.Rect(offset_x, offset_y, width, height)
        pg.draw.rect(self._screen, UI_PALETTE["panel_bg"], panel_rect)
        pg.draw.rect(self._screen, UI_PALETTE["border"], panel_rect, 1)

        header = self._font_main.render("DECISION TRACE INSPECTOR", True, UI_PALETTE["neon_cyan"])
        self._screen.blit(header, (offset_x + 10, offset_y + 10))

        trace = state.decision_trace
        y_cursor = offset_y + 35

        if trace is None:
            no_trace = self._font_small.render(
                "Click an agent or press TAB to inspect.", True, UI_PALETTE["text_muted"]
            )
            self._screen.blit(no_trace, (offset_x + 10, y_cursor))
            return

        lines = [
            f"Agent: {trace.agent_id}",
            f"Step: {trace.step_id}",
            f"Mechanism: {trace.algorithm_name}",
            f"Selected Action: {trace.rationale.get('selected_action', 'N/A')}",
            f"Reason: {trace.rationale.get('reason', 'N/A')[:32]}",
            "--- Candidate Scores ---",
        ]
        candidates = trace.rationale.get("candidate_actions", {})
        for act, sc in list(candidates.items())[:6]:
            lines.append(f" {act:8s}: {sc:.2f}")

        mem_used = trace.rationale.get("memory_used", [])
        if mem_used:
            lines.append(f"Memory Items Used: {len(mem_used)}")

        for line in lines:
            txt_surf = self._font_small.render(line, True, UI_PALETTE["text_main"])
            self._screen.blit(txt_surf, (offset_x + 10, y_cursor))
            y_cursor += 18

    def _render_scrubber_bar(
        self, state: ViewerState, offset_x: int, offset_y: int, width: int, height: int
    ) -> None:
        pg = self._pygame
        bar_rect = pg.Rect(offset_x, offset_y, width, height)
        pg.draw.rect(self._screen, UI_PALETTE["panel_bg"], bar_rect)
        pg.draw.rect(self._screen, UI_PALETTE["border"], bar_rect, 1)

        tm = state.time_machine
        status_txt = "PAUSED [SPACE to Play]" if tm.is_paused else "RUNNING [SPACE to Pause]"
        ctrl_str = (
            f"TIME MACHINE | {status_txt} | Speed: {tm.playback_speed}x | "
            f"Step: {tm.current_step} / {tm.total_steps} | Snapshots: {len(tm.available_snapshots)}"
        )
        self._screen.blit(
            self._font_main.render(ctrl_str, True, UI_PALETTE["accent_amber"]),
            (offset_x + 15, offset_y + 15),
        )

        # Scrubber bar track
        track_y = offset_y + 45
        track_w = width - 40
        pg.draw.rect(
            self._screen,
            (50, 50, 70),
            pg.Rect(offset_x + 20, track_y, track_w, 8),
            border_radius=4,
        )

        # Progress fill
        total = max(1, tm.total_steps)
        progress = tm.current_step / total
        fill_w = int(track_w * progress)
        pg.draw.rect(
            self._screen,
            UI_PALETTE["neon_cyan"],
            pg.Rect(offset_x + 20, track_y, fill_w, 8),
            border_radius=4,
        )

        # Key help
        help_lines = [
            "Hotkeys:",
            "  [SPACE] Play/Pause    [RIGHT/LEFT] Step Forward / Rewind    [UP/DOWN] Speed 2x / 0.5x",
            "  [TAB] Next Agent      [F] Toggle Fog-of-War                 [T] Toggle Trails",
            "  [H] Toggle Heatmaps   [C] Toggle Collisions                 [ESC] Exit",
        ]
        hy = offset_y + 65
        for hl in help_lines:
            self._screen.blit(
                self._font_small.render(hl, True, UI_PALETTE["text_muted"]), (offset_x + 20, hy)
            )
            hy += 16
