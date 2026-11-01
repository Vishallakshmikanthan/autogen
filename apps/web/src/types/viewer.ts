export type TerrainType = 'grassland' | 'forest' | 'desert' | 'water' | 'mountain' | 'obstacle';

export interface ViewerCellState {
  x: number;
  y: number;
  terrain: TerrainType;
  food: number;
  movement_cost: number;
}

export interface ViewerAgentState {
  agent_id: string;
  x: number;
  y: number;
  energy: number;
  health: number;
  carried_food: number;
  is_alive: boolean;
}

export interface DecisionTraceDTO {
  agent_id: string;
  step_id: number;
  algorithm_name: string;
  decision_type: string;
  rationale: Record<string, any>;
  alternatives_considered?: string[];
  computation_time_ms?: number;
}

export interface ViewerAgentMindCell {
  x: number;
  y: number;
  terrain: TerrainType;
  food: number;
  confidence: number;
  last_observed_tick: number;
}

export interface ViewerOverlayData {
  fog_of_war_visible: [number, number][];
  movement_trails: Record<string, [number, number][]>;
  collision_markers: [number, number][];
  resource_intensity: Record<string, number>;
}

export interface ViewerTimeMachineState {
  current_step: number;
  total_steps: number;
  available_snapshots?: number[];
  is_paused: boolean;
  playback_speed: number;
  can_fork: boolean;
}

export interface ViewerStateDTO {
  step: number;
  width: number;
  height: number;
  seed: number;
  state_hash: string;
  grid_cells: ViewerCellState[];
  agents: ViewerAgentState[];
  selected_agent_id?: string | null;
  decision_trace?: DecisionTraceDTO | null;
  agent_mind?: ViewerAgentMindCell[] | null;
  overlays: ViewerOverlayData;
  time_machine: ViewerTimeMachineState;
}

export interface StreamFrameEvent {
  event_type: 'FRAME';
  sim_id: string;
  step: number;
  state_hash: string;
  delta: {
    step: number;
    is_keyframe: boolean;
    state_hash: string;
    full_state?: ViewerStateDTO;
    updated_agents?: ViewerAgentState[];
    modified_cells?: ViewerCellState[];
    decision_trace?: DecisionTraceDTO | null;
  };
}

