export interface VillagerUtilityWeightsDTO {
  w_phys: number;
  w_econ: number;
  w_soc: number;
  theta_risk: number;
  crisis_threshold_tau: number;
  crisis_penalty_kappa: number;
  epsilon: number;
}

export interface TopCandidateDTO {
  action: string;
  expected_u: number;
  risk_penalty: number;
  is_unethical: boolean;
}

export interface VillagerDecisionTraceDTO {
  agent_id: string;
  step_id: number;
  timestamp_ns: number;
  algorithm_name: string;
  decision_type: string;
  metrics: Record<string, number | string>;
  rationale: {
    village_action?: string;
    strategic_goal_action?: string;
    target_zone?: string;
    is_unethical?: boolean;
    expected_total_utility?: number;
    u_phys_current?: number;
    u_phys_projected?: number;
    u_econ_projected?: number;
    u_soc_projected?: number;
    detection_probability?: number;
    expected_risk_penalty?: number;
    best_unethical_utility?: number;
    rationale_note?: string;
    top_candidates?: TopCandidateDTO[];
  };
}

export interface VillagerCardDTO {
  agent_id: string;
  name: string;
  role: string;
  position: { x: number; y: number };
  pos_3d: [number, number];
  home_3d: [number, number];
  hunger: number;
  hydration: number;
  energy: number;
  health: number;
  coins: number;
  inventory: Record<string, number>;
  secret_chest_inventory: Record<string, number>;
  societal_status: number;
  status_pct: number;
  in_pillory_ticks: number;
  crimes_committed: number;
  crimes_detected: number;
  tithes_contributed: number;
  last_action: string;
  last_utterance?: string;
  current_goal_zone: string;
  utility_weights: VillagerUtilityWeightsDTO;
  believed_detection_risk: number;
  episodic_memories: string[];
  decision_trace: VillagerDecisionTraceDTO | null;
}

export interface VillageZoneDTO {
  zone_id: string;
  zone_type: string;
  label: string;
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
  center: { x: number; y: number };
  world_3d: [number, number];
  radius_3d: number;
}

export interface VillageDialogueEventDTO {
  step: number;
  speaker_id: string;
  speaker_name: string;
  speaker_role: string;
  recipient_id: string | null;
  recipient_name: string | null;
  subject_id: string | null;
  event_type: 'SpeechEvent' | 'GossipEvent' | 'PublicDeclaration' | 'DiaryEntry';
  utterance: string;
  action_chosen: string;
  is_unethical: boolean;
  mode: string;
  position: [number, number];
  trust_score: number;
}

export interface SocialGraphPayloadDTO {
  agent_ids: string[];
  status_map: Record<string, number>;
  leaderboard: Array<{ agent_id: string; status: number }>;
  edges: Array<{ source: string; target: string; trust: number }>;
  pagerank_iterations: number;
  pagerank_residual: number;
}

export interface VillageStatePayloadDTO {
  step: number;
  state_hash: string;
  time_of_day: number;
  ambient_light: number;
  drought_active: boolean;
  grid: { width: number; height: number };
  zones: VillageZoneDTO[];
  economy: {
    clearing_prices: Record<string, number>;
    granary_storage: Record<string, number>;
    well_water_level: number;
    recent_trades: Array<{
      step: number;
      buyer_id: string;
      seller_id: string;
      commodity: string;
      quantity: number;
      clearing_unit_price: number;
      total_coins: number;
    }>;
  };
  villagers: VillagerCardDTO[];
  social_graph: SocialGraphPayloadDTO;
  dialogue_events: VillageDialogueEventDTO[];
  step_outcomes: Array<{
    step: number;
    actor_id: string;
    target_id: string | null;
    action_type: string;
    category: string;
    commodity: string | null;
    quantity: number;
    detection_probability: number;
    detected: boolean;
    description: string;
  }>;
  cinematic_export: Record<string, unknown>;
}

export interface VillageExperimentDTO {
  scenario_id: string;
  title: string;
  hypothesis: string;
  seeds_evaluated: number;
  metrics: Record<string, number>;
  series: Array<Record<string, number | string>>;
  conclusion: string;
  limitations: string;
}

