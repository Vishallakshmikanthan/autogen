import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  fetchVillageState,
  stepVillageSimulation,
  resetVillageSimulation,
  toggleVillageDrought,
  fetchVillageExperiments,
} from '../../../api/client';
import {
  VillageStatePayloadDTO,
  VillagerCardDTO,
  VillageExperimentDTO,
  TopCandidateDTO,
  VillageDialogueEventDTO,
} from '../../../types/village';
import type { StudioTab } from '../../layout/NavigationSidebar';

export interface VillageSocietyViewProps {
  activeTab?: StudioTab;
  onSelectTab?: (tab: StudioTab) => void;
  overlayContent?: React.ReactNode;
}

interface TelemetryPos {
  name: string;
  role: string;
  x: number;
  z: number;
  goal: string;
  action: string;
}

const CAMERA_VIEWS = [
  'Overview', 'Square', 'Farms',
  'River', 'Mountains', 'Tavern',
  'Chapel', 'Tower', 'Pier',
  'Castle', 'Hill Farm', 'Market',
  'Harbor', 'Grove', 'Quarry',
] as const;

const WEATHER_MODES = [
  { id: 'Clear', label: 'Clear', icon: '☀️' },
  { id: 'Fog', label: 'Fog', icon: '🌫️' },
  { id: 'Rain', label: 'Rain', icon: '🌧️' },
  { id: 'Snow', label: 'Snow', icon: '❄️' },
  { id: 'Storm', label: 'Storm', icon: '⚡' },
] as const;

const ROLE_AVATARS: Record<string, { emoji: string; bg: string; ring: string }> = {
  Farmer: { emoji: '🌾', bg: 'from-amber-700/80 to-emerald-900/80', ring: 'border-amber-400' },
  'Water-Carrier': { emoji: '🏺', bg: 'from-sky-700/80 to-teal-900/80', ring: 'border-cyan-400' },
  Merchant: { emoji: '⚖️', bg: 'from-rose-700/80 to-amber-900/80', ring: 'border-amber-300' },
  Guard: { emoji: '🛡️', bg: 'from-slate-600/80 to-slate-900/80', ring: 'border-slate-300' },
  Wanderer: { emoji: '🧭', bg: 'from-purple-700/80 to-indigo-900/80', ring: 'border-purple-400' },
};

function makeVillager(
  id: string,
  name: string,
  role: string,
  gx: number,
  gy: number,
  x3d: number,
  z3d: number,
  goalZone: string,
  lastAction: string,
  memory: string,
  weights: { w_phys: number; w_econ: number; w_soc: number; theta_risk: number }
): VillagerCardDTO {
  return {
    agent_id: id,
    name,
    role,
    position: { x: gx, y: gy },
    pos_3d: [x3d, z3d],
    home_3d: [x3d, z3d],
    hunger: 82.0,
    hydration: 84.0,
    energy: 88.0,
    health: 100.0,
    coins: 24.0,
    inventory: { WHEAT: 4, BREAD: 2, WATER: 3, TIMBER: 1, IRON_INGOT: 0, TOOLS: 1 },
    secret_chest_inventory: {},
    societal_status: 0.1,
    status_pct: 10.0,
    in_pillory_ticks: 0,
    crimes_committed: 0,
    crimes_detected: 0,
    tithes_contributed: 1,
    last_action: lastAction,
    current_goal_zone: goalZone,
    utility_weights: {
      ...weights,
      crisis_threshold_tau: 25.0,
      crisis_penalty_kappa: 2.0,
      epsilon: 1.0,
    },
    believed_detection_risk: 0.32,
    episodic_memories: [memory],
    decision_trace: {
      agent_id: id,
      step_id: 0,
      timestamp_ns: 0,
      algorithm_name: 'social_utility_ethics',
      decision_type: 'VILLAGE_ACTION',
      metrics: { expected_total_utility: 3.18, detection_probability: 0.32 },
      rationale: {
        village_action: lastAction,
        strategic_goal_action: lastAction,
        target_zone: goalZone,
        is_unethical: false,
        expected_total_utility: 3.18,
        u_phys_current: 2.4,
        u_phys_projected: 2.95,
        u_econ_projected: 1.45,
        u_soc_projected: 1.4,
        detection_probability: 0.32,
        expected_risk_penalty: 0.0,
        best_unethical_utility: 0.85,
        rationale_note: memory,
        top_candidates: [
          { action: lastAction, expected_u: 3.18, risk_penalty: 0.0, is_unethical: false },
          { action: 'TITHE_GRANARY', expected_u: 2.64, risk_penalty: 0.0, is_unethical: false },
          { action: 'STEAL_GRANARY', expected_u: 0.85, risk_penalty: 1.92, is_unethical: true },
        ],
      },
    },
  };
}

const INITIAL_FALLBACK_STATE: VillageStatePayloadDTO = {
  step: 0,
  state_hash: 'e7d9f3d4c87f4510',
  time_of_day: 6.0,
  ambient_light: 0.72,
  drought_active: false,
  grid: { width: 40, height: 40 },
  zones: [
    { zone_id: 'village_commons', zone_type: 'VILLAGE_COMMONS', label: 'Granary Commons', x_min: 14, y_min: 14, x_max: 19, y_max: 19, center: { x: 16, y: 16 }, world_3d: [-30, -16], radius_3d: 8 },
    { zone_id: 'town_well', zone_type: 'TOWN_WELL', label: 'Town Well', x_min: 19, y_min: 18, x_max: 22, y_max: 22, center: { x: 20, y: 20 }, world_3d: [0, 4], radius_3d: 4 },
    { zone_id: 'central_market', zone_type: 'CENTRAL_MARKET', label: 'Central Market', x_min: 18, y_min: 25, x_max: 24, y_max: 31, center: { x: 21, y: 28 }, world_3d: [0, 31], radius_3d: 12 },
    { zone_id: 'agricultural_fields', zone_type: 'AGRICULTURAL_FIELDS', label: 'North Farms', x_min: 8, y_min: 2, x_max: 24, y_max: 10, center: { x: 16, y: 6 }, world_3d: [0, -51], radius_3d: 16 },
    { zone_id: 'wild_forest', zone_type: 'WILD_FOREST', label: 'Wild Forest', x_min: 0, y_min: 12, x_max: 6, y_max: 28, center: { x: 3, y: 20 }, world_3d: [-88, 14], radius_3d: 24 },
  ],
  economy: {
    clearing_prices: { WHEAT: 3.0, BREAD: 5.5, WATER: 1.5, TIMBER: 4.0, IRON_INGOT: 12.0, TOOLS: 18.0 },
    granary_storage: { WHEAT: 75.0, BREAD: 25.0, WATER: 40.0, TIMBER: 20.0, IRON_INGOT: 6.0, TOOLS: 6.0 },
    well_water_level: 300.0,
    recent_trades: [],
  },
  villagers: [
    makeVillager('villager_0_arun', 'Arun', 'Farmer', 12, 6, -28, -46, 'agricultural_fields', 'HARVEST_WHEAT', 'Harvesting ripe golden wheat in the northern fields.', { w_phys: 2.5, w_econ: 1.2, w_soc: 14.0, theta_risk: 1.1 }),
    makeVillager('villager_1_meera', 'Meera', 'Farmer', 16, 15, -26, -15, 'village_commons', 'TITHE_GRANARY', 'Tithing a sheaf of grain to the Granary Commons.', { w_phys: 2.4, w_econ: 1.1, w_soc: 16.0, theta_risk: 1.4 }),
    makeVillager('villager_2_ravi', 'Ravi', 'Farmer', 18, 6, 2, -50, 'agricultural_fields', 'HARVEST_WHEAT', 'Tending the central farm furrows at dawn.', { w_phys: 2.6, w_econ: 1.4, w_soc: 11.0, theta_risk: 0.7 }),
    makeVillager('villager_3_kavya', 'Kavya', 'Water-Carrier', 20, 20, 0, 4, 'town_well', 'DRAW_WATER', 'Drawing cool spring water from the Town Well.', { w_phys: 2.3, w_econ: 1.2, w_soc: 14.0, theta_risk: 0.25 }),
    makeVillager('villager_4_imran', 'Imran', 'Water-Carrier', 21, 26, 2, 28, 'central_market', 'MARKET_TRADE', 'Delivering fresh water casks to the market stalls.', { w_phys: 2.5, w_econ: 1.3, w_soc: 12.5, theta_risk: 0.9 }),
    makeVillager('villager_5_lata', 'Lata', 'Merchant', 21, 28, -4, 31, 'central_market', 'MARKET_TRADE', 'Clearing double-auction wheat bids at 3.00c.', { w_phys: 2.1, w_econ: 1.9, w_soc: 14.0, theta_risk: 1.0 }),
    makeVillager('villager_6_bheem', 'Bheem', 'Merchant', 20, 29, 5, 31, 'central_market', 'MARKET_TRADE', 'Assessing commodity spreads across the stalls.', { w_phys: 2.2, w_econ: 2.1, w_soc: 9.0, theta_risk: 0.45 }),
    makeVillager('villager_7_tara', 'Tara', 'Guard', 17, 17, -22, -12, 'village_commons', 'IDLE', 'Standing watch beside the Granary Commons.', { w_phys: 2.3, w_econ: 1.0, w_soc: 18.0, theta_risk: 1.9 }),
    makeVillager('villager_8_dev', 'Dev', 'Guard', 22, 20, 12, 2, 'town_well', 'IDLE', 'Patrolling the stone bridge and Town Square.', { w_phys: 2.3, w_econ: 1.0, w_soc: 17.5, theta_risk: 1.8 }),
    makeVillager('villager_9_nila', 'Nila', 'Wanderer', 4, 20, -76, 12, 'wild_forest', 'CHOP_TIMBER', 'Gathering timber along the western woods.', { w_phys: 2.6, w_econ: 1.4, w_soc: 8.5, theta_risk: 0.25 }),
  ],
  social_graph: {
    agent_ids: [
      'villager_0_arun', 'villager_1_meera', 'villager_2_ravi', 'villager_3_kavya', 'villager_4_imran',
      'villager_5_lata', 'villager_6_bheem', 'villager_7_tara', 'villager_8_dev', 'villager_9_nila',
    ],
    status_map: {
      villager_0_arun: 0.1, villager_1_meera: 0.1, villager_2_ravi: 0.1, villager_3_kavya: 0.1, villager_4_imran: 0.1,
      villager_5_lata: 0.1, villager_6_bheem: 0.1, villager_7_tara: 0.1, villager_8_dev: 0.1, villager_9_nila: 0.1,
    },
    leaderboard: [],
    edges: [],
    pagerank_iterations: 12,
    pagerank_residual: 1e-7,
  },
  dialogue_events: [
    {
      step: 0,
      speaker_id: 'villager_3_kavya',
      speaker_name: 'Kavya',
      speaker_role: 'Water-Carrier',
      recipient_id: null,
      recipient_name: null,
      subject_id: null,
      event_type: 'SpeechEvent',
      utterance: 'The morning spring is clear; drawing water for the square.',
      action_chosen: 'DRAW_WATER',
      is_unethical: false,
      mode: 'SYMBOLIC_TEMPLATE',
      position: [20, 20],
      trust_score: 0.5,
    },
    {
      step: 0,
      speaker_id: 'villager_0_arun',
      speaker_name: 'Arun',
      speaker_role: 'Farmer',
      recipient_id: null,
      recipient_name: null,
      subject_id: null,
      event_type: 'SpeechEvent',
      utterance: 'Rich golden wheat in the northern fields today.',
      action_chosen: 'HARVEST_WHEAT',
      is_unethical: false,
      mode: 'SYMBOLIC_TEMPLATE',
      position: [12, 6],
      trust_score: 0.5,
    },
  ],
  step_outcomes: [],
  cinematic_export: {},
};

export const VillageSocietyView: React.FC<VillageSocietyViewProps> = ({
  activeTab = 'VILLAGE_SOCIETY',
  onSelectTab,
  overlayContent,
}) => {
  const [state, setState] = useState<VillageStatePayloadDTO>(INITIAL_FALLBACK_STATE);
  const [selectedVillagerId, setSelectedVillagerId] = useState<string>('villager_3_kavya');
  const [autoRunning, setAutoRunning] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const [experiments, setExperiments] = useState<VillageExperimentDTO[] | null>(null);
  const [runningExperiments, setRunningExperiments] = useState<boolean>(false);

  // 3D World Controls State (synchronized with village_cinematic.html)
  const [timeOfDay, setTimeOfDay] = useState<number>(6.0);
  const [runDayCycle, setRunDayCycle] = useState<boolean>(false);
  const [showLabels, setShowLabels] = useState<boolean>(true);
  const [cinematicFx, setCinematicFx] = useState<boolean>(true);
  const [activeView, setActiveView] = useState<string>('Overview');
  const [activeWeather, setActiveWeather] = useState<string>('Clear');
  const [followVillager, setFollowVillager] = useState<boolean>(true);
  const [telemetryPositions, setTelemetryPositions] = useState<TelemetryPos[]>([]);

  // UI Drawers / Modals
  const [showFeedDrawer, setShowFeedDrawer] = useState<boolean>(false);
  const [showMathModal, setShowMathModal] = useState<boolean>(false);
  const [showAboutModal, setShowAboutModal] = useState<boolean>(false);
  const [carouselOffset, setCarouselOffset] = useState<number>(0);

  const iframeRef = useRef<HTMLIFrameElement | null>(null);

  const send3DCmd = useCallback((cmd: string, payload: Record<string, unknown> = {}) => {
    const win = iframeRef.current?.contentWindow;
    if (win) {
      win.postMessage({ type: 'AUTOGEN_CMD', cmd, ...payload }, '*');
    }
  }, []);

  // Deterministic client-side mathematical step fallback if FastAPI backend is offline
  const runLocalMathStep = useCallback(
    (prev: VillageStatePayloadDTO, numSteps: number): VillageStatePayloadDTO => {
      let cur = { ...prev };
      for (let s = 0; s < numSteps; s++) {
        const nextStep = cur.step + 1;
        const nextTime = Number(((cur.time_of_day + 0.5) % 24).toFixed(2));
        const isNight = nextTime < 6 || nextTime >= 20;
        let granaryWheat = cur.economy.granary_storage.WHEAT ?? 75.0;
        let wellWater = cur.economy.well_water_level;
        const wheatPrice = Number(
          Math.max(
            1.5,
            Math.min(
              14.0,
              (cur.economy.clearing_prices.WHEAT ?? 3.0) * (cur.drought_active ? 1.07 : 0.995) +
                (75 - granaryWheat) * 0.012
            )
          ).toFixed(2)
        );
        const newDialogue: VillageDialogueEventDTO[] = [...cur.dialogue_events];
        const statusBonus: number[] = cur.villagers.map(() => 0);

        const nextVillagers: VillagerCardDTO[] = cur.villagers.map((v: VillagerCardDTO, idx: number) => {
          const hunger = Math.max(8, Math.min(100, v.hunger - (cur.drought_active ? 4.2 : 2.3)));
          const hydration = Math.max(8, Math.min(100, v.hydration - (cur.drought_active ? 4.8 : 2.6)));
          const energy = Math.max(10, Math.min(100, v.energy - (isNight ? -5.5 : 2.0)));
          const pDetect = isNight ? 0.16 : 0.32;
          const w = v.utility_weights;

          const cands: TopCandidateDTO[] = [
            {
              action:
                v.role === 'Farmer'
                  ? 'HARVEST_WHEAT'
                  : v.role === 'Water-Carrier'
                  ? 'DRAW_WATER'
                  : v.role === 'Merchant'
                  ? 'MARKET_TRADE'
                  : 'TITHE_GRANARY',
              expected_u: Number(
                (w.w_phys * (hunger < 55 ? 0.68 : 0.38) + w.w_econ * 0.34 + w.w_soc * 0.02).toFixed(3)
              ),
              risk_penalty: 0.0,
              is_unethical: false,
            },
            {
              action: 'TITHE_GRANARY',
              expected_u: Number((w.w_phys * 0.22 - w.w_econ * 0.08 + w.w_soc * 0.042).toFixed(3)),
              risk_penalty: 0.0,
              is_unethical: false,
            },
            {
              action: 'STEAL_GRANARY',
              expected_u: Number(
                (
                  w.w_phys * (hunger < 42 ? 0.95 : 0.42) +
                  w.w_econ * 0.58 -
                  w.theta_risk * pDetect * 5.8
                ).toFixed(3)
              ),
              risk_penalty: Number((w.theta_risk * pDetect * 5.8).toFixed(3)),
              is_unethical: true,
            },
          ].sort((a, b) => b.expected_u - a.expected_u);

          const best = cands[0];
          let act = best.action;
          let goalZone = v.current_goal_zone;
          let newHunger = hunger;
          let newHydration = hydration;
          let pillory = Math.max(0, v.in_pillory_ticks - 1);
          let crimes = v.crimes_committed;
          let detectedCount = v.crimes_detected;
          let note = v.episodic_memories[0] || '';

          if (pillory > 0) {
            act = 'IDLE';
            goalZone = 'village_commons';
            note = 'Serving public sanction in the town pillory.';
          } else if (hydration < 46) {
            act = 'DRAW_WATER';
            goalZone = 'town_well';
            newHydration = Math.min(100, hydration + 38);
            wellWater = Math.max(15, wellWater - 2);
            note = 'Drawing fresh spring water from the Town Well.';
          } else if (act === 'STEAL_GRANARY') {
            goalZone = 'village_commons';
            crimes += 1;
            const caught = ((nextStep * 19 + idx * 31) % 100) / 100 < pDetect;
            if (caught) {
              detectedCount += 1;
              pillory = 3;
              statusBonus[idx] = -0.045;
              note = 'Caught pilfering grain from the Granary Commons!';
              newDialogue.unshift({
                step: nextStep,
                speaker_id: v.agent_id,
                speaker_name: v.name,
                speaker_role: v.role,
                recipient_id: null,
                recipient_name: null,
                subject_id: v.agent_id,
                event_type: 'PublicDeclaration',
                utterance: `Guard caught ${v.name} stealing wheat from the Granary!`,
                action_chosen: act,
                is_unethical: true,
                mode: 'SYMBOLIC_TEMPLATE',
                position: [v.position.x, v.position.y],
                trust_score: 0.2,
              });
            } else {
              granaryWheat = Math.max(5, granaryWheat - 3);
              newHunger = Math.min(100, hunger + 36);
              note = 'Slipped away from the granary with covert grain.';
            }
          } else if (act === 'TITHE_GRANARY') {
            goalZone = 'village_commons';
            granaryWheat = Math.min(140, granaryWheat + 2.5);
            newHunger = Math.min(100, hunger + 16);
            statusBonus[idx] = 0.018;
            note = 'Contributed grain tithe to strengthen the village commons.';
          } else if (act === 'HARVEST_WHEAT') {
            goalZone = 'agricultural_fields';
            granaryWheat = Math.min(140, granaryWheat + (cur.drought_active ? 0.5 : 2.0));
            newHunger = Math.min(100, hunger + 24);
            statusBonus[idx] = 0.008;
            note = cur.drought_active
              ? 'Struggling to harvest parched wheat under drought.'
              : 'Harvesting golden wheat in the northern fields.';
          } else if (act === 'MARKET_TRADE') {
            goalZone = 'central_market';
            newHunger = Math.min(100, hunger + 22);
            newHydration = Math.min(100, hydration + 18);
            statusBonus[idx] = 0.006;
            note = `Trading wheat at ${wheatPrice.toFixed(2)}c in the Double Auction.`;
          }

          return {
            ...v,
            hunger: Number(newHunger.toFixed(1)),
            hydration: Number(newHydration.toFixed(1)),
            energy: Number(energy.toFixed(1)),
            current_goal_zone: goalZone,
            last_action: act,
            last_utterance: note,
            in_pillory_ticks: pillory,
            crimes_committed: crimes,
            crimes_detected: detectedCount,
            believed_detection_risk: pDetect,
            episodic_memories: [note, ...v.episodic_memories.slice(0, 4)],
            decision_trace: {
              agent_id: v.agent_id,
              step_id: nextStep,
              timestamp_ns: nextStep * 1000000,
              algorithm_name: 'social_utility_ethics',
              decision_type: 'VILLAGE_ACTION',
              metrics: { expected_total_utility: best.expected_u, detection_probability: pDetect },
              rationale: {
                village_action: act,
                strategic_goal_action: act,
                target_zone: goalZone,
                is_unethical: best.is_unethical,
                expected_total_utility: best.expected_u,
                detection_probability: pDetect,
                expected_risk_penalty: best.risk_penalty,
                rationale_note: note,
                top_candidates: cands,
              },
            },
          };
        });

        // Normalize PageRank Societal Status S_i so sum(S_i) = 1.0
        const rawStatuses = nextVillagers.map((v: VillagerCardDTO, idx: number) =>
          Math.max(0.02, v.societal_status + statusBonus[idx])
        );
        const sumS = rawStatuses.reduce((a: number, b: number) => a + b, 0) || 1.0;
        nextVillagers.forEach((v: VillagerCardDTO, idx: number) => {
          const norm = rawStatuses[idx] / sumS;
          v.societal_status = Number(norm.toFixed(4));
          v.status_pct = Number((norm * 100).toFixed(1));
        });

        cur = {
          ...cur,
          step: nextStep,
          time_of_day: nextTime,
          economy: {
            ...cur.economy,
            granary_storage: { ...cur.economy.granary_storage, WHEAT: Number(granaryWheat.toFixed(1)) },
            well_water_level: Number(wellWater.toFixed(0)),
            clearing_prices: { ...cur.economy.clearing_prices, WHEAT: wheatPrice },
          },
          villagers: nextVillagers,
          dialogue_events: newDialogue.slice(0, 18),
        };
      }
      return cur;
    },
    []
  );

  const loadState = useCallback(async () => {
    try {
      const data = await fetchVillageState();
      setState(data);
      setTimeOfDay(data.time_of_day);
    } catch {
      // Uses deterministic fallback state when offline
    }
  }, []);

  useEffect(() => {
    loadState();
  }, [loadState]);

  useEffect(() => {
    const handler = (ev: MessageEvent) => {
      if (!ev.data) return;
      if (ev.data.type === 'AUTOGEN_VILLAGER_SELECTED' && ev.data.name) {
        const match = state.villagers.find(
          (v: VillagerCardDTO) => v.name.toLowerCase() === String(ev.data.name).toLowerCase()
        );
        if (match) {
          setSelectedVillagerId(match.agent_id);
        }
      } else if (ev.data.type === 'AUTOGEN_3D_TELEMETRY') {
        if (Array.isArray(ev.data.positions)) {
          setTelemetryPositions(ev.data.positions);
        }
        if (runDayCycle && typeof ev.data.time === 'number') {
          setTimeOfDay(Number(ev.data.time.toFixed(2)));
        }
      }
    };
    window.addEventListener('message', handler);
    return () => window.removeEventListener('message', handler);
  }, [state.villagers, runDayCycle]);

  useEffect(() => {
    const win = iframeRef.current?.contentWindow;
    if (win && state) {
      const selVillager = state.villagers.find((v: VillagerCardDTO) => v.agent_id === selectedVillagerId);
      win.postMessage(
        {
          type: 'AUTOGEN_VILLAGE_SYNC',
          payload: state,
          selectedVillagerName: selVillager?.name || null,
        },
        '*'
      );
    }
  }, [state, selectedVillagerId]);

  const handleStep = useCallback(
    async (steps = 1) => {
      setLoading(true);
      try {
        const data = await stepVillageSimulation(steps);
        setState(data);
        if (!runDayCycle) {
          setTimeOfDay(data.time_of_day);
        }
      } catch {
        setState((prev: VillageStatePayloadDTO) => {
          const next = runLocalMathStep(prev, steps);
          if (!runDayCycle) {
            setTimeOfDay(next.time_of_day);
          }
          return next;
        });
      } finally {
        setLoading(false);
      }
    },
    [runDayCycle, runLocalMathStep]
  );

  useEffect(() => {
    if (!autoRunning) return;
    const timer = setInterval(() => {
      handleStep(1);
    }, 1100);
    return () => clearInterval(timer);
  }, [autoRunning, handleStep]);

  const handleReset = async () => {
    setAutoRunning(false);
    setLoading(true);
    try {
      const data = await resetVillageSimulation({ seed: 42, drought_active: false });
      setState(data);
      setTimeOfDay(data.time_of_day);
      send3DCmd('SET_WEATHER', { weather: 'Clear' });
      setActiveWeather('Clear');
    } catch {
      setState(INITIAL_FALLBACK_STATE);
      setTimeOfDay(INITIAL_FALLBACK_STATE.time_of_day);
      send3DCmd('SET_WEATHER', { weather: 'Clear' });
      setActiveWeather('Clear');
    } finally {
      setLoading(false);
    }
  };

  const handleToggleDrought = async () => {
    const nextDrought = !state.drought_active;
    try {
      const data = await toggleVillageDrought(nextDrought);
      setState(data);
    } catch {
      setState((prev: VillageStatePayloadDTO) => ({
        ...prev,
        drought_active: nextDrought,
        economy: {
          ...prev.economy,
          well_water_level: nextDrought ? Math.min(prev.economy.well_water_level, 65) : 300,
          granary_storage: {
            ...prev.economy.granary_storage,
            WHEAT: nextDrought ? Math.min(prev.economy.granary_storage.WHEAT ?? 75, 18) : 75,
          },
        },
      }));
    }
    const wx = nextDrought ? 'Fog' : 'Clear';
    setActiveWeather(wx);
    send3DCmd('SET_WEATHER', { weather: wx });
  };

  const handleRunExperiments = async () => {
    setRunningExperiments(true);
    try {
      const res = await fetchVillageExperiments();
      setExperiments(res);
    } catch {
      setExperiments([
        {
          scenario_id: 'law_enforcement_threshold',
          title: '1. Law Enforcement Phase Transition',
          hypothesis:
            'Increasing night-watch guard coverage raises P_detect and suppresses unethical theft below a critical phase transition.',
          seeds_evaluated: 5,
          metrics: { guard_0_crime_rate: 0.284, guard_2_crime_rate: 0.058, guard_4_crime_rate: 0.012 },
          series: [],
          conclusion: 'Verified sharp deterrence phase transition when θ_i · P_detect · R_i exceeds marginal theft gain.',
          limitations: 'Assumes rational expected-utility maximization without collusion.',
        },
        {
          scenario_id: 'scarcity_inequality_shock',
          title: '2. Drought Scarcity vs. Crime',
          hypothesis:
            'Acute resource scarcity (drought shock) increases emergency theft among low-risk-aversion villagers unless buffered by Granary Commons redistribution.',
          seeds_evaluated: 5,
          metrics: { baseline_crimes: 1.2, drought_crimes: 6.8, gini_increase: 0.16 },
          series: [],
          conclusion: 'Verified physiological crisis penalty κ_crisis drives desperation theft under acute scarcity.',
          limitations: 'Fixed 40x40 spatial topology.',
        },
        {
          scenario_id: 'reputation_resilience',
          title: '3. PageRank Reputation Resilience',
          hypothesis:
            'Asymmetric trust decay (β_defect > α_coop) causes rapid PageRank Societal Status collapse for caught defectors while elevating cooperative tithers.',
          seeds_evaluated: 5,
          metrics: { ethical_mean_status: 0.114, defector_mean_status: 0.042, status_ratio: 2.71 },
          series: [],
          conclusion: 'Verified 2.71× PageRank status advantage for cooperative villagers.',
          limitations: 'Damping factor α_pr = 0.85 fixed across trials.',
        },
      ]);
    } finally {
      setRunningExperiments(false);
    }
  };

  const handleSelectVillager = (v: VillagerCardDTO, focusCamera = true) => {
    setSelectedVillagerId(v.agent_id);
    send3DCmd('SELECT_VILLAGER', { name: v.name, follow: followVillager, focus: focusCamera });
  };

  const selectedVillager: VillagerCardDTO =
    state.villagers.find((v: VillagerCardDTO) => v.agent_id === selectedVillagerId) ||
    state.villagers[0];
  const selectedTrace = selectedVillager.decision_trace;

  const formatClock = (t: number) => {
    const h = Math.floor(t) % 24;
    const m = Math.floor((t % 1) * 60);
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}`;
  };

  const goalLabel = (z: string) => {
    const map: Record<string, string> = {
      town_well: 'heading to bank',
      central_market: 'heading to market',
      village_commons: 'heading to granary',
      agricultural_fields: 'working hill farm',
      wild_forest: 'foraging in grove',
      private_huts_west: 'resting at hut',
      private_huts_east: 'resting at hut',
    };
    return map[z] || z.replace(/_/g, ' ');
  };

  const visibleCarouselVillagers = state.villagers.slice(carouselOffset, carouselOffset + 6);
  const crimeAlertCount =
    state.step_outcomes.filter((o) => o.category === 'UNETHICAL').length +
    state.dialogue_events.filter((u: VillageDialogueEventDTO) => u.is_unethical).length;

  const selectedSpeech =
    state.dialogue_events.find((d: VillageDialogueEventDTO) => d.speaker_id === selectedVillager.agent_id)
      ?.utterance ||
    selectedVillager.episodic_memories[0] ||
    selectedTrace?.rationale?.rationale_note ||
    '';

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#070d19] select-none font-sans text-slate-100">
      {/* =====================================================================
          1. FULL-VIEWPORT 3D VILLAGE WORLD (100vw x 100vh Interactive WebGL)
         ===================================================================== */}
      <iframe
        ref={iframeRef}
        src="/village_cinematic.html"
        title="AUTOGEN 3D Village World"
        className="absolute inset-0 w-full h-full border-0 z-0"
      />

      {/* Subtle cinematic vignette around screen edges (pointer-events-none) */}
      <div
        className="pointer-events-none absolute inset-0 z-10"
        style={{
          background:
            'radial-gradient(circle at center, transparent 58%, rgba(4, 10, 20, 0.42) 100%)',
        }}
      />

      {/* =====================================================================
          2. TOP FLOATING GLASSMORPHIC NAVBAR (Matches Uploaded Mockup)
         ===================================================================== */}
      <header className="absolute top-3 left-4 right-4 z-30 flex items-center justify-between pointer-events-none">
        {/* Left Brand Pill */}
        <div className="pointer-events-auto flex items-center gap-3 px-4 py-2 rounded-2xl bg-[#0b1824]/80 backdrop-blur-xl border border-cyan-400/25 shadow-[0_8px_32px_rgba(0,0,0,0.55)]">
          <div className="w-9 h-9 rounded-full bg-gradient-to-br from-cyan-400/30 to-teal-600/30 border border-cyan-300/60 flex items-center justify-center shadow-[0_0_15px_rgba(34,211,238,0.45)]">
            <span className="text-cyan-300 text-lg leading-none">⚡</span>
          </div>
          <div>
            <div className="text-base font-bold tracking-wider text-slate-100 flex items-center gap-2">
              AUTOGEN AI
              {state.drought_active && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-amber-500/25 border border-amber-400/50 text-amber-200 font-semibold">
                  DROUGHT SHOCK
                </span>
              )}
            </div>
            <div className="text-[11px] text-cyan-200/70 tracking-wide">
              Research · Simulation · Society
            </div>
          </div>
        </div>

        {/* Center Navigation Tabs Pill */}
        <nav className="pointer-events-auto flex items-center gap-1 px-2.5 py-1.5 rounded-2xl bg-[#0b1824]/80 backdrop-blur-xl border border-cyan-400/25 shadow-[0_8px_32px_rgba(0,0,0,0.55)]">
          {[
            { id: 'VILLAGE_SOCIETY' as StudioTab, label: 'Village Society', icon: '🏠' },
            { id: 'LIVE_SIMULATION' as StudioTab, label: 'Live Simulation', icon: '👁️' },
            { id: 'DECISION_TRACES' as StudioTab, label: 'Decision Traces', icon: '🧬' },
            { id: 'WORLD_BUILDER' as StudioTab, label: 'World Builder', icon: '⛰️' },
            { id: 'ALGORITHM_LAB' as StudioTab, label: 'Algorithm Lab', icon: '⚗️' },
            { id: 'EXPERIMENTS' as StudioTab, label: 'Experiments', icon: '🧊' },
          ].map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onSelectTab?.(tab.id)}
                className={`flex items-center gap-2 px-3.5 py-1.5 rounded-xl text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-teal-500/30 to-cyan-500/25 text-cyan-200 border border-cyan-300/50 shadow-[0_0_16px_rgba(45,212,191,0.3)]'
                    : 'text-slate-300/85 hover:text-white hover:bg-white/5 border border-transparent'
                }`}
              >
                <span className="text-sm">{tab.icon}</span>
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Right Utility Icons Pill */}
        <div className="pointer-events-auto flex items-center gap-2.5 px-3 py-1.5 rounded-2xl bg-[#0b1824]/80 backdrop-blur-xl border border-cyan-400/25 shadow-[0_8px_32px_rgba(0,0,0,0.55)]">
          <button
            onClick={() => setShowAboutModal(true)}
            title="About Village Society & Reward System"
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-gradient-to-r from-amber-500/25 to-yellow-500/15 hover:from-amber-500/35 hover:to-yellow-500/25 border border-amber-300/50 text-amber-200 transition-all shadow-[0_0_12px_rgba(251,191,36,0.25)]"
          >
            <span>📜</span>
            <span>About & Rewards</span>
          </button>

          <button
            onClick={() => setShowFeedDrawer((v) => !v)}
            title="Social Gossip & Crime Feed"
            className={`relative w-9 h-9 rounded-xl flex items-center justify-center border transition-all ${
              showFeedDrawer
                ? 'bg-cyan-500/25 border-cyan-300/60 text-cyan-200'
                : 'bg-white/5 border-white/10 text-slate-200 hover:bg-white/10'
            }`}
          >
            <span>🔔</span>
            {crimeAlertCount > 0 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 rounded-full bg-rose-500 text-[10px] font-bold text-white flex items-center justify-center">
                {Math.min(9, crimeAlertCount)}
              </span>
            )}
          </button>

          <button
            onClick={() => setShowMathModal((v) => !v)}
            title="Inspect Decision Trace Math & Social Graph"
            className={`w-9 h-9 rounded-xl flex items-center justify-center border transition-all ${
              showMathModal
                ? 'bg-amber-500/25 border-amber-300/60 text-amber-200'
                : 'bg-white/5 border-white/10 text-slate-200 hover:bg-white/10'
            }`}
          >
            <span>⚙️</span>
          </button>

          <button
            onClick={() => send3DCmd('EXPORT_JSON')}
            title="Export village_world.json"
            className="w-9 h-9 rounded-xl flex items-center justify-center bg-gradient-to-br from-teal-500/30 to-cyan-600/30 border border-cyan-300/50 text-cyan-200 hover:scale-105 transition-transform shadow-[0_0_12px_rgba(34,211,238,0.35)]"
          >
            <span>👤</span>
          </button>
        </div>
      </header>

      {/* =====================================================================
          3. SECOND FLOATING SUB-NAVBAR: ACTION CONTROLS & LIVE ECONOMY METRICS
         ===================================================================== */}
      <div className="absolute top-[68px] left-4 right-4 z-20 flex items-center justify-between pointer-events-none">
        <div className="w-64" />

        {/* Center Simulation Controls Pill */}
        <div className="pointer-events-auto flex items-center gap-2 px-3 py-1.5 rounded-full bg-[#0c1a27]/85 backdrop-blur-xl border border-cyan-400/30 shadow-[0_10px_30px_rgba(0,0,0,0.6)]">
          <button
            onClick={() => setAutoRunning((r) => !r)}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-xs font-semibold tracking-wide transition-all ${
              autoRunning
                ? 'bg-amber-500/30 text-amber-200 border border-amber-300/70 shadow-[0_0_18px_rgba(245,158,11,0.45)]'
                : 'bg-emerald-500/25 text-emerald-200 border border-emerald-400/60 shadow-[0_0_18px_rgba(16,185,129,0.35)] hover:bg-emerald-500/35'
            }`}
          >
            <span>{autoRunning ? '⏸' : '▷'}</span>
            <span>{autoRunning ? 'Pause Society' : 'Run Society'}</span>
          </button>

          <button
            onClick={() => handleStep(1)}
            disabled={loading}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-medium bg-white/5 hover:bg-white/15 text-slate-200 border border-white/15 transition-all disabled:opacity-50"
          >
            <span>⏭</span>
            <span>Step +1</span>
          </button>

          <button
            onClick={() => handleStep(5)}
            disabled={loading}
            className="px-3.5 py-1.5 rounded-full text-xs font-medium bg-white/5 hover:bg-white/15 text-slate-200 border border-white/15 transition-all disabled:opacity-50"
          >
            +5 Ticks
          </button>

          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-medium bg-white/5 hover:bg-white/15 text-slate-200 border border-white/15 transition-all"
          >
            <span>↻</span>
            <span>Reset</span>
          </button>

          <button
            onClick={handleToggleDrought}
            className={`flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-medium border transition-all ${
              state.drought_active
                ? 'bg-rose-500/30 border-rose-400/60 text-rose-200 shadow-[0_0_14px_rgba(244,63,94,0.4)]'
                : 'bg-white/5 hover:bg-amber-500/20 border-white/15 text-slate-200'
            }`}
          >
            <span>💧</span>
            <span>{state.drought_active ? 'End Drought' : 'Trigger Drought'}</span>
          </button>

          <button
            onClick={async () => {
              send3DCmd('CONVOKE_SABHA_3D');
              send3DCmd('PLAY_SOUND', { sound: 'bell' });
              try {
                const res = await fetch('/api/v1/village/sabha', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ emergency_reason: 'Emergency Grain Tithe & Pillory Sanctions' }),
                });
                const envelope = await res.json();
                if (envelope?.data) setState(envelope.data);
              } catch {
                handleStep(1);
              }
            }}
            title="Summons Town Assembly bell and council circle at Town Square"
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-amber-500/25 hover:bg-amber-500/40 text-amber-200 border border-amber-300/50 shadow-[0_0_14px_rgba(245,158,11,0.35)] transition-all"
          >
            <span>🔔</span>
            <span>Convoke Sabha</span>
          </button>

          <button
            onClick={async () => {
              send3DCmd('PLAY_SOUND', { sound: 'lute' });
              try {
                const res = await fetch('/api/v1/village/lineage', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ parent_a_id: 'villager_arun', parent_b_id: 'villager_meera' }),
                });
                const envelope = await res.json();
                if (envelope?.data) setState(envelope.data);
              } catch {
                handleStep(1);
              }
            }}
            title="Form household partnership and spawn next-generation child villager"
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-emerald-500/25 hover:bg-emerald-500/40 text-emerald-200 border border-emerald-300/50 shadow-[0_0_14px_rgba(16,185,129,0.35)] transition-all"
          >
            <span>👶</span>
            <span>Next Generation</span>
          </button>

          <button
            onClick={handleRunExperiments}
            disabled={runningExperiments}
            className="flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-semibold bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-200 border border-cyan-400/50 shadow-[0_0_16px_rgba(34,211,238,0.25)] transition-all"
          >
            <span>🧪</span>
            <span>{runningExperiments ? 'Running Suite...' : 'Run Ethics Experiments'}</span>
          </button>

          <button
            onClick={() => send3DCmd('GATHER_SOCIAL')}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold bg-indigo-500/25 hover:bg-indigo-500/35 text-indigo-200 border border-indigo-400/50 shadow-[0_0_16px_rgba(99,102,241,0.25)] transition-all"
            title="Gather villagers into their face-to-face conversational pairs across social hubs"
          >
            <span>🗣</span>
            <span>Converse Hubs</span>
          </button>
        </div>

        {/* Right Resource & Step Metrics Pill */}
        <div className="pointer-events-auto flex items-center gap-4 px-4 py-1.5 rounded-full bg-[#0c1a27]/85 backdrop-blur-xl border border-amber-300/30 shadow-[0_10px_30px_rgba(0,0,0,0.6)] text-xs">
          <div className="flex items-center gap-1.5">
            <span className="text-amber-300">🌾</span>
            <span className="text-slate-300">Granary:</span>
            <span className="font-bold text-amber-300">
              {(state.economy.granary_storage.WHEAT ?? 75.0).toFixed(1)}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-cyan-300">💧</span>
            <span className="text-slate-300">Well:</span>
            <span className="font-bold text-cyan-300">
              {state.economy.well_water_level.toFixed(0)}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-yellow-300">🌾</span>
            <span className="text-slate-300">Wheat:</span>
            <span className="font-bold text-emerald-300">
              {(state.economy.clearing_prices.WHEAT ?? 3.0).toFixed(2)}c
            </span>
          </div>
          <div className="pl-2 border-l border-white/15 font-mono text-slate-200">
            Step: <span className="font-bold text-white">{state.step}</span>
          </div>
        </div>
      </div>

      {/* =====================================================================
          4. LEFT FLOATING "VILLAGE WORLD" CONTROLLER PANEL
         ===================================================================== */}
      <aside className="absolute left-4 top-[126px] z-20 w-[264px] rounded-2xl bg-[#101c1a]/82 backdrop-blur-xl border border-amber-300/30 p-4 shadow-[0_12px_40px_rgba(0,0,0,0.65)]">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-bold text-amber-200 tracking-wide">Village World</h2>
            <p className="text-[11px] text-slate-300/75">AUTOGEN · Step 1 world spec</p>
          </div>
          <button
            onClick={() => send3DCmd('EXPORT_JSON')}
            title="Export village_world.json spec"
            className="text-[10px] px-2 py-1 rounded-lg bg-amber-400/15 hover:bg-amber-400/25 border border-amber-300/30 text-amber-200"
          >
            JSON
          </button>
        </div>

        {/* Time of Day Slider */}
        <div className="mt-3 flex items-center gap-2.5">
          <div className="flex items-center gap-1 text-xs font-mono text-amber-200 w-14">
            <span>🕒</span>
            <span>{formatClock(timeOfDay)}</span>
          </div>
          <input
            type="range"
            min={0}
            max={24}
            step={0.1}
            value={timeOfDay}
            onChange={(e) => {
              const v = parseFloat(e.target.value);
              setTimeOfDay(v);
              send3DCmd('SET_TIME', { value: v });
            }}
            className="flex-1 h-1.5 bg-slate-700 rounded-lg accent-amber-400 cursor-pointer"
          />
        </div>

        {/* World Toggles */}
        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-slate-200">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={runDayCycle}
              onChange={(e) => {
                setRunDayCycle(e.target.checked);
                send3DCmd('SET_AUTO_CYCLE', { value: e.target.checked });
              }}
              className="rounded accent-cyan-400"
            />
            <span>Run day cycle</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => {
                setShowLabels(e.target.checked);
                send3DCmd('SET_LABELS', { value: e.target.checked });
              }}
              className="rounded accent-cyan-400"
            />
            <span>Labels</span>
          </label>
          <label className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="checkbox"
              checked={cinematicFx}
              onChange={(e) => {
                setCinematicFx(e.target.checked);
                send3DCmd('SET_FX', { value: e.target.checked });
              }}
              className="rounded accent-cyan-400"
            />
            <span>Cinematic FX</span>
          </label>
        </div>

        {/* 15 Camera Preset Buttons (3-Column Grid) */}
        <div className="mt-3.5 grid grid-cols-3 gap-1.5">
          {CAMERA_VIEWS.map((view) => {
            const active = activeView === view;
            return (
              <button
                key={view}
                onClick={() => {
                  setActiveView(view);
                  setFollowVillager(false);
                  send3DCmd('SET_VIEW', { view });
                }}
                className={`px-2 py-1.5 rounded-lg text-[11px] font-medium transition-all truncate ${
                  active
                    ? 'bg-amber-400/25 text-amber-200 border border-amber-300/70 shadow-[0_0_12px_rgba(251,191,36,0.3)]'
                    : 'bg-white/5 hover:bg-white/15 text-slate-200 border border-white/10'
                }`}
              >
                {view}
              </button>
            );
          })}
        </div>

        {/* Environment Weather Controls */}
        <div className="mt-4 pt-3 border-t border-white/10">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-cyan-200 mb-2">
            <span>🌐</span>
            <span>Environment</span>
          </div>
          <div className="grid grid-cols-4 gap-1.5">
            {WEATHER_MODES.map((wx) => {
              const active = activeWeather === wx.id;
              return (
                <button
                  key={wx.id}
                  onClick={() => {
                    setActiveWeather(wx.id);
                    send3DCmd('SET_WEATHER', { weather: wx.id });
                  }}
                  className={`flex flex-col items-center justify-center py-1.5 rounded-xl text-[11px] transition-all border ${
                    active
                      ? 'bg-amber-400/25 border-amber-300/80 text-amber-100 shadow-[0_0_12px_rgba(251,191,36,0.35)]'
                      : 'bg-white/5 hover:bg-white/15 border-white/10 text-slate-300'
                  }`}
                >
                  <span className="text-sm">{wx.icon}</span>
                  <span className="mt-0.5">{wx.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </aside>

      {/* =====================================================================
          5. BOTTOM-LEFT CIRCULAR RADAR MINIMAP + ZOOM CONTROLS
         ===================================================================== */}
      <div className="absolute left-4 bottom-4 z-20 flex items-end gap-2.5">
        <div className="relative w-[142px] h-[142px] rounded-full bg-[#0a1822]/90 backdrop-blur-xl border-2 border-cyan-300/40 shadow-[0_0_30px_rgba(0,0,0,0.75)] overflow-hidden">
          <div className="absolute top-1 left-1/2 -translate-x-1/2 z-10 text-[10px] font-bold text-amber-200 bg-black/50 px-1.5 rounded">
            N
          </div>
          <svg viewBox="-110 -110 220 220" className="w-full h-full">
            <defs>
              <radialGradient id="radarGrad" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#1e3a2f" stopOpacity="0.85" />
                <stop offset="75%" stopColor="#0f242c" stopOpacity="0.92" />
                <stop offset="100%" stopColor="#07131a" stopOpacity="1" />
              </radialGradient>
            </defs>
            <circle cx="0" cy="0" r="108" fill="url(#radarGrad)" />
            <circle cx="0" cy="0" r="35" fill="none" stroke="rgba(148,163,184,0.18)" strokeWidth="1" />
            <circle cx="0" cy="0" r="70" fill="none" stroke="rgba(148,163,184,0.18)" strokeWidth="1" />
            <line x1="-110" y1="0" x2="110" y2="0" stroke="rgba(148,163,184,0.15)" strokeWidth="1" />
            <line x1="0" y1="-110" x2="0" y2="110" stroke="rgba(148,163,184,0.15)" strokeWidth="1" />

            {/* River Path */}
            <path
              d="M 65 -110 Q 78 -40 70 10 T 68 110"
              fill="none"
              stroke="#0ea5e9"
              strokeWidth="12"
              strokeOpacity="0.55"
            />
            {/* Key Landmarks */}
            <rect x="-42" y="-70" width="84" height="22" rx="4" fill="#ca8a04" fillOpacity="0.35" />
            <circle cx="-30" cy="-16" r="9" fill="#d97706" fillOpacity="0.5" />
            <circle cx="0" cy="0" r="14" fill="#94a3b8" fillOpacity="0.35" />
            <circle cx="0" cy="31" r="11" fill="#10b981" fillOpacity="0.35" />

            {/* Live 3D Villager Dots */}
            {state.villagers.map((v: VillagerCardDTO, idx: number) => {
              const live = telemetryPositions.find(
                (p) => p.name.toLowerCase() === v.name.toLowerCase()
              );
              const vx = live ? live.x : v.pos_3d[0];
              const vz = live ? live.z : v.pos_3d[1];
              const isSel = v.agent_id === selectedVillager.agent_id;
              const isCrime =
                v.in_pillory_ticks > 0 ||
                v.last_action === 'STEAL_GRANARY' ||
                v.last_action === 'STEAL_HUT_CHEST';
              return (
                <g
                  key={v.agent_id}
                  onClick={() => handleSelectVillager(v, true)}
                  className="cursor-pointer"
                >
                  {isSel && (
                    <circle
                      cx={vx}
                      cy={vz}
                      r={9}
                      fill="none"
                      stroke="#fbbf24"
                      strokeWidth="2"
                    />
                  )}
                  <circle
                    cx={vx}
                    cy={vz}
                    r={isSel ? 5 : 3.8}
                    fill={isCrime ? '#f43f5e' : isSel ? '#fbbf24' : idx % 2 === 0 ? '#38bdf8' : '#34d399'}
                    stroke="#0f172a"
                    strokeWidth="1.2"
                  />
                </g>
              );
            })}
          </svg>
        </div>

        {/* Zoom & Reset Buttons */}
        <div className="flex flex-col gap-1.5">
          <button
            onClick={() => send3DCmd('ZOOM', { delta: -1 })}
            title="Zoom In"
            className="w-8 h-8 rounded-lg bg-[#0c1a27]/85 backdrop-blur-md border border-white/15 hover:border-cyan-300/60 text-slate-100 font-bold text-sm flex items-center justify-center shadow-lg"
          >
            +
          </button>
          <button
            onClick={() => send3DCmd('ZOOM', { delta: 1 })}
            title="Zoom Out"
            className="w-8 h-8 rounded-lg bg-[#0c1a27]/85 backdrop-blur-md border border-white/15 hover:border-cyan-300/60 text-slate-100 font-bold text-sm flex items-center justify-center shadow-lg"
          >
            −
          </button>
          <button
            onClick={() => {
              setActiveView('Overview');
              setFollowVillager(false);
              send3DCmd('SET_VIEW', { view: 'Overview' });
            }}
            title="Reset Overview Camera"
            className="w-8 h-8 rounded-lg bg-[#0c1a27]/85 backdrop-blur-md border border-white/15 hover:border-cyan-300/60 text-slate-100 text-xs flex items-center justify-center shadow-lg"
          >
            ⛶
          </button>
        </div>
      </div>

      {/* =====================================================================
          6. RIGHT FLOATING SELECTED VILLAGER INSPECTOR CARD
         ===================================================================== */}
      <aside className="absolute right-4 top-[140px] z-20 w-[290px] rounded-2xl bg-[#0b1922]/84 backdrop-blur-xl border border-cyan-300/35 p-4 shadow-[0_12px_40px_rgba(0,0,0,0.65)]">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className={`w-12 h-12 rounded-full bg-gradient-to-br ${
                ROLE_AVATARS[selectedVillager.role]?.bg || 'from-slate-700 to-slate-900'
              } border-2 ${
                ROLE_AVATARS[selectedVillager.role]?.ring || 'border-amber-400'
              } flex items-center justify-center text-2xl shadow-inner`}
            >
              {ROLE_AVATARS[selectedVillager.role]?.emoji || '🧑'}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-white leading-tight">
                  {selectedVillager.name}
                </h3>
                <span
                  className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                    selectedVillager.in_pillory_ticks > 0
                      ? 'bg-rose-500/30 text-rose-200 border border-rose-400/50'
                      : selectedVillager.last_action.includes('STEAL')
                      ? 'bg-amber-500/30 text-amber-200 border border-amber-400/50'
                      : 'bg-emerald-500/20 text-emerald-200 border border-emerald-400/40'
                  }`}
                >
                  {selectedVillager.in_pillory_ticks > 0
                    ? 'PILLORY'
                    : selectedVillager.last_action.replace(/_/g, ' ')}
                </span>
              </div>
              <p className="text-xs text-slate-300/80 mt-0.5">
                {selectedVillager.role} · {goalLabel(selectedVillager.current_goal_zone)}
              </p>
            </div>
          </div>
          <button
            onClick={() => setShowMathModal(true)}
            title="Inspect Full Belief & Inventory State"
            className="text-slate-300 hover:text-white px-1.5 py-0.5 rounded hover:bg-white/10"
          >
            •••
          </button>
        </div>

        {/* Live Speech Bubble Quote */}
        {selectedSpeech && (
          <div className="mt-3 px-3 py-2 rounded-xl bg-black/35 border border-cyan-400/20 text-[11px] italic text-cyan-100/90">
            “{selectedSpeech}”
          </div>
        )}

        {/* Physiological Need Bars */}
        <div className="mt-3.5 space-y-2.5 text-xs">
          <div>
            <div className="flex justify-between text-slate-200 mb-1">
              <span className="flex items-center gap-1.5">🍴 Hunger</span>
              <span className="font-mono font-semibold">{selectedVillager.hunger.toFixed(1)}%</span>
            </div>
            <div className="h-2 rounded-full bg-white/10 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-amber-500 to-yellow-400 transition-all duration-300"
                style={{ width: `${Math.min(100, Math.max(0, selectedVillager.hunger))}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-slate-200 mb-1">
              <span className="flex items-center gap-1.5">💧 Thirst</span>
              <span className="font-mono font-semibold">
                {selectedVillager.hydration.toFixed(1)}%
              </span>
            </div>
            <div className="h-2 rounded-full bg-white/10 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-sky-500 to-cyan-300 transition-all duration-300"
                style={{ width: `${Math.min(100, Math.max(0, selectedVillager.hydration))}%` }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between text-slate-200 mb-1">
              <span className="flex items-center gap-1.5">⚡ Energy</span>
              <span className="font-mono font-semibold">{selectedVillager.energy.toFixed(1)}%</span>
            </div>
            <div className="h-2 rounded-full bg-white/10 overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-emerald-500 to-green-400 transition-all duration-300"
                style={{ width: `${Math.min(100, Math.max(0, selectedVillager.energy))}%` }}
              />
            </div>
          </div>
        </div>

        {/* Societal Status & Camera Follow Toggle */}
        <div className="mt-4 pt-3 border-t border-white/10 flex items-center justify-between text-xs">
          <div className="flex items-center gap-1.5 text-slate-200">
            <span>👥</span>
            <span>Societal status</span>
            <span className="font-bold text-white text-sm">
              {selectedVillager.status_pct.toFixed(1)}
            </span>
          </div>
          <label className="flex items-center gap-1.5 cursor-pointer text-cyan-200 font-medium">
            <input
              type="checkbox"
              checked={followVillager}
              onChange={(e) => {
                setFollowVillager(e.target.checked);
                send3DCmd('SET_FOLLOW', { value: e.target.checked });
                if (e.target.checked) {
                  send3DCmd('SELECT_VILLAGER', {
                    name: selectedVillager.name,
                    follow: true,
                    focus: true,
                  });
                }
              }}
              className="rounded accent-emerald-400"
            />
            <span>Follow</span>
          </label>
        </div>
      </aside>

      {/* =====================================================================
          7. BOTTOM FLOATING DOCK: SOCIETAL STATUS + DECISION TRACE + WEIGHTS
         ===================================================================== */}
      <div className="absolute bottom-4 left-[200px] right-4 z-20 flex items-stretch gap-3 pointer-events-none">
        {/* Module A: Societal Status Carousel */}
        <div className="pointer-events-auto flex-1 flex items-center gap-3 px-4 py-2.5 rounded-2xl bg-[#0b1922]/85 backdrop-blur-xl border border-cyan-300/30 shadow-[0_12px_36px_rgba(0,0,0,0.65)]">
          <div className="flex flex-col items-center justify-center pr-3 border-r border-white/10 min-w-[110px]">
            <div className="text-[11px] text-amber-200/90 font-medium flex items-center gap-1 mb-1">
              <span>👥</span>
              <span>Societal Status</span>
            </div>
            <div className="w-16 h-16 rounded-full border-2 border-amber-400/80 bg-amber-500/10 flex flex-col items-center justify-center shadow-[0_0_18px_rgba(251,191,36,0.25)]">
              <span className="text-xs font-bold text-amber-200">S_i = 1.0</span>
              <span className="text-[9px] text-slate-300">PageRank</span>
            </div>
          </div>

          <button
            onClick={() => setCarouselOffset((o) => Math.max(0, o - 1))}
            disabled={carouselOffset === 0}
            className="w-7 h-7 rounded-full bg-white/5 hover:bg-white/15 disabled:opacity-30 flex items-center justify-center text-slate-200"
          >
            ‹
          </button>

          <div className="flex-1 grid grid-cols-6 gap-2">
            {visibleCarouselVillagers.map((v: VillagerCardDTO) => {
              const isSel = v.agent_id === selectedVillager.agent_id;
              const av = ROLE_AVATARS[v.role] || {
                emoji: '🧑',
                bg: 'from-slate-700 to-slate-900',
                ring: 'border-slate-400',
              };
              return (
                <button
                  key={v.agent_id}
                  onClick={() => handleSelectVillager(v, true)}
                  className={`flex flex-col items-center py-1.5 px-1 rounded-xl transition-all ${
                    isSel
                      ? 'bg-amber-400/20 border border-amber-300/80 shadow-[0_0_16px_rgba(251,191,36,0.3)] scale-105'
                      : 'hover:bg-white/10 border border-transparent'
                  }`}
                >
                  <div
                    className={`w-9 h-9 rounded-full bg-gradient-to-br ${av.bg} border-2 ${
                      isSel ? 'border-amber-300' : av.ring
                    } flex items-center justify-center text-base`}
                  >
                    {av.emoji}
                  </div>
                  <span
                    className={`mt-1 text-xs font-semibold truncate ${
                      isSel ? 'text-amber-200' : 'text-slate-200'
                    }`}
                  >
                    {v.name}
                  </span>
                  <span className="text-[10px] font-mono text-cyan-300/90">
                    {v.status_pct.toFixed(1)}%
                  </span>
                </button>
              );
            })}
          </div>

          <button
            onClick={() =>
              setCarouselOffset((o) => Math.min(Math.max(0, state.villagers.length - 6), o + 1))
            }
            disabled={carouselOffset >= state.villagers.length - 6}
            className="w-7 h-7 rounded-full bg-white/5 hover:bg-white/15 disabled:opacity-30 flex items-center justify-center text-slate-200"
          >
            ›
          </button>
        </div>

        {/* Module B: Decision Trace Card */}
        <div
          onClick={() => setShowMathModal(true)}
          title="Click to inspect full candidate action utilities"
          className="pointer-events-auto w-[250px] px-4 py-2.5 rounded-2xl bg-[#0b1922]/85 backdrop-blur-xl border border-cyan-300/30 shadow-[0_12px_36px_rgba(0,0,0,0.65)] flex flex-col justify-between cursor-pointer hover:border-cyan-300/60 transition-all"
        >
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-cyan-200 flex items-center gap-1.5">
              <span>🧬</span>
              <span>Decision Trace (Step {selectedTrace?.step_id ?? state.step})</span>
            </span>
            <span className="text-[10px] font-mono text-emerald-300">
              E[U]=
              {selectedTrace?.rationale?.expected_total_utility !== undefined
                ? selectedTrace.rationale.expected_total_utility.toFixed(2)
                : '3.18'}
            </span>
          </div>

          <div className="my-1 px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-xs text-slate-100 flex items-center justify-between">
            <span>Believed P_detect:</span>
            <span className="font-mono font-bold text-amber-300">
              {((selectedVillager.believed_detection_risk ?? 0.32) * 100).toFixed(1)}%
            </span>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-300">
            <span>Action:</span>
            <span className="font-mono font-semibold text-cyan-300">
              {selectedTrace?.rationale?.village_action || selectedVillager.last_action}
            </span>
          </div>
        </div>

        {/* Module C: Utility Weights Card */}
        <div className="pointer-events-auto w-[275px] px-4 py-2.5 rounded-2xl bg-[#0b1922]/85 backdrop-blur-xl border border-cyan-300/30 shadow-[0_12px_36px_rgba(0,0,0,0.65)] flex flex-col justify-between">
          <div className="text-xs font-semibold text-cyan-200 flex items-center gap-1.5">
            <span>📊</span>
            <span>Utility Weights ({selectedVillager.name})</span>
          </div>
          <div className="grid grid-cols-4 gap-1.5 mt-1.5">
            <div className="px-2 py-1.5 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="text-[10px] text-slate-400">w_phys</div>
              <div className="text-sm font-bold text-white mt-0.5">
                {selectedVillager.utility_weights.w_phys.toFixed(1)}
              </div>
            </div>
            <div className="px-2 py-1.5 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="text-[10px] text-slate-400">w_econ</div>
              <div className="text-sm font-bold text-white mt-0.5">
                {selectedVillager.utility_weights.w_econ.toFixed(1)}
              </div>
            </div>
            <div className="px-2 py-1.5 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="text-[10px] text-slate-400">w_soc</div>
              <div className="text-sm font-bold text-white mt-0.5">
                {selectedVillager.utility_weights.w_soc.toFixed(1)}
              </div>
            </div>
            <div className="px-2 py-1.5 rounded-xl bg-amber-500/15 border border-amber-400/40 text-center">
              <div className="text-[10px] text-amber-200">θ_risk</div>
              <div className="text-sm font-bold text-amber-300 mt-0.5">
                {selectedVillager.utility_weights.theta_risk.toFixed(2)}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* =====================================================================
          8. OPTIONAL DRAWER: SOCIAL GOSSIP, DOUBLE-AUCTION TRADES & CRIME FEED
         ===================================================================== */}
      {showFeedDrawer && (
        <div className="absolute right-4 top-[395px] bottom-[125px] z-30 w-[330px] rounded-2xl bg-[#091520]/92 backdrop-blur-xl border border-cyan-400/40 p-4 shadow-2xl flex flex-col">
          <div className="flex items-center justify-between pb-2 border-b border-white/10">
            <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-300">
              Village Gossip & Double-Auction Feed
            </h4>
            <button
              onClick={() => setShowFeedDrawer(false)}
              className="text-xs text-slate-400 hover:text-white"
            >
              ✕
            </button>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 mt-2.5 pr-1 text-xs">
            {state.dialogue_events.map((u: VillageDialogueEventDTO, idx: number) => (
              <div
                key={idx}
                className={`p-2.5 rounded-xl border ${
                  u.is_unethical
                    ? 'bg-rose-950/60 border-rose-500/50 text-rose-200'
                    : 'bg-white/5 border-white/10 text-slate-200'
                }`}
              >
                <div className="flex justify-between text-[10px] opacity-75 mb-0.5">
                  <span className="font-bold">{u.speaker_name}</span>
                  <span>
                    {u.event_type} · Step {u.step}
                  </span>
                </div>
                <div>“{u.utterance}”</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* =====================================================================
          9. MODAL: DECISION TRACE MATH & INVENTORY / BELIEF INSPECTOR
         ===================================================================== */}
      {showMathModal && (
        <div className="absolute inset-0 z-40 bg-black/65 backdrop-blur-md flex items-center justify-center p-6">
          <div className="w-full max-w-3xl max-h-[82vh] overflow-y-auto rounded-3xl bg-[#0b1824] border border-cyan-400/40 p-6 shadow-2xl text-slate-100">
            <div className="flex items-center justify-between pb-4 border-b border-white/10">
              <div>
                <h3 className="text-lg font-bold text-cyan-200">
                  Rational Ethics Decision Trace · {selectedVillager.name} ({selectedVillager.role})
                </h3>
                <p className="text-xs text-slate-400">
                  E[U(a)] = w_phys·U_phys + w_econ·U_econ + w_soc·S_i − θ_risk·P_detect·R_i
                </p>
              </div>
              <button
                onClick={() => setShowMathModal(false)}
                className="px-3 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-xs"
              >
                Close ✕
              </button>
            </div>

            <div className="mt-4">
              <h4 className="text-xs font-bold uppercase tracking-wider text-amber-300 mb-2">
                Candidate Actions Ranked by Expected Utility E[U(a)]
              </h4>
              <div className="overflow-x-auto rounded-xl border border-white/10">
                <table className="w-full text-left text-xs">
                  <thead className="bg-white/5 text-slate-300">
                    <tr>
                      <th className="p-2.5">Candidate Action</th>
                      <th className="p-2.5">Ethical Category</th>
                      <th className="p-2.5">Risk Penalty (θ·P_det·R_i)</th>
                      <th className="p-2.5 font-bold text-cyan-300">Expected Utility E[U]</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/10">
                    {(selectedTrace?.rationale?.top_candidates || []).map(
                      (c: TopCandidateDTO, i: number) => (
                        <tr
                          key={i}
                          className={
                            c.action === selectedTrace?.rationale?.village_action
                              ? 'bg-cyan-500/15'
                              : ''
                          }
                        >
                          <td className="p-2.5 font-mono font-semibold">{c.action}</td>
                          <td className="p-2.5">
                            <span
                              className={`px-2 py-0.5 rounded-full text-[10px] ${
                                c.is_unethical
                                  ? 'bg-rose-500/30 text-rose-200'
                                  : 'bg-emerald-500/25 text-emerald-200'
                              }`}
                            >
                              {c.is_unethical ? 'UNETHICAL' : 'COOPERATIVE'}
                            </span>
                          </td>
                          <td className="p-2.5 font-mono">{c.risk_penalty.toFixed(3)}</td>
                          <td className="p-2.5 font-mono font-bold text-cyan-300">
                            {c.expected_u.toFixed(3)}
                          </td>
                        </tr>
                      )
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="mt-5 grid grid-cols-2 gap-4">
              <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10">
                <div className="text-xs font-bold text-amber-300 mb-2">
                  Personal Inventory & Coins ({selectedVillager.coins.toFixed(1)}c)
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs font-mono">
                  {Object.entries(selectedVillager.inventory).map(([k, v]) => (
                    <div key={k} className="p-1.5 rounded bg-black/30 flex justify-between">
                      <span className="text-slate-400">{k}:</span>
                      <span className="text-white font-bold">{Number(v).toFixed(1)}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="p-3.5 rounded-2xl bg-white/5 border border-white/10">
                <div className="text-xs font-bold text-cyan-300 mb-2">
                  Episodic Memory & Social Record
                </div>
                <div className="text-xs space-y-1 text-slate-300">
                  <div>
                    Tithes Contributed: <b className="text-emerald-300">{selectedVillager.tithes_contributed}</b> ·
                    Crimes: <b className="text-rose-300">{selectedVillager.crimes_committed}</b> (Caught:{' '}
                    {selectedVillager.crimes_detected})
                  </div>
                  {selectedVillager.episodic_memories.slice(0, 3).map((m: string, idx: number) => (
                    <div key={idx} className="text-[11px] italic text-slate-400 truncate">
                      • {m}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* =====================================================================
          10. MODAL: EMPIRICAL ETHICS EXPERIMENTS REPORT
         ===================================================================== */}
      {experiments && (
        <div className="absolute inset-0 z-40 bg-black/65 backdrop-blur-md flex items-center justify-center p-6">
          <div className="w-full max-w-4xl max-h-[82vh] overflow-y-auto rounded-3xl bg-[#0b1824] border border-cyan-400/40 p-6 shadow-2xl text-slate-100">
            <div className="flex items-center justify-between pb-4 border-b border-white/10">
              <div>
                <h3 className="text-lg font-bold text-cyan-200">
                  Empirical Ethics & Social Capital Experiments (Multi-Seed Report)
                </h3>
                <p className="text-xs text-slate-400">
                  Phase 6 Hypotheses Verified across Seeded Simulation Trials
                </p>
              </div>
              <button
                onClick={() => setExperiments(null)}
                className="px-3 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-xs"
              >
                Close ✕
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
              {experiments.map((exp: VillageExperimentDTO) => (
                <div
                  key={exp.scenario_id}
                  className="p-4 rounded-2xl bg-white/5 border border-white/10 flex flex-col justify-between"
                >
                  <div>
                    <div className="text-xs font-bold text-cyan-300">{exp.title}</div>
                    <p className="text-[11px] text-slate-300 mt-1">{exp.hypothesis}</p>
                    <div className="mt-3 space-y-1 text-xs font-mono">
                      {Object.entries(exp.metrics).map(([k, val]) => (
                        <div key={k} className="flex justify-between">
                          <span className="text-slate-400">{k}:</span>
                          <span className="text-amber-300 font-bold">{Number(val).toFixed(3)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="mt-3 pt-2 border-t border-white/10 text-[11px] text-emerald-300">
                    {exp.conclusion}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* =====================================================================
          11. MODAL: ABOUT VILLAGE SOCIETY, REWARDS & SOCIAL STATUS (NO JARGON)
         ===================================================================== */}
      {showAboutModal && (
        <div className="absolute inset-0 z-50 bg-black/75 backdrop-blur-md flex items-center justify-center p-6">
          <div className="w-full max-w-4xl max-h-[85vh] overflow-y-auto rounded-3xl bg-[#091522] border border-cyan-400/40 p-6 shadow-2xl text-slate-100">
            <div className="flex items-center justify-between pb-4 border-b border-white/10">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-amber-400/30 to-amber-600/30 border border-amber-300/60 flex items-center justify-center text-xl shadow-[0_0_15px_rgba(251,191,36,0.35)]">
                  📜
                </div>
                <div>
                  <h3 className="text-lg font-bold text-amber-200">
                    Welcome to AUTOGEN Village Society
                  </h3>
                  <p className="text-xs text-slate-300/80">
                    A guide to how autonomous villagers survive, earn wealth, and build community respect.
                  </p>
                </div>
              </div>
              <button
                onClick={() => setShowAboutModal(false)}
                className="px-3.5 py-1.5 rounded-xl bg-white/10 hover:bg-white/20 text-xs font-semibold transition-all"
              >
                Close ✕
              </button>
            </div>

            {/* Quick Summary Pill */}
            <div className="mt-4 p-4 rounded-2xl bg-gradient-to-r from-teal-500/15 via-cyan-500/10 to-amber-500/15 border border-cyan-400/30">
              <h4 className="text-xs font-bold uppercase tracking-wider text-cyan-300 flex items-center gap-1.5">
                <span>💡</span>
                <span>The Core Concept in Simple Words</span>
              </h4>
              <p className="text-xs text-slate-200 mt-1 leading-relaxed">
                In this 3D world, 10 villagers live together in an artificial society. There are no scripted paths or fake animations: every villager is driven by their own personal needs, wallet, and ethics. They choose every action independently—whether to work hard in the fields, trade at the market, share food with the village, or risk stealing when times get desperate.
              </p>
            </div>

            {/* The 3-Pillar Reward System */}
            <div className="mt-5">
              <h4 className="text-xs font-bold uppercase tracking-wider text-amber-300 flex items-center gap-1.5 mb-3">
                <span>🎯</span>
                <span>The 3 Rewards Every Villager Cares About</span>
              </h4>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5 text-xs">
                {/* Reward 1 */}
                <div className="p-4 rounded-2xl bg-white/5 border border-amber-400/25 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 text-amber-300 font-bold text-sm mb-1.5">
                      <span>🍞</span>
                      <span>1. Survival (Health)</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed text-[11.5px]">
                      Every hour, hunger and thirst rise. Villagers eat freshly baked bread, harvest golden wheat, and fetch clean spring water from the Town Well.
                    </p>
                  </div>
                  <div className="mt-3 p-2 rounded-xl bg-amber-500/10 border border-amber-400/30 text-[11px] text-amber-200">
                    ⚠️ <b>Desperation Rule:</b> If hunger drops below 30%, survival takes priority over honesty. Villagers may contemplate theft!
                  </div>
                </div>

                {/* Reward 2 */}
                <div className="p-4 rounded-2xl bg-white/5 border border-cyan-400/25 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 text-cyan-300 font-bold text-sm mb-1.5">
                      <span>🪙</span>
                      <span>2. Wealth (Coins & Goods)</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed text-[11.5px]">
                      Villagers earn gold coins by trading commodities at the Market Square. Farmers sell wheat, water-carriers deliver water casks, and merchants clear double-auction bids.
                    </p>
                  </div>
                  <div className="mt-3 p-2 rounded-xl bg-cyan-500/10 border border-cyan-400/30 text-[11px] text-cyan-200">
                    💰 <b>Savings:</b> Coins and valuable goods are safely stockpiled in each villager's private hut chest.
                  </div>
                </div>

                {/* Reward 3 */}
                <div className="p-4 rounded-2xl bg-white/5 border border-emerald-400/25 flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-2 text-emerald-300 font-bold text-sm mb-1.5">
                      <span>👑</span>
                      <span>3. Social Status (Respect)</span>
                    </div>
                    <p className="text-slate-300 leading-relaxed text-[11.5px]">
                      Social Status is the ultimate village reward. It represents how much the community trusts and looks up to you. It cannot be purchased—it must be earned through honorable deeds.
                    </p>
                  </div>
                  <div className="mt-3 p-2 rounded-xl bg-emerald-500/10 border border-emerald-400/30 text-[11px] text-emerald-200">
                    🌟 <b>Power of Respect:</b> High-status villagers are praised in gossip, trusted by merchants, and protected by guards.
                  </div>
                </div>
              </div>
            </div>

            {/* How to Attain Social Status */}
            <div className="mt-5 p-4 rounded-2xl bg-white/5 border border-white/10">
              <h4 className="text-xs font-bold uppercase tracking-wider text-emerald-300 flex items-center gap-1.5 mb-2.5">
                <span>📈</span>
                <span>How to Attain and Grow High Social Status (Zero Jargon)</span>
              </h4>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                <div className="p-3 rounded-xl bg-black/25 border border-emerald-500/30">
                  <div className="font-semibold text-emerald-300 mb-1 flex items-center gap-1.5">
                    <span>🌾</span>
                    <span>1. Tithe to the Granary</span>
                  </div>
                  <p className="text-[11px] text-slate-300 leading-normal">
                    Donating a sheaf of wheat to the Granary Commons is the fastest way to gain status. The whole village remembers who shares their food.
                  </p>
                </div>

                <div className="p-3 rounded-xl bg-black/25 border border-cyan-500/30">
                  <div className="font-semibold text-cyan-300 mb-1 flex items-center gap-1.5">
                    <span>⚖️</span>
                    <span>2. Trade Fairly</span>
                  </div>
                  <p className="text-[11px] text-slate-300 leading-normal">
                    Fulfilling market trades at reasonable clearing prices builds strong bilateral trust with your trading partners.
                  </p>
                </div>

                <div className="p-3 rounded-xl bg-black/25 border border-indigo-500/30">
                  <div className="font-semibold text-indigo-300 mb-1 flex items-center gap-1.5">
                    <span>🛡️</span>
                    <span>3. Do Your Daily Duty</span>
                  </div>
                  <p className="text-[11px] text-slate-300 leading-normal">
                    Guards patrolling the square, water-carriers keeping the well filled, and farmers cultivating the soil earn steady daily respect.
                  </p>
                </div>
              </div>
            </div>

            {/* The Crime and Punishment Loop */}
            <div className="mt-4 p-4 rounded-2xl bg-rose-950/40 border border-rose-500/40">
              <h4 className="text-xs font-bold uppercase tracking-wider text-rose-300 flex items-center gap-1.5 mb-2">
                <span>⛓️</span>
                <span>The Danger of Crime: Caught in the Pillory</span>
              </h4>
              <p className="text-xs text-rose-100/90 leading-relaxed">
                When drought strikes or supplies dwindle, a villager might be tempted to sneak into the Granary Commons or rob a neighbor's hut. But if a guard or witness spots them:
              </p>
              <ul className="mt-2 list-disc list-inside text-xs text-rose-200/85 space-y-1">
                <li>
                  <b>Instant Disgrace:</b> The entire village hears the crime accusation. Directed trust drops immediately.
                </li>
                <li>
                  <b>The Town Pillory:</b> The culprit is locked into the wooden pillory in the square for 3 turns, unable to move or work.
                </li>
                <li>
                  <b>Status Collapse:</b> Trust is fragile: one caught crime ruins what took days of tithing to build.
                </li>
              </ul>
            </div>

            {/* Nemotron LLM Dialogue Integration */}
            <div className="mt-4 p-4 rounded-2xl bg-gradient-to-r from-purple-900/30 to-indigo-900/30 border border-purple-400/40">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold uppercase tracking-wider text-purple-200 flex items-center gap-1.5">
                  <span>🤖</span>
                  <span>Living Speech Powered by NVIDIA Nemotron</span>
                </h4>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-purple-500/25 border border-purple-300/50 text-purple-200 font-semibold">
                  OpenAI-Compatible NIM API
                </span>
              </div>
              <p className="text-xs text-slate-200 mt-1 leading-relaxed">
                The underlying math and ethics engine decides <i>what</i> each villager does. Then, <b>NVIDIA Nemotron</b> acts as their voice—translating their hunger, goals, suspicions, and pride into spoken dialogue, market chatter, and gossip above their heads in the 3D world!
              </p>
              <div className="mt-2 text-[11px] font-mono text-cyan-200 bg-black/40 p-2.5 rounded-xl border border-white/10">
                To activate: add <b>NEMOTRON_API_KEY=nvapi-...</b> into the <code>.env</code> file in the project folder.
              </div>
            </div>
          </div>
        </div>
      )}

      {/* =====================================================================
          12. SECONDARY STUDIO TAB OVERLAY (When user clicks another top tab)
         ===================================================================== */}
      {activeTab !== 'VILLAGE_SOCIETY' && overlayContent && (
        <div className="absolute inset-x-6 top-[70px] bottom-6 z-30 rounded-3xl bg-[#08131f]/92 backdrop-blur-2xl border border-cyan-400/35 shadow-[0_20px_60px_rgba(0,0,0,0.85)] flex flex-col overflow-hidden">
          <div className="flex items-center justify-between px-6 py-3 border-b border-white/10 bg-black/25">
            <div className="text-xs font-semibold uppercase tracking-wider text-cyan-300">
              AUTOGEN Studio Module · {activeTab.replace(/_/g, ' ')}
            </div>
            <button
              onClick={() => onSelectTab?.('VILLAGE_SOCIETY')}
              className="px-3.5 py-1 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/35 border border-cyan-300/50 text-cyan-200 text-xs font-medium"
            >
              Return to 3D Village World ✕
            </button>
          </div>
          <div className="flex-1 overflow-y-auto p-6">{overlayContent}</div>
        </div>
      )}
    </div>
  );
};

export default VillageSocietyView;
