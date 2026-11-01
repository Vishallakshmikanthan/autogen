import { APIResponseEnvelope, EnvironmentDTO, AgentDTO, SimulationStatusDTO, SimulationStateResponseDTO } from '../types/api';
import { VillageStatePayloadDTO, VillageExperimentDTO } from '../types/village';

const API_BASE = '/api/v1';

export async function fetchEnvironments(): Promise<EnvironmentDTO[]> {
  const res = await fetch(`${API_BASE}/environments`);
  const json: APIResponseEnvelope<EnvironmentDTO[]> = await res.json();
  return json.data;
}

export async function createEnvironment(payload: {
  name: string;
  width: number;
  height: number;
  obstacle_density: number;
  resource_density: number;
}): Promise<EnvironmentDTO> {
  const res = await fetch(`${API_BASE}/environments`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const json: APIResponseEnvelope<EnvironmentDTO> = await res.json();
  return json.data;
}

export async function fetchAgents(): Promise<AgentDTO[]> {
  const res = await fetch(`${API_BASE}/agents`);
  const json: APIResponseEnvelope<AgentDTO[]> = await res.json();
  return json.data;
}

export async function createSimulation(payload: {
  name: string;
  seed: number;
  width: number;
  height: number;
  max_steps: number;
  agents?: Array<{ agent_id: string; agent_type: string }>;
}): Promise<SimulationStatusDTO> {
  const res = await fetch(`${API_BASE}/simulations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  const json: APIResponseEnvelope<SimulationStatusDTO> = await res.json();
  return json.data;
}

export async function sendSimulationControl(simId: string, action: 'start' | 'pause' | 'step' | 'reset'): Promise<void> {
  await fetch(`${API_BASE}/simulations/${simId}/${action}`, {
    method: 'POST',
  });
}

export async function fetchSimulationState(simId: string): Promise<SimulationStateResponseDTO> {
  const res = await fetch(`${API_BASE}/simulations/${simId}/state`);
  const json: APIResponseEnvelope<SimulationStateResponseDTO> = await res.json();
  return json.data;
}

export async function fetchVillageState(): Promise<VillageStatePayloadDTO> {
  const res = await fetch(`${API_BASE}/village/state`);
  const json: APIResponseEnvelope<VillageStatePayloadDTO> = await res.json();
  return json.data;
}

export async function stepVillageSimulation(steps = 1, timeOfDayOverride?: number): Promise<VillageStatePayloadDTO> {
  const res = await fetch(`${API_BASE}/village/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      steps,
      ...(timeOfDayOverride !== undefined ? { time_of_day_override: timeOfDayOverride } : {}),
    }),
  });
  const json: APIResponseEnvelope<VillageStatePayloadDTO> = await res.json();
  return json.data;
}

export async function resetVillageSimulation(payload: {
  seed?: number;
  drought_active?: boolean;
  enable_llm_persona?: boolean;
} = {}): Promise<VillageStatePayloadDTO> {
  const res = await fetch(`${API_BASE}/village/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      seed: payload.seed ?? 42,
      width: 40,
      height: 40,
      drought_active: payload.drought_active ?? false,
      enable_llm_persona: payload.enable_llm_persona ?? false,
    }),
  });
  const json: APIResponseEnvelope<VillageStatePayloadDTO> = await res.json();
  return json.data;
}

export async function toggleVillageDrought(active: boolean): Promise<VillageStatePayloadDTO> {
  const res = await fetch(`${API_BASE}/village/drought?active=${active}`, {
    method: 'POST',
  });
  const json: APIResponseEnvelope<VillageStatePayloadDTO> = await res.json();
  return json.data;
}

export async function fetchVillageExperiments(): Promise<VillageExperimentDTO[]> {
  const res = await fetch(`${API_BASE}/village/experiments`);
  const json: APIResponseEnvelope<VillageExperimentDTO[]> = await res.json();
  return json.data;
}


