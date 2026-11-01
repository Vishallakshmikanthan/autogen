import { APIResponseEnvelope, EnvironmentDTO, AgentDTO, SimulationStatusDTO, SimulationStateResponseDTO } from '../types/api';

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

