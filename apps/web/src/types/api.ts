export interface APIResponseEnvelope<T> {
  success: boolean;
  data: T;
  message: string;
}

export interface EnvironmentDTO {
  id: string;
  name: string;
  width: number;
  height: number;
  version: number;
  version_id: string;
  config: {
    width?: number;
    height?: number;
    obstacle_density?: number;
    resource_density?: number;
    energy_max?: number;
  };
  config_hash: string;
}

export interface AgentDTO {
  id: string;
  name: string;
  agent_type: string;
  version: number;
  version_id: string;
  policy_params: Record<string, any>;
}

export interface SimulationStatusDTO {
  id: string;
  name: string;
  status: 'CREATED' | 'RUNNING' | 'PAUSED' | 'FINISHED' | 'STOPPED' | 'ERROR';
  current_step: number;
  max_steps: number;
  seed: number;
  state_hash: string;
  env_version_id?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export interface SimulationStateResponseDTO {
  sim_id: string;
  step: number;
  state_hash: string;
  snapshot: any;
}

