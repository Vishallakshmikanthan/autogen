import { useEffect, useRef, useState, useCallback } from 'react';
import { ViewerStateDTO, StreamFrameEvent } from '../types/viewer';

interface UseSimulationStreamResult {
  viewerState: ViewerStateDTO | null;
  isConnected: boolean;
  fps: number;
  sendCommand: (command: 'PLAY' | 'PAUSE' | 'STEP' | 'RESET' | 'SPEED', params?: { speed?: number }) => void;
}

export function useSimulationStream(simId: string | null): UseSimulationStreamResult {
  const [viewerState, setViewerState] = useState<ViewerStateDTO | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [fps, setFps] = useState<number>(0);

  const wsRef = useRef<WebSocket | null>(null);
  const stateRef = useRef<ViewerStateDTO | null>(null);
  const frameCountRef = useRef<number>(0);
  const lastFpsCalcRef = useRef<number>(performance.now());

  stateRef.current = viewerState;

  useEffect(() => {
    if (!simId) {
      setViewerState(null);
      setIsConnected(false);
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/api/v1/simulations/${simId}/stream`;

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data: StreamFrameEvent = JSON.parse(event.data);
        if (data.event_type !== 'FRAME') return;

        // Calculate FPS
        frameCountRef.current += 1;
        const now = performance.now();
        if (now - lastFpsCalcRef.current >= 1000) {
          setFps(Math.round((frameCountRef.current * 1000) / (now - lastFpsCalcRef.current)));
          frameCountRef.current = 0;
          lastFpsCalcRef.current = now;
        }

        const delta = data.delta;
        if (delta.is_keyframe && delta.full_state) {
          setViewerState(delta.full_state);
        } else if (stateRef.current) {
          const prevState = stateRef.current;
          const nextState: ViewerStateDTO = {
            ...prevState,
            step: delta.step,
            state_hash: delta.state_hash,
            decision_trace: delta.decision_trace !== undefined ? delta.decision_trace : prevState.decision_trace,
          };

          // Apply updated agents
          if (delta.updated_agents && delta.updated_agents.length > 0) {
            const agentMap = new Map(prevState.agents.map((a) => [a.agent_id, a]));
            for (const ag of delta.updated_agents) {
              agentMap.set(ag.agent_id, ag);
            }
            nextState.agents = Array.from(agentMap.values());
          }

          // Apply modified cells
          if (delta.modified_cells && delta.modified_cells.length > 0) {
            const cellMap = new Map(prevState.grid_cells.map((c) => [`${c.x},${c.y}`, c]));
            for (const c of delta.modified_cells) {
              cellMap.set(`${c.x},${c.y}`, c);
            }
            nextState.grid_cells = Array.from(cellMap.values());
          }

          setViewerState(nextState);
        }
      } catch (err) {
        console.error('Failed to parse simulation stream frame', err);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [simId]);

  const sendCommand = useCallback((command: 'PLAY' | 'PAUSE' | 'STEP' | 'RESET' | 'SPEED', params?: { speed?: number }) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command, ...params }));
    }
  }, []);

  return { viewerState, isConnected, fps, sendCommand };
}

