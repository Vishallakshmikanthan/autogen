import React, { useState, useEffect } from 'react';
import { CanvasRenderer } from './CanvasRenderer';
import { AgentInspector } from './AgentInspector';
import { EventTimeline, SimulationEventItem } from './EventTimeline';
import { ControlToolbar } from '../../common/ControlToolbar';
import { useSimulationStream } from '../../../api/useSimulationStream';
import { sendSimulationControl } from '../../../api/client';
import { Radio } from 'lucide-react';

interface LiveSimulationViewProps {
  simId: string | null;
}

export const LiveSimulationView: React.FC<LiveSimulationViewProps> = ({ simId }) => {
  const { viewerState, isConnected, fps, sendCommand } = useSimulationStream(simId);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [speed, setSpeed] = useState<number>(1.0);
  const [events, setEvents] = useState<SimulationEventItem[]>([]);

  // Automatically select first alive agent if none selected
  useEffect(() => {
    if (viewerState && viewerState.agents.length > 0 && !selectedAgentId) {
      const firstAlive = viewerState.agents.find((a) => a.is_alive) || viewerState.agents[0];
      setSelectedAgentId(firstAlive.agent_id);
    }
  }, [viewerState, selectedAgentId]);

  // Log events on step update
  useEffect(() => {
    if (viewerState && viewerState.decision_trace) {
      const trace = viewerState.decision_trace;
      const newEv: SimulationEventItem = {
        id: `${viewerState.step}_${trace.agent_id}_${Math.random()}`,
        step: viewerState.step,
        type: trace.decision_type.includes('veto') ? 'VETO' : 'MOVE',
        message: `${trace.agent_id}: ${trace.algorithm_name} -> ${trace.decision_type}`,
        timestamp: new Date().toISOString(),
      };
      setEvents((prev) => [...prev.slice(-40), newEv]);
    }
  }, [viewerState?.step]);

  const selectedAgent = viewerState?.agents.find((a) => a.agent_id === selectedAgentId) || null;
  const isPaused = viewerState?.time_machine.is_paused ?? true;

  const handlePlay = () => {
    if (simId) {
      sendSimulationControl(simId, 'start');
      sendCommand('PLAY');
    }
  };

  const handlePause = () => {
    if (simId) {
      sendSimulationControl(simId, 'pause');
      sendCommand('PAUSE');
    }
  };

  const handleStep = () => {
    if (simId) {
      sendSimulationControl(simId, 'step');
      sendCommand('STEP');
    }
  };

  const handleReset = () => {
    if (simId) {
      sendSimulationControl(simId, 'reset');
      sendCommand('RESET');
      setEvents([]);
    }
  };

  const handleSpeedChange = (newSpeed: number) => {
    setSpeed(newSpeed);
    sendCommand('SPEED', { speed: newSpeed });
  };

  return (
    <div className="flex flex-col space-y-4">
      {/* Simulation Telemetry Bar & Controls */}
      <div className="flex items-center justify-between bg-research-panel border border-research-border px-4 py-2.5 rounded-lg font-mono text-xs">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-1.5">
            <Radio size={12} className={isConnected ? 'text-emerald-400 animate-pulse' : 'text-slate-500'} />
            <span className={isConnected ? 'text-emerald-400' : 'text-slate-500'}>
              {isConnected ? 'LIVE STREAM' : 'DISCONNECTED'}
            </span>
          </div>

          <div className="text-slate-400">
            STEP: <span className="text-cyan-400 font-semibold">{viewerState?.step ?? 0}</span>
          </div>

          <div className="text-slate-400">
            FPS: <span className="text-slate-200 font-semibold">{fps}</span>
          </div>

          <div className="text-slate-400 truncate max-w-xs">
            HASH: <span className="text-slate-500">{viewerState?.state_hash ? viewerState.state_hash.slice(0, 16) + '...' : 'none'}</span>
          </div>
        </div>

        <ControlToolbar
          isPaused={isPaused}
          speed={speed}
          onPlay={handlePlay}
          onPause={handlePause}
          onStep={handleStep}
          onReset={handleReset}
          onSpeedChange={handleSpeedChange}
          disabled={!simId}
        />
      </div>

      {/* Main Grid & Panels Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left: 2D Canvas Viewport */}
        <div className="lg:col-span-8 flex justify-center">
          <CanvasRenderer
            viewerState={viewerState}
            selectedAgentId={selectedAgentId}
            onSelectAgent={setSelectedAgentId}
            width={580}
            height={580}
          />
        </div>

        {/* Right: Agent Inspector and Event Timeline */}
        <div className="lg:col-span-4 flex flex-col space-y-4">
          <AgentInspector
            agent={selectedAgent}
            agentList={viewerState?.agents || []}
            selectedAgentId={selectedAgentId}
            onSelectAgent={setSelectedAgentId}
          />

          <EventTimeline events={events} />
        </div>
      </div>
    </div>
  );
};
