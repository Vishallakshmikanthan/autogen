import React from 'react';
import { ViewerAgentState } from '../../../types/viewer';
import { MetricBadge } from '../../common/MetricBadge';
import { Activity, Battery, Heart, MapPin } from 'lucide-react';

interface AgentInspectorProps {
  agent: ViewerAgentState | null;
  agentList: ViewerAgentState[];
  selectedAgentId: string | null;
  onSelectAgent: (id: string) => void;
}

export const AgentInspector: React.FC<AgentInspectorProps> = ({
  agent,
  agentList,
  selectedAgentId,
  onSelectAgent,
}) => {
  return (
    <div className="bg-research-panel border border-research-border rounded-lg p-4 flex flex-col space-y-3 font-mono text-xs">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2">
          <Activity size={14} className="text-research-belief" />
          <span className="font-semibold text-slate-200">AGENT INSPECTOR</span>
        </div>
        {/* Agent Selector Dropdown */}
        <select
          value={selectedAgentId || ''}
          onChange={(e) => onSelectAgent(e.target.value)}
          className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-slate-200 focus:outline-none focus:border-cyan-500"
        >
          {agentList.map((a) => (
            <option key={a.agent_id} value={a.agent_id}>
              {a.agent_id} {a.is_alive ? '' : '(DEAD)'}
            </option>
          ))}
        </select>
      </div>

      {agent ? (
        <div className="space-y-3">
          {/* Coordinates & Status */}
          <div className="flex items-center justify-between text-slate-300">
            <div className="flex items-center space-x-1">
              <MapPin size={12} className="text-slate-500" />
              <span>Position:</span>
              <span className="font-semibold text-cyan-400">({agent.x}, {agent.y})</span>
            </div>
            <span className={`px-2 py-0.5 rounded text-[10px] ${agent.is_alive ? 'bg-emerald-900/40 text-emerald-400' : 'bg-red-900/40 text-red-400'}`}>
              {agent.is_alive ? 'ALIVE' : 'STARVED / DEAD'}
            </span>
          </div>

          {/* Metric Badges Grid */}
          <div className="grid grid-cols-3 gap-2">
            <MetricBadge
              label="Energy"
              value={agent.energy.toFixed(1)}
              unit="/ 100"
              colorScheme="truth"
            />
            <MetricBadge
              label="Health"
              value={agent.health.toFixed(1)}
              unit="/ 100"
              colorScheme="belief"
            />
            <MetricBadge
              label="Carried Food"
              value={agent.carried_food.toFixed(1)}
              unit="units"
              colorScheme="memory"
            />
          </div>

          {/* Vital Bars */}
          <div className="space-y-1.5 pt-1">
            <div>
              <div className="flex justify-between text-[10px] text-slate-400 mb-0.5">
                <span className="flex items-center space-x-1"><Battery size={10} /><span>Energy Reserve</span></span>
                <span>{agent.energy.toFixed(0)}%</span>
              </div>
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-500 transition-all duration-200"
                  style={{ width: `${Math.min(100, Math.max(0, agent.energy))}%` }}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between text-[10px] text-slate-400 mb-0.5">
                <span className="flex items-center space-x-1"><Heart size={10} /><span>Physical Health</span></span>
                <span>{agent.health.toFixed(0)}%</span>
              </div>
              <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan-500 transition-all duration-200"
                  style={{ width: `${Math.min(100, Math.max(0, agent.health))}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="text-slate-500 py-6 text-center italic">
          No agent currently selected.
        </div>
      )}
    </div>
  );
};
