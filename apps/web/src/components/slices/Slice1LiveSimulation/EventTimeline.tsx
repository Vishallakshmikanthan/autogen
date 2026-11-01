import React from 'react';
import { Terminal, ShieldAlert, CheckCircle, ArrowRight } from 'lucide-react';

export interface SimulationEventItem {
  id: string;
  step: number;
  type: 'MOVE' | 'COLLECT' | 'COLLISION' | 'VETO' | 'DEATH';
  message: string;
  timestamp: string;
}

interface EventTimelineProps {
  events: SimulationEventItem[];
}

export const EventTimeline: React.FC<EventTimelineProps> = ({ events }) => {
  const getIcon = (type: SimulationEventItem['type']) => {
    switch (type) {
      case 'COLLECT':
        return <CheckCircle size={12} className="text-emerald-400" />;
      case 'COLLISION':
      case 'VETO':
        return <ShieldAlert size={12} className="text-rose-400" />;
      default:
        return <ArrowRight size={12} className="text-cyan-400" />;
    }
  };

  return (
    <div className="bg-research-panel border border-research-border rounded-lg p-3 font-mono text-xs flex flex-col h-48">
      <div className="flex items-center space-x-2 border-b border-slate-800 pb-2 mb-2 text-slate-300">
        <Terminal size={14} className="text-slate-400" />
        <span className="font-semibold text-slate-200">EVENT TELEMETRY TIMELINE</span>
      </div>

      <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
        {events.length === 0 ? (
          <div className="text-slate-500 italic text-center py-8">
            Awaiting tick events...
          </div>
        ) : (
          events.slice(-15).reverse().map((ev) => (
            <div key={ev.id} className="flex items-center space-x-2 text-[11px] py-0.5 border-b border-slate-900">
              <span className="text-slate-500 text-[10px]">T+{ev.step}</span>
              {getIcon(ev.type)}
              <span className="text-slate-300 flex-1 truncate">{ev.message}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

