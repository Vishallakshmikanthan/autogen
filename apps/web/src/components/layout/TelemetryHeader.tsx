import React from 'react';
import { ShieldCheck, Terminal } from 'lucide-react';

interface TelemetryHeaderProps {
  activeSimId: string | null;
}

export const TelemetryHeader: React.FC<TelemetryHeaderProps> = ({ activeSimId }) => {
  return (
    <header className="h-12 bg-research-panel border-b border-research-border px-6 flex items-center justify-between font-mono text-xs">
      <div className="flex items-center space-x-3">
        <span className="text-slate-400">SESSION:</span>
        <span className="px-2 py-0.5 rounded bg-slate-800 text-cyan-400 font-semibold">
          {activeSimId || 'STANDALONE / DEFAULT'}
        </span>
      </div>

      <div className="flex items-center space-x-6 text-[11px] text-slate-400">
        <div className="flex items-center space-x-1.5">
          <ShieldCheck size={14} className="text-emerald-400" />
          <span>SAFETY SHIELD: ACTIVE</span>
        </div>
        <div className="flex items-center space-x-1.5">
          <Terminal size={14} className="text-purple-400" />
          <span>FASTAPI ENGINE: CONNECTED</span>
        </div>
      </div>
    </header>
  );
};
