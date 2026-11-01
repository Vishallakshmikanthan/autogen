import React from 'react';
import { DecisionTraceDTO } from '../../../types/viewer';
import { ShieldCheck, Compass, Brain, Cpu, Clock } from 'lucide-react';

interface DecisionTraceViewerProps {
  trace: DecisionTraceDTO | null | undefined;
}

export const DecisionTraceViewer: React.FC<DecisionTraceViewerProps> = ({ trace }) => {
  if (!trace) {
    return (
      <div className="bg-research-panel border border-research-border rounded-lg p-6 text-center font-mono text-xs text-slate-500 italic">
        No active decision trace available for the selected agent. Advance the simulation to observe deliberation.
      </div>
    );
  }

  const rationale = trace.rationale || {};

  return (
    <div className="bg-research-panel border border-research-border rounded-lg p-4 font-mono text-xs space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <div className="flex items-center space-x-2">
          <Brain size={16} className="text-research-belief" />
          <span className="font-semibold text-slate-200">GLASS-BOX DECISION TRACE</span>
        </div>
        <div className="flex items-center space-x-2 text-slate-400 text-[11px]">
          <Clock size={12} />
          <span>{trace.computation_time_ms ? `${trace.computation_time_ms.toFixed(2)} ms` : '< 1 ms'}</span>
        </div>
      </div>

      {/* Subsystem Decision Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {/* Navigation Layer */}
        <div className="bg-slate-900/60 border border-slate-800 rounded p-3 space-y-1.5">
          <div className="flex items-center space-x-1.5 text-cyan-400 font-semibold text-[11px]">
            <Compass size={14} />
            <span>NAVIGATION / PATH</span>
          </div>
          <div className="text-slate-300">
            Algorithm: <span className="text-slate-100 font-medium">{trace.algorithm_name}</span>
          </div>
          <div className="text-slate-400 text-[11px]">
            Action Target: <span className="text-emerald-400">{trace.decision_type}</span>
          </div>
        </div>

        {/* Cognitive & Probabilistic Reasoning */}
        <div className="bg-slate-900/60 border border-slate-800 rounded p-3 space-y-1.5">
          <div className="flex items-center space-x-1.5 text-purple-400 font-semibold text-[11px]">
            <Cpu size={14} />
            <span>COGNITIVE RATIONALE</span>
          </div>
          <div className="text-slate-300 truncate">
            Decision: <span className="text-slate-100 font-medium">{trace.decision_type}</span>
          </div>
          <div className="text-slate-400 text-[11px]">
            Agent ID: <span className="text-purple-300">{trace.agent_id}</span>
          </div>
        </div>

        {/* Safety Shield Verification */}
        <div className="bg-slate-900/60 border border-slate-800 rounded p-3 space-y-1.5">
          <div className="flex items-center space-x-1.5 text-rose-400 font-semibold text-[11px]">
            <ShieldCheck size={14} />
            <span>SAFETY SHIELD VETO</span>
          </div>
          <div className="text-slate-300">
            Veto Status: <span className={rationale.veto_triggered ? 'text-rose-400 font-bold' : 'text-emerald-400'}>
              {rationale.veto_triggered ? 'VETO TRIGGERED' : 'CLEAR (NO VETO)'}
            </span>
          </div>
          {rationale.veto_reason && (
            <div className="text-rose-300 text-[10px] truncate">
              Reason: {rationale.veto_reason}
            </div>
          )}
        </div>
      </div>

      {/* Raw Rationale Key-Value Inspector */}
      <div className="bg-slate-950 border border-slate-800 rounded p-3">
        <span className="text-[10px] text-slate-500 uppercase tracking-wider block mb-1">Deliberation Rationale Data</span>
        <pre className="text-[11px] text-slate-300 overflow-x-auto whitespace-pre-wrap leading-relaxed">
          {JSON.stringify(rationale, null, 2)}
        </pre>
      </div>
    </div>
  );
};

