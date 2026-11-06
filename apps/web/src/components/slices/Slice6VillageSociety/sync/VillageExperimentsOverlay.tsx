import React from 'react';
import { VillageExperimentDTO } from '../../../../types/village';
import { FlaskConical, Play, CheckCircle2, TrendingUp } from 'lucide-react';

interface VillageExperimentsOverlayProps {
  experiments: VillageExperimentDTO[] | null;
  onRunExperiments: () => void;
  running: boolean;
}

export const VillageExperimentsOverlay: React.FC<VillageExperimentsOverlayProps> = ({
  experiments,
  onRunExperiments,
  running,
}) => {
  return (
    <div className="space-y-6 text-slate-100 font-sans max-w-5xl mx-auto">
      {/* 1. Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-black/40 border border-white/10 backdrop-blur-md">
        <div>
          <div className="flex items-center gap-2">
            <FlaskConical className="text-cyan-400" size={20} />
            <h2 className="text-lg font-bold text-white tracking-wide">
              Village Empirical Research & Experiment Suite
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Phase 6 empirical ablations: Drought stress tests, guard deterrence, and lineage inheritance
          </p>
        </div>

        <button
          onClick={onRunExperiments}
          disabled={running}
          className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center gap-2 transition disabled:opacity-50 shadow-lg shadow-cyan-500/20"
        >
          {running ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-slate-900 border-t-transparent rounded-full animate-spin" />
              <span>Simulating Scenarios across Seeds...</span>
            </>
          ) : (
            <>
              <Play size={14} className="fill-current" />
              <span>Run Empirical Scenarios</span>
            </>
          )}
        </button>
      </div>

      {/* 2. Results List or Placeholder */}
      {experiments && experiments.length > 0 ? (
        <div className="space-y-4">
          {experiments.map((exp) => (
            <div
              key={exp.scenario_id}
              className="p-5 rounded-2xl bg-black/35 border border-white/10 space-y-4"
            >
              <div className="flex items-start justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs text-cyan-400 uppercase font-bold">
                      {exp.scenario_id}
                    </span>
                    <span className="text-slate-400">·</span>
                    <h3 className="text-sm font-bold text-white">{exp.title}</h3>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">{exp.hypothesis}</p>
                </div>
                <div className="px-2.5 py-1 rounded-xl text-xs font-semibold flex items-center gap-1.5 border bg-cyan-500/20 text-cyan-300 border-cyan-400/40 font-mono">
                  <CheckCircle2 size={13} />
                  <span>{exp.seeds_evaluated} Seeds Evaluated</span>
                </div>
              </div>

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
                {Object.entries(exp.metrics).map(([k, val]) => (
                  <div key={k} className="p-3 rounded-xl bg-slate-900/60 border border-white/5">
                    <div className="text-slate-400 text-[10px] uppercase truncate">{k.replace(/_/g, ' ')}</div>
                    <div className="text-cyan-300 font-bold text-sm mt-0.5">
                      {typeof val === 'number' ? val.toFixed(3) : String(val)}
                    </div>
                  </div>
                ))}
              </div>

              {/* Conclusion */}
              <div className="p-3 rounded-xl bg-slate-900/40 border border-cyan-500/20 text-xs text-slate-300 flex items-start gap-2">
                <TrendingUp size={15} className="text-cyan-400 mt-0.5 flex-shrink-0" />
                <div>
                  <span className="font-bold text-cyan-300">Empirical Finding: </span>
                  {exp.conclusion}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="p-12 text-center rounded-2xl bg-black/25 border border-white/10 space-y-3">
          <FlaskConical size={36} className="mx-auto text-slate-600" />
          <h3 className="text-sm font-bold text-slate-300">No empirical experiment run active</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Click <b>"Run Empirical Scenarios"</b> to execute automated multi-seed ablation tests comparing drought resilience, guard deterrence invariants, and generational wealth transmission.
          </p>
        </div>
      )}
    </div>
  );
};
