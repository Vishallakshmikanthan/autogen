import React, { useState } from 'react';
import { VillageStatePayloadDTO } from '../../../../types/village';
import { Sliders, RotateCcw, Download, Sparkles, CheckCircle2 } from 'lucide-react';
import { exportVillageWorldJson } from '../../../../api/client';

interface VillageWorldBuilderOverlayProps {
  state: VillageStatePayloadDTO;
  onResetSimulation: (seed?: number, width?: number, height?: number, drought?: boolean) => void;
  loading: boolean;
}

export const VillageWorldBuilderOverlay: React.FC<VillageWorldBuilderOverlayProps> = ({
  state,
  onResetSimulation,
  loading,
}) => {
  const [seed, setSeed] = useState<number>(42);
  const [width, setWidth] = useState<number>(40);
  const [height, setHeight] = useState<number>(40);
  const [drought, setDrought] = useState<boolean>(state.drought_active || false);
  const [applied, setApplied] = useState<boolean>(false);

  const handleApply = () => {
    onResetSimulation(seed, width, height, drought);
    setApplied(true);
    setTimeout(() => setApplied(false), 3000);
  };

  const handleExportJson = async () => {
    try {
      const data = await exportVillageWorldJson();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `village_world_spec_seed_${seed}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error('Failed to export village world spec:', e);
    }
  };

  return (
    <div className="space-y-6 text-slate-100 font-sans max-w-4xl mx-auto">
      {/* 1. Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-black/40 border border-white/10 backdrop-blur-md">
        <div>
          <div className="flex items-center gap-2">
            <Sliders className="text-cyan-400" size={20} />
            <h2 className="text-lg font-bold text-white tracking-wide">
              3D Village World Builder & Parameter Configurator
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Modify simulation environment constraints, stochastic seed streams, and resource regimes
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleExportJson}
            className="px-3.5 py-1.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-semibold flex items-center gap-1.5 text-slate-200 transition"
          >
            <Download size={14} /> Export village_world.json
          </button>
          <button
            onClick={handleApply}
            disabled={loading}
            className="px-4 py-1.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center gap-1.5 transition disabled:opacity-50 shadow-lg shadow-cyan-500/20"
          >
            {applied ? <CheckCircle2 size={14} /> : <RotateCcw size={14} />}
            {applied ? '3D World Re-initialized!' : 'Re-instantiate 3D World'}
          </button>
        </div>
      </div>

      {/* 2. Configuration Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Spatial & Stochastic Seed Card */}
        <div className="p-5 rounded-2xl bg-black/35 border border-white/10 space-y-4">
          <h3 className="text-xs uppercase font-mono tracking-wider text-cyan-400 font-bold flex items-center gap-1.5">
            <Sparkles size={14} /> Spatial & Seed Determinism
          </h3>

          <div className="space-y-3 text-xs">
            <div>
              <label className="block text-slate-300 mb-1">Random Generator Seed (SeedSequence)</label>
              <input
                type="number"
                min={0}
                max={999999}
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-white/10 text-white font-mono text-xs focus:border-cyan-400 focus:outline-none"
              />
              <p className="text-[11px] text-slate-500 mt-1">
                Guarantees bit-identical deterministic simulation trajectories across repeat runs.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="block text-slate-300 mb-1">Grid Width (cells)</label>
                <input
                  type="number"
                  min={20}
                  max={80}
                  value={width}
                  onChange={(e) => setWidth(Number(e.target.value))}
                  className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-white/10 text-white font-mono text-xs focus:border-cyan-400 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-slate-300 mb-1">Grid Height (cells)</label>
                <input
                  type="number"
                  min={20}
                  max={80}
                  value={height}
                  onChange={(e) => setHeight(Number(e.target.value))}
                  className="w-full px-3 py-2 rounded-xl bg-slate-900 border border-white/10 text-white font-mono text-xs focus:border-cyan-400 focus:outline-none"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Environmental & Resource Regimes Card */}
        <div className="p-5 rounded-2xl bg-black/35 border border-white/10 space-y-4">
          <h3 className="text-xs uppercase font-mono tracking-wider text-cyan-400 font-bold flex items-center gap-1.5">
            <Sliders size={14} /> Environmental & Resource Pressures
          </h3>

          <div className="space-y-3 text-xs">
            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-white/5 space-y-2">
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-semibold text-slate-200">Drought & Famine Pressure</div>
                  <div className="text-[11px] text-slate-400">Restricts well water regeneration to 2.5L/step</div>
                </div>
                <input
                  type="checkbox"
                  checked={drought}
                  onChange={(e) => setDrought(e.target.checked)}
                  className="w-4 h-4 rounded text-cyan-500 cursor-pointer"
                />
              </div>
            </div>

            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-white/5 space-y-2 text-slate-300">
              <div className="font-semibold text-slate-200">Starting Commons Storage</div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div>Wheat: 75.0 bu</div>
                <div>Water: 50.0 bbl</div>
                <div>Timber: 20.0 logs</div>
                <div>Tools: 6.0 units</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

