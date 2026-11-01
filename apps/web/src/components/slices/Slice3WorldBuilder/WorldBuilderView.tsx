import React, { useState } from 'react';
import { createEnvironment, createSimulation } from '../../../api/client';
import { Sliders, Sparkles, CheckCircle2 } from 'lucide-react';

interface WorldBuilderViewProps {
  onSimulationCreated: (simId: string) => void;
}

export const WorldBuilderView: React.FC<WorldBuilderViewProps> = ({ onSimulationCreated }) => {
  const [name, setName] = useState('custom_arena');
  const [width, setWidth] = useState(20);
  const [height, setHeight] = useState(20);
  const [obstacleDensity, setObstacleDensity] = useState(0.15);
  const [resourceDensity, setResourceDensity] = useState(0.10);
  const [seed, setSeed] = useState(42);
  const [loading, setLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);

  const handleGenerateAndSpawn = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setStatusMsg(null);
    try {
      // 1. Create environment
      await createEnvironment({
        name,
        width,
        height,
        obstacle_density: obstacleDensity,
        resource_density: resourceDensity,
      });

      // 2. Spawn simulation session
      const sim = await createSimulation({
        name: `sim_${name}`,
        seed,
        width,
        height,
        max_steps: 200,
        agents: [
          { agent_id: 'explorer_1', agent_type: 'simple_reflex' },
          { agent_id: 'random_1', agent_type: 'random' },
        ],
      });

      setStatusMsg(`World created successfully! Sim ID: ${sim.id}`);
      onSimulationCreated(sim.id);
    } catch (err) {
      console.error('World generation failed:', err);
      setStatusMsg('Error creating world. Check backend API connection.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-research-panel border border-research-border rounded-lg p-6 font-mono text-xs max-w-2xl mx-auto space-y-6">
      <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
        <Sliders size={18} className="text-cyan-400" />
        <h2 className="text-sm font-semibold text-slate-100">WORLD BUILDER & PROCEDURAL GENERATOR</h2>
      </div>

      <form onSubmit={handleGenerateAndSpawn} className="space-y-4">
        {/* Environment Name */}
        <div>
          <label className="block text-slate-400 text-[11px] mb-1">Environment Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500"
            required
          />
        </div>

        {/* Grid Dimensions */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-slate-400 text-[11px] mb-1">Width ({width} cells)</label>
            <input
              type="range"
              min="8"
              max="40"
              value={width}
              onChange={(e) => setWidth(parseInt(e.target.value))}
              className="w-full accent-cyan-500"
            />
          </div>
          <div>
            <label className="block text-slate-400 text-[11px] mb-1">Height ({height} cells)</label>
            <input
              type="range"
              min="8"
              max="40"
              value={height}
              onChange={(e) => setHeight(parseInt(e.target.value))}
              className="w-full accent-cyan-500"
            />
          </div>
        </div>

        {/* Density Parameters */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-slate-400 text-[11px] mb-1">
              Obstacle Density: {(obstacleDensity * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.0"
              max="0.4"
              step="0.02"
              value={obstacleDensity}
              onChange={(e) => setObstacleDensity(parseFloat(e.target.value))}
              className="w-full accent-rose-500"
            />
          </div>
          <div>
            <label className="block text-slate-400 text-[11px] mb-1">
              Resource Density: {(resourceDensity * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.02"
              max="0.3"
              step="0.02"
              value={resourceDensity}
              onChange={(e) => setResourceDensity(parseFloat(e.target.value))}
              className="w-full accent-emerald-500"
            />
          </div>
        </div>

        {/* Random Seed */}
        <div>
          <label className="block text-slate-400 text-[11px] mb-1">Deterministic Seed</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
            className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-1.5 text-slate-200 focus:outline-none focus:border-cyan-500"
          />
        </div>

        {/* Status Message */}
        {statusMsg && (
          <div className="flex items-center space-x-2 text-cyan-400 bg-cyan-950/40 border border-cyan-800/40 p-2.5 rounded">
            <CheckCircle2 size={14} />
            <span>{statusMsg}</span>
          </div>
        )}

        {/* Action Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-2.5 px-4 bg-cyan-600 hover:bg-cyan-500 text-black font-semibold rounded transition flex items-center justify-center space-x-2 disabled:opacity-50"
        >
          <Sparkles size={16} />
          <span>{loading ? 'GENERATING...' : 'GENERATE WORLD & INITIALIZE SIMULATION'}</span>
        </button>
      </form>
    </div>
  );
};
