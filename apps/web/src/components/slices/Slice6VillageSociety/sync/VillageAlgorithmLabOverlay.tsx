import React, { useState } from 'react';
import { VillageStatePayloadDTO } from '../../../../types/village';
import { BookOpen, Network, Activity, Scale, TrendingUp } from 'lucide-react';

interface VillageAlgorithmLabOverlayProps {
  state: VillageStatePayloadDTO;
}

export const VillageAlgorithmLabOverlay: React.FC<VillageAlgorithmLabOverlayProps> = ({ state }) => {
  const [selectedAlgo, setSelectedAlgo] = useState<'PAGERANK' | 'PAD_ODE' | 'UTILITY' | 'AUCTION'>('PAGERANK');

  const socialGraph = state.social_graph || { leaderboard: [], edges: [], pagerank_residual: 0 };
  const leaderboard = socialGraph.leaderboard || [];

  return (
    <div className="space-y-6 text-slate-100 font-sans max-w-5xl mx-auto">
      {/* 1. Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-black/40 border border-white/10 backdrop-blur-md">
        <div>
          <div className="flex items-center gap-2">
            <BookOpen className="text-cyan-400" size={20} />
            <h2 className="text-lg font-bold text-white tracking-wide">
              Village Algorithm Lab · Mathematical Inspect & Duel Arena
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time scratch algorithm telemetry powering the 3D autonomous villagers
          </p>
        </div>

        {/* Algorithm Selector Buttons */}
        <div className="flex items-center gap-1.5 overflow-x-auto">
          <button
            onClick={() => setSelectedAlgo('PAGERANK')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition ${
              selectedAlgo === 'PAGERANK'
                ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/60 shadow-sm'
                : 'bg-white/5 hover:bg-white/10 text-slate-300 border border-white/5'
            }`}
          >
            <Network size={14} /> Social PageRank
          </button>
          <button
            onClick={() => setSelectedAlgo('PAD_ODE')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition ${
              selectedAlgo === 'PAD_ODE'
                ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/60 shadow-sm'
                : 'bg-white/5 hover:bg-white/10 text-slate-300 border border-white/5'
            }`}
          >
            <Activity size={14} /> 3D PAD Dynamics
          </button>
          <button
            onClick={() => setSelectedAlgo('UTILITY')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition ${
              selectedAlgo === 'UTILITY'
                ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/60 shadow-sm'
                : 'bg-white/5 hover:bg-white/10 text-slate-300 border border-white/5'
            }`}
          >
            <Scale size={14} /> Multi-Objective Utility
          </button>
          <button
            onClick={() => setSelectedAlgo('AUCTION')}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 transition ${
              selectedAlgo === 'AUCTION'
                ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/60 shadow-sm'
                : 'bg-white/5 hover:bg-white/10 text-slate-300 border border-white/5'
            }`}
          >
            <TrendingUp size={14} /> Double Auction
          </button>
        </div>
      </div>

      {/* 2. Algorithm 1: Social Capital PageRank */}
      {selectedAlgo === 'PAGERANK' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-1 p-5 rounded-2xl bg-black/35 border border-white/10 space-y-4">
            <h3 className="text-xs uppercase font-mono tracking-wider text-cyan-400 font-bold">
              Algorithm Definition & Math
            </h3>
            <div className="p-3.5 rounded-xl bg-slate-900/80 border border-cyan-500/20 font-mono text-xs space-y-2">
              <div className="text-cyan-300 font-bold">Power Iteration Equation:</div>
              <div className="bg-black/50 p-2 rounded text-[11px] text-amber-300">
                s_(t+1) = d · T^T s_t + ((1-d)/N) · 1
              </div>
              <div className="text-[10px] text-slate-400">
                Damping d = 0.85 · Residual: {socialGraph.pagerank_residual?.toExponential(2) || '< 1e-6'}
              </div>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Every peer interaction, tithe, and trade modifies directed trust edges. Power iteration computes the stationary distribution ranking village status.
            </p>
          </div>

          <div className="md:col-span-2 p-5 rounded-2xl bg-black/35 border border-white/10 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono uppercase text-slate-400">
              <span>Societal Status Leaderboard (PageRank S_i)</span>
              <span>{leaderboard.length} Nodes</span>
            </div>
            <div className="space-y-2 max-h-[340px] overflow-y-auto pr-1">
              {leaderboard.map((item, idx) => {
                const villager = state.villagers?.find((v) => v.agent_id === item.agent_id);
                const pct = (item.status * 100).toFixed(1);
                return (
                  <div
                    key={item.agent_id}
                    className="p-3 rounded-xl bg-slate-900/50 border border-white/5 flex items-center justify-between"
                  >
                    <div className="flex items-center gap-3">
                      <span className="w-5 font-mono text-xs text-cyan-400 font-bold">#{idx + 1}</span>
                      <div>
                        <div className="text-xs font-semibold text-white">{villager?.name || item.agent_id}</div>
                        <div className="text-[10px] text-slate-400">{villager?.role || 'Citizen'}</div>
                      </div>
                    </div>
                    <div className="w-48 flex items-center gap-3">
                      <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div className="h-full bg-cyan-400 rounded-full" style={{ width: `${Math.min(100, Number(pct) * 5)}%` }} />
                      </div>
                      <span className="font-mono text-xs text-cyan-300 font-bold w-12 text-right">{pct}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* 3. Algorithm 2: PAD Dynamical System */}
      {selectedAlgo === 'PAD_ODE' && (
        <div className="p-6 rounded-2xl bg-black/35 border border-white/10 space-y-5">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Activity className="text-purple-400" size={18} />
              <span>Affective 3D PAD Emotional Trajectories (Continuous ODE System)</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Differential equation: dE_i(t)/dt = -λ_i (E_i(t) - E_base,i) + Σ_k δ(t - t_k) · ΔE_k
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-emerald-500/20">
              <div className="text-emerald-400 font-bold text-xs uppercase font-mono">Pleasure Axis (P)</div>
              <div className="text-xs text-slate-300 mt-2">
                Boosted by eating (+0.3), successful fair trade (+0.2), and community praise. Suppressed by starvation (-0.4) and pillory lockup (-0.8).
              </div>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-amber-500/20">
              <div className="text-amber-400 font-bold text-xs uppercase font-mono">Arousal Axis (A)</div>
              <div className="text-xs text-slate-300 mt-2">
                Elevated during theft attempts (+0.6), drought emergencies (+0.5), and Gram Sabha assemblies (+0.4). Decays to baseline rest during sleep.
              </div>
            </div>
            <div className="p-4 rounded-xl bg-slate-900/60 border border-purple-500/20">
              <div className="text-purple-400 font-bold text-xs uppercase font-mono">Dominance Axis (D)</div>
              <div className="text-xs text-slate-300 mt-2">
                High status (S_i &gt; 0.15) and wealth yield high dominance (+0.6). Being placed in the pillory crushes dominance (-0.9).
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 4. Algorithm 3: Multi-Objective Utility */}
      {selectedAlgo === 'UTILITY' && (
        <div className="p-6 rounded-2xl bg-black/35 border border-white/10 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Scale className="text-cyan-400" size={18} />
            <span>Multi-Objective Pareto Social Utility Optimizer</span>
          </h3>
          <div className="p-4 rounded-xl bg-slate-900/80 font-mono text-xs text-amber-300 space-y-2">
            <div>U(a) = w_phys · u_phys(a) + w_econ · u_econ(a) + w_soc · u_soc(a) - I(a ∈ Unethical) · θ_risk · P_detect · κ_penalty</div>
            <div className="text-[11px] text-slate-400 font-sans">
              Where:
              <br />• <b>u_phys(a)</b>: Hunger and hydration delta relative to starvation crisis threshold τ.
              <br />• <b>u_econ(a)</b>: Expected coin clearing value from trading or harvesting.
              <br />• <b>u_soc(a)</b>: PageRank centrality delta ΔS_i gained from tithes or lost from ostracism.
              <br />• <b>θ_risk · P_detect · κ_penalty</b>: Expected penal deterrence from guard arrest and pillory sentencing.
            </div>
          </div>
        </div>
      )}

      {/* 5. Algorithm 4: Double Auction */}
      {selectedAlgo === 'AUCTION' && (
        <div className="p-6 rounded-2xl bg-black/35 border border-white/10 space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <TrendingUp className="text-amber-400" size={18} />
            <span>Double-Auction Marketplace Equilibrium Clearing</span>
          </h3>
          <p className="text-xs text-slate-400">
            Bids (buyers seeking water/food) and asks (merchants/farmers selling surplus) are sorted into supply and demand curves. The market clearing price p* is computed where supply meets demand.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs font-mono">
            {Object.entries(state.economy.clearing_prices).slice(0, 8).map(([item, price]) => (
              <div key={item} className="p-3 rounded-xl bg-slate-900/60 border border-white/5">
                <div className="text-slate-400 uppercase text-[10px]">{item}</div>
                <div className="text-amber-300 font-bold text-sm mt-0.5">{price.toFixed(2)} coins</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
