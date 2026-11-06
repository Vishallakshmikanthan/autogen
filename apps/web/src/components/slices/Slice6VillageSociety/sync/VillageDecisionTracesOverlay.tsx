import React, { useState } from 'react';
import {
  VillageStatePayloadDTO,
  VillagerCardDTO,
  VillagerDecisionTraceDTO,
  TopCandidateDTO,
} from '../../../../types/village';
import { Brain, ShieldAlert, Heart, Coins, Award, Eye, AlertTriangle } from 'lucide-react';

interface VillageDecisionTracesOverlayProps {
  state: VillageStatePayloadDTO;
  selectedVillagerId: string;
  onSelectVillager: (id: string) => void;
}

export const VillageDecisionTracesOverlay: React.FC<VillageDecisionTracesOverlayProps> = ({
  state,
  selectedVillagerId,
  onSelectVillager,
}) => {
  const villagers = state.villagers || [];
  const currentVillager: VillagerCardDTO | undefined =
    villagers.find((v) => v.agent_id === selectedVillagerId) || villagers[0];

  const trace: VillagerDecisionTraceDTO | null = currentVillager?.decision_trace || null;
  const rationale = trace?.rationale || {};
  const candidates: TopCandidateDTO[] = rationale.top_candidates || [];

  const [activeSubTab, setActiveSubTab] = useState<'UTILITY' | 'TRUTH_VS_BELIEF' | 'EMOTION_PAD'>('UTILITY');

  if (!currentVillager) {
    return (
      <div className="p-8 text-center text-slate-400 font-mono">
        No active villager telemetry available.
      </div>
    );
  }

  const uPhysCurrent = rationale.u_phys_current ?? 12.5;
  const uPhysProj = rationale.u_phys_projected ?? 13.4;
  const uEconProj = rationale.u_econ_projected ?? 7.6;
  const uSocProj = rationale.u_soc_projected ?? 0.15;
  const expectedTotalU = rationale.expected_total_utility ?? trace?.metrics?.expected_utility ?? 40.77;
  const pDetect = rationale.detection_probability ?? currentVillager.believed_detection_risk ?? 0.0;
  const pad = currentVillager.pad_vector || [0.5, 0.2, 0.4];

  return (
    <div className="space-y-6 text-slate-100 font-sans">
      {/* 1. Header with Villager Selector Pills */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-4 rounded-2xl bg-black/40 border border-white/10 backdrop-blur-md">
        <div>
          <div className="flex items-center gap-2">
            <Brain className="text-cyan-400" size={20} />
            <h2 className="text-lg font-bold text-white tracking-wide">
              3D Villager Decision & Explainability Suite
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time multi-objective social utility breakdown & belief state (Step {state.step})
          </p>
        </div>

        {/* Villager selection pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 max-w-full">
          {villagers.map((v) => {
            const isSelected = v.agent_id === currentVillager.agent_id;
            return (
              <button
                key={v.agent_id}
                onClick={() => onSelectVillager(v.agent_id)}
                className={`px-3 py-1.5 rounded-xl text-xs font-medium transition flex items-center gap-1.5 whitespace-nowrap ${
                  isSelected
                    ? 'bg-cyan-500/30 text-cyan-200 border border-cyan-400/60 shadow-[0_0_12px_rgba(34,211,238,0.35)] font-semibold'
                    : 'bg-white/5 hover:bg-white/10 text-slate-300 border border-white/5'
                }`}
              >
                <span>{v.name}</span>
                <span className="text-[10px] text-slate-400">({v.role})</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* 2. Sub-Tab Switcher */}
      <div className="flex items-center gap-3 border-b border-white/10 pb-2">
        <button
          onClick={() => setActiveSubTab('UTILITY')}
          className={`px-4 py-1.5 rounded-xl text-xs font-semibold tracking-wider transition ${
            activeSubTab === 'UTILITY'
              ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-400/40 shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Social Utility & Candidate Trees
        </button>
        <button
          onClick={() => setActiveSubTab('TRUTH_VS_BELIEF')}
          className={`px-4 py-1.5 rounded-xl text-xs font-semibold tracking-wider transition ${
            activeSubTab === 'TRUTH_VS_BELIEF'
              ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-400/40 shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          Ground Truth vs. Agent Belief (POMDP)
        </button>
        <button
          onClick={() => setActiveSubTab('EMOTION_PAD')}
          className={`px-4 py-1.5 rounded-xl text-xs font-semibold tracking-wider transition ${
            activeSubTab === 'EMOTION_PAD'
              ? 'bg-cyan-500/25 text-cyan-300 border border-cyan-400/40 shadow-sm'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          3D Affective Emotion Dynamics (PAD)
        </button>
      </div>

      {/* 3. Sub-Tab 1: Social Utility & Candidates */}
      {activeSubTab === 'UTILITY' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Multi-Objective Utility Formula */}
          <div className="lg:col-span-1 p-5 rounded-2xl bg-black/35 border border-white/10 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase font-mono tracking-wider text-cyan-400 font-bold">
                Multi-Objective Payoff
              </span>
              <span className="px-2 py-0.5 rounded text-[11px] font-mono bg-cyan-950 text-cyan-300 border border-cyan-800">
                E[U] = {Number(expectedTotalU).toFixed(2)}
              </span>
            </div>

            <div className="p-3 rounded-xl bg-slate-900/80 border border-cyan-500/20 font-mono text-xs text-slate-200 space-y-2">
              <div className="text-[11px] text-cyan-300 font-semibold">Optimization Target:</div>
              <div className="bg-black/50 p-2 rounded text-[11px] text-amber-300 overflow-x-auto">
                U(a) = w_phys · u_phys + w_econ · u_econ + w_soc · u_soc - Penalty
              </div>
            </div>

            {/* Utility Weight Sliders / Display */}
            <div className="space-y-3 pt-2 text-xs">
              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span className="flex items-center gap-1.5">
                    <Heart size={14} className="text-rose-400" /> Physiological Weight (w_phys)
                  </span>
                  <span className="font-mono text-rose-300">{currentVillager.utility_weights.w_phys.toFixed(1)}</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-rose-500 rounded-full" style={{ width: `${Math.min(100, currentVillager.utility_weights.w_phys * 20)}%` }} />
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">Projected u_phys: {Number(uPhysProj).toFixed(2)} (Current: {Number(uPhysCurrent).toFixed(2)})</div>
              </div>

              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span className="flex items-center gap-1.5">
                    <Coins size={14} className="text-amber-400" /> Economic Weight (w_econ)
                  </span>
                  <span className="font-mono text-amber-300">{currentVillager.utility_weights.w_econ.toFixed(1)}</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-amber-500 rounded-full" style={{ width: `${Math.min(100, currentVillager.utility_weights.w_econ * 20)}%` }} />
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">Projected u_econ: {Number(uEconProj).toFixed(2)}</div>
              </div>

              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span className="flex items-center gap-1.5">
                    <Award size={14} className="text-purple-400" /> Social Status Weight (w_soc)
                  </span>
                  <span className="font-mono text-purple-300">{currentVillager.utility_weights.w_soc.toFixed(1)}</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-purple-500 rounded-full" style={{ width: `${Math.min(100, currentVillager.utility_weights.w_soc * 6)}%` }} />
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">Projected u_soc: {Number(uSocProj).toFixed(2)} (PageRank S_i: {currentVillager.status_pct.toFixed(1)}%)</div>
              </div>

              <div>
                <div className="flex justify-between text-slate-300 mb-1">
                  <span className="flex items-center gap-1.5">
                    <ShieldAlert size={14} className="text-red-400" /> Risk Aversion (theta_risk)
                  </span>
                  <span className="font-mono text-red-300">{currentVillager.utility_weights.theta_risk.toFixed(2)}</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div className="h-full bg-red-500 rounded-full" style={{ width: `${Math.min(100, currentVillager.utility_weights.theta_risk * 100)}%` }} />
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">Believed P(detect): {(Number(pDetect) * 100).toFixed(1)}%</div>
              </div>
            </div>
          </div>

          {/* Right Column: Ranked Candidates Tree */}
          <div className="lg:col-span-2 p-5 rounded-2xl bg-black/35 border border-white/10 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs uppercase font-mono tracking-wider text-cyan-400 font-bold">
                Evaluated Action Candidates (Ranked by Expected Utility)
              </span>
              <span className="text-[11px] text-slate-400">
                Action Chosen: <b className="text-emerald-300">{rationale.village_action || currentVillager.last_action}</b>
              </span>
            </div>

            <div className="space-y-2.5">
              {candidates.length > 0 ? (
                candidates.map((c, idx) => {
                  const isWinner = idx === 0;
                  return (
                    <div
                      key={c.action}
                      className={`p-3 rounded-xl border flex items-center justify-between transition ${
                        isWinner
                          ? 'bg-emerald-950/40 border-emerald-500/50 shadow-[0_0_15px_rgba(16,185,129,0.15)]'
                          : c.is_unethical
                          ? 'bg-rose-950/20 border-rose-500/30'
                          : 'bg-slate-900/40 border-white/5'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold font-mono ${
                            isWinner ? 'bg-emerald-500 text-black' : 'bg-slate-800 text-slate-400'
                          }`}
                        >
                          #{idx + 1}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-semibold text-xs tracking-wide text-white">
                              {c.action}
                            </span>
                            {isWinner && (
                              <span className="px-1.5 py-0.2 rounded text-[10px] bg-emerald-500/20 text-emerald-300 font-medium border border-emerald-400/40">
                                WINNER
                              </span>
                            )}
                            {c.is_unethical && (
                              <span className="px-1.5 py-0.2 rounded text-[10px] bg-rose-500/20 text-rose-300 font-medium border border-rose-400/40 flex items-center gap-1">
                                <AlertTriangle size={10} /> Unethical (Temptation)
                              </span>
                            )}
                          </div>
                          <div className="text-[11px] text-slate-400 mt-0.5">
                            Target Zone: {rationale.target_zone || 'current'} · Rationale: {rationale.rationale_note || 'Utility maximization'}
                          </div>
                        </div>
                      </div>

                      <div className="text-right font-mono">
                        <div className="text-xs font-bold text-cyan-300">
                          E[U] = {c.expected_u.toFixed(2)}
                        </div>
                        {c.risk_penalty > 0 && (
                          <div className="text-[10px] text-rose-400">
                            - Risk Penalty: {c.risk_penalty.toFixed(2)}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })
              ) : (
                <div className="p-4 rounded-xl bg-slate-900/40 text-center text-xs text-slate-400 italic font-mono">
                  Current single evaluated action: {currentVillager.last_action} (E[U] = {Number(expectedTotalU).toFixed(2)})
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* 4. Sub-Tab 2: Ground Truth vs Agent Belief State */}
      {activeSubTab === 'TRUTH_VS_BELIEF' && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Ground Truth */}
          <div className="p-5 rounded-2xl bg-emerald-950/20 border border-emerald-500/30 space-y-3">
            <div className="flex items-center gap-2 text-emerald-400 font-bold text-xs uppercase font-mono">
              <Eye size={16} /> Ground Truth (Actual World Simulation State)
            </div>
            <div className="space-y-2 text-xs">
              <div className="p-3 rounded-xl bg-black/40 border border-emerald-500/20">
                <div className="text-slate-400">Actual Granary Storage:</div>
                <div className="font-mono text-emerald-300 font-semibold mt-0.5">
                  Wheat: {state.economy.granary_storage.wheat || state.economy.granary_storage.WHEAT || 75.0} bu · Water: {state.economy.granary_storage.water || state.economy.granary_storage.WATER || 40.0} bbl
                </div>
              </div>
              <div className="p-3 rounded-xl bg-black/40 border border-emerald-500/20">
                <div className="text-slate-400">Actual Guard Locations:</div>
                <div className="font-mono text-emerald-300 font-semibold mt-0.5">
                  Dev: [{villagers.find(v => v.name === 'Dev')?.position.x ?? 20}, {villagers.find(v => v.name === 'Dev')?.position.y ?? 20}] · Tara: [{villagers.find(v => v.name === 'Tara')?.position.x ?? 12}, {villagers.find(v => v.name === 'Tara')?.position.y ?? 15}]
                </div>
              </div>
              <div className="p-3 rounded-xl bg-black/40 border border-emerald-500/20">
                <div className="text-slate-400">Legal Record & Active Penalties:</div>
                <div className="font-mono text-emerald-300 font-semibold mt-0.5">
                  In Pillory: {currentVillager.in_pillory_ticks > 0 ? `YES (${currentVillager.in_pillory_ticks} ticks remaining)` : 'NO (Free citizen)'} · Crimes Detected: {currentVillager.crimes_detected}
                </div>
              </div>
            </div>
          </div>

          {/* Agent Belief State */}
          <div className="p-5 rounded-2xl bg-cyan-950/20 border border-cyan-500/30 space-y-3">
            <div className="flex items-center gap-2 text-cyan-400 font-bold text-xs uppercase font-mono">
              <Brain size={16} /> Agent Internal Belief State (Partially Observable)
            </div>
            <div className="space-y-2 text-xs">
              <div className="p-3 rounded-xl bg-black/40 border border-cyan-500/20">
                <div className="text-slate-400">Believed Guard Detection Risk:</div>
                <div className="font-mono text-cyan-300 font-semibold mt-0.5">
                  {(Number(pDetect) * 100).toFixed(1)}% detection risk estimated from Bayesian prior
                </div>
              </div>
              <div className="p-3 rounded-xl bg-black/40 border border-cyan-500/20">
                <div className="text-slate-400">Believed Social Status (PageRank S_i):</div>
                <div className="font-mono text-cyan-300 font-semibold mt-0.5">
                  {currentVillager.status_pct.toFixed(1)}% (Estimated ranking based on peer deference)
                </div>
              </div>
              <div className="p-3 rounded-xl bg-black/40 border border-cyan-500/20">
                <div className="text-slate-400">Episodic Memory Reflection:</div>
                <div className="font-sans text-cyan-200 mt-0.5 italic">
                  "{currentVillager.episodic_memories[0] || 'No immediate episodic grievance.'}"
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 5. Sub-Tab 3: 3D Affective Emotion Dynamics */}
      {activeSubTab === 'EMOTION_PAD' && (
        <div className="p-6 rounded-2xl bg-black/35 border border-white/10 space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <span>3D Pleasure-Arousal-Dominance (PAD) Dynamical Model</span>
                <span className="px-2 py-0.5 rounded text-[11px] bg-purple-500/20 text-purple-300 border border-purple-500/30">
                  Current: {currentVillager.dominant_emotion || 'Calm / Composed'}
                </span>
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Continuous dynamical state: dE/dt = -λ(E - E_base) + Σ Perturbations(outcomes)
              </p>
            </div>
            {currentVillager.aura_color_hex && (
              <div className="flex items-center gap-2 text-xs font-mono">
                <span>Aura Color:</span>
                <span
                  className="w-4 h-4 rounded-full border border-white/40 shadow-sm"
                  style={{ backgroundColor: currentVillager.aura_color_hex }}
                />
              </div>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 rounded-xl bg-slate-900/60 border border-white/5 space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-medium">Pleasure (Valence)</span>
                <span className="font-mono text-emerald-400">{pad[0] > 0 ? `+${pad[0].toFixed(2)}` : pad[0].toFixed(2)}</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-emerald-500 rounded-full"
                  style={{ width: `${Math.max(0, Math.min(100, (pad[0] + 1) * 50))}%` }}
                />
              </div>
              <div className="text-[10px] text-slate-500">Unpleasant (-1.0) to Pleasant (+1.0)</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-white/5 space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-medium">Arousal (Activation)</span>
                <span className="font-mono text-amber-400">{pad[1] > 0 ? `+${pad[1].toFixed(2)}` : pad[1].toFixed(2)}</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-amber-500 rounded-full"
                  style={{ width: `${Math.max(0, Math.min(100, (pad[1] + 1) * 50))}%` }}
                />
              </div>
              <div className="text-[10px] text-slate-500">Sleepy/Calm (-1.0) to Frantic/Alert (+1.0)</div>
            </div>

            <div className="p-4 rounded-xl bg-slate-900/60 border border-white/5 space-y-2">
              <div className="flex justify-between text-xs">
                <span className="text-slate-300 font-medium">Dominance (Control)</span>
                <span className="font-mono text-purple-400">{pad[2] > 0 ? `+${pad[2].toFixed(2)}` : pad[2].toFixed(2)}</span>
              </div>
              <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-500 rounded-full"
                  style={{ width: `${Math.max(0, Math.min(100, (pad[2] + 1) * 50))}%` }}
                />
              </div>
              <div className="text-[10px] text-slate-500">Submissive (-1.0) to Autonomous/In-Charge (+1.0)</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
