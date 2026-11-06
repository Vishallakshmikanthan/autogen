import React from 'react';
import { ShieldAlert, Scale, Heart, Droplets, Battery, Coins, Lock, Brain } from 'lucide-react';
import { VillagerCardDTO } from '../../../types/village';

interface VillagerInspectorProps {
  villager: VillagerCardDTO | null;
}

export const VillagerInspector: React.FC<VillagerInspectorProps> = ({ villager }) => {
  if (!villager) {
    return (
      <div className="bg-research-panel border border-research-border rounded-lg p-6 text-center text-xs font-mono text-slate-500">
        Select a villager on the canvas or leaderboard to inspect utility weights, secret hut chest inventory, and real-time DecisionTraces.
      </div>
    );
  }

  const trace = villager.decision_trace;
  const rat = trace?.rationale || {};
  const topCandidates = rat.top_candidates || [];

  const renderBar = (label: string, val: number, color: string, icon: React.ReactNode) => (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[11px] font-mono">
        <span className="flex items-center space-x-1 text-slate-300">
          {icon}
          <span>{label}</span>
        </span>
        <span className="font-bold text-slate-100">{val.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-slate-800 rounded overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-300`}
          style={{ width: `${Math.max(0, Math.min(100, val))}%` }}
        />
      </div>
    </div>
  );

  return (
    <div className="bg-research-panel border border-research-border rounded-lg p-4 space-y-4 font-mono">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-research-border pb-3">
        <div>
          <div className="flex items-center space-x-2">
            <h3 className="text-base font-bold text-amber-400">{villager.name}</h3>
            <span className="px-2 py-0.5 text-[10px] rounded bg-cyan-500/15 text-cyan-300 border border-cyan-500/30">
              {villager.role}
            </span>
            {villager.in_pillory_ticks > 0 && (
              <span className="px-2 py-0.5 text-[10px] rounded bg-rose-500/20 text-rose-300 border border-rose-500/40">
                PILLORY ({villager.in_pillory_ticks}t)
              </span>
            )}
          </div>
          <p className="text-[11px] text-slate-400 mt-0.5">
            Grid ({villager.position.x}, {villager.position.y}) · Goal: {villager.current_goal_zone}
          </p>
        </div>
        <div className="text-right">
          <div className="text-xs text-slate-400">PageRank Status S_i</div>
          <div className="text-lg font-bold text-amber-400">
            {(villager.societal_status * 100).toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Physiological Vitals */}
      <div className="grid grid-cols-3 gap-3">
        {renderBar('Hunger H_i', villager.hunger, 'bg-amber-500', <Heart size={11} className="text-amber-400" />)}
        {renderBar('Hydration T_i', villager.hydration, 'bg-sky-500', <Droplets size={11} className="text-sky-400" />)}
        {renderBar('Energy E_i', villager.energy, 'bg-emerald-500', <Battery size={11} className="text-emerald-400" />)}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-slate-300">
            <span className="text-[11px] text-amber-300 font-semibold">
              {villager.archetype_title || `${villager.role} Specialist`} · Gen-{villager.generation || 1} ({villager.life_stage || 'Adult'})
            </span>
            {villager.dominant_emotion && (
              <span className="text-[11px] text-cyan-300 font-medium">
                🎭 {villager.dominant_emotion}
              </span>
            )}
          </div>
          <div className="grid grid-cols-3 gap-3">
            {renderBar('Hunger H_i', villager.hunger, 'bg-amber-500', <Heart size={11} className="text-amber-400" />)}
            {renderBar('Hydration T_i', villager.hydration, 'bg-sky-500', <Droplets size={11} className="text-sky-400" />)}
            {renderBar('Energy E_i', villager.energy, 'bg-emerald-500', <Battery size={11} className="text-emerald-400" />)}
          </div>
        </div>

        {/* Economic & Secret Chest Inventories */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          <div className="bg-slate-900/70 border border-slate-800 rounded p-2.5 space-y-1.5">
            <div className="flex items-center justify-between text-slate-300 font-semibold">
              <span className="flex items-center space-x-1.5">
                <Coins size={13} className="text-amber-400" />
                <span>Carried Inventory</span>
              </span>
              <span className="text-amber-400 font-bold">{villager.coins.toFixed(1)} Coins</span>
            </div>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {Object.entries(villager.inventory).length === 0 ? (
                <span className="text-[11px] text-slate-500">Empty satchel</span>
              ) : (
                Object.entries(villager.inventory).map(([k, v]) => (
                  <span
                    key={k}
                    className="px-2 py-0.5 rounded bg-slate-800 text-slate-200 text-[11px]"
                  >
                    {k}: <b>{v}</b>
                  </span>
                ))
              )}
            </div>
          </div>

          <div className="bg-slate-900/70 border border-slate-800 rounded p-2.5 space-y-1.5">
            <div className="flex items-center justify-between text-slate-300 font-semibold">
              <span className="flex items-center space-x-1.5">
                <Lock size={13} className="text-purple-400" />
                <span>Secret Hut Chest (Ground Truth)</span>
              </span>
              <span className="text-[10px] text-slate-400">Tithes: {villager.tithes_contributed}</span>
            </div>
            <div className="flex flex-wrap gap-1.5 pt-1">
              {Object.entries(villager.secret_chest_inventory).map(([k, v]) => (
                <span
                  key={k}
                  className="px-2 py-0.5 rounded bg-purple-950/40 border border-purple-500/30 text-purple-200 text-[11px]"
                >
                  {k}: <b>{v}</b>
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Utility Weights & Ethics Risk Profile */}
        <div className="bg-slate-900/70 border border-slate-800 rounded p-3 space-y-2">
          <div className="flex items-center justify-between text-xs font-bold text-slate-200">
            <span className="flex items-center space-x-1.5">
              <Scale size={14} className="text-cyan-400" />
              <span>Multi-Objective Utility Weights &amp; Risk Tolerance</span>
            </span>
            <span className="text-[11px] text-rose-400">
              Crimes: {villager.crimes_committed} (Caught: {villager.crimes_detected})
            </span>
          </div>
          <div className="grid grid-cols-4 gap-2 text-[11px] text-center pt-1">
            <div className="bg-slate-950 p-1.5 rounded border border-slate-800">
              <div className="text-slate-400">w_phys</div>
              <div className="font-bold text-slate-100">{villager.utility_weights.w_phys}</div>
            </div>
            <div className="bg-slate-950 p-1.5 rounded border border-slate-800">
              <div className="text-slate-400">w_econ</div>
              <div className="font-bold text-slate-100">{villager.utility_weights.w_econ}</div>
            </div>
            <div className="bg-slate-950 p-1.5 rounded border border-slate-800">
              <div className="text-slate-400">w_soc</div>
              <div className="font-bold text-slate-100">{villager.utility_weights.w_soc}</div>
            </div>
            <div className="bg-slate-950 p-1.5 rounded border border-slate-800">
              <div className="text-slate-400">θ_risk</div>
              <div className="font-bold text-amber-400">{villager.utility_weights.theta_risk}</div>
            </div>
          </div>
        </div>

        {/* Live DecisionTrace Breakdown */}
        <div className="bg-slate-950/90 border border-slate-800 rounded p-3 space-y-2">
          <div className="flex items-center justify-between text-xs font-bold">
            <span className="flex items-center space-x-1.5 text-cyan-400">
              <Brain size={14} />
              <span>Verified DecisionTrace (Step {trace?.step_id ?? 0})</span>
            </span>
            <span className="text-[11px] text-slate-400">
              Believed P_detect: {(villager.believed_detection_risk * 100).toFixed(1)}%
            </span>
          </div>

          {trace ? (
            <div className="space-y-2 text-[11px]">
              <div className="flex items-center justify-between bg-slate-900 px-2.5 py-1.5 rounded border border-slate-800">
                <span>
                  Selected: <b className="text-amber-300">{rat.village_action}</b> &rarr; Goal:{' '}
                  <b className="text-cyan-300">{rat.strategic_goal_action}</b>
                </span>
                <span className="font-bold text-emerald-400">
                  E[U] = {Number(rat.expected_total_utility ?? 0).toFixed(2)}
                </span>
              </div>
              {rat.rationale_note && (
                <div className="text-slate-400 italic">{rat.rationale_note}</div>
              )}
              <div className="space-y-1 pt-1">
                <div className="text-[10px] uppercase tracking-wider text-slate-500">
                  Ranked Legal vs. Unethical Candidate Pathways:
                </div>
                {topCandidates.map((cand, i) => (
                  <div
                    key={`${cand.action}-${i}`}
                    className={`flex items-center justify-between px-2 py-1 rounded text-[10px] ${cand.is_unethical
                      ? 'bg-rose-950/40 border border-rose-500/30 text-rose-200'
                      : 'bg-slate-900/60 text-slate-300'
                      }`}
                  >
                    <span className="flex items-center space-x-1.5">
                      {cand.is_unethical && <ShieldAlert size={11} className="text-rose-400" />}
                      <span>
                        {i + 1}. {cand.action}
                      </span>
                    </span>
                    <span>
                      E[U]={cand.expected_u.toFixed(2)} (Risk R={cand.risk_penalty.toFixed(2)})
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-slate-500 text-xs">Step simulation to generate DecisionTrace.</div>
          )}
        </div>
      </div>
      );
    </div>
  );
};

