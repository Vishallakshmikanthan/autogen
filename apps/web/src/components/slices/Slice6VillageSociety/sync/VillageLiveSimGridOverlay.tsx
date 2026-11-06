import React from 'react';
import { VillageStatePayloadDTO } from '../../../../types/village';
import { Play, RotateCcw, CloudSun, CloudRain, Users, ShoppingBag, Wheat, Droplets } from 'lucide-react';

interface VillageLiveSimGridOverlayProps {
  state: VillageStatePayloadDTO;
  selectedVillagerId: string;
  onSelectVillager: (id: string) => void;
  onStepSimulation: (steps?: number) => void;
  onResetSimulation: () => void;
  onToggleDrought: () => void;
  onConvokeSabha: () => void;
  loading: boolean;
}

export const VillageLiveSimGridOverlay: React.FC<VillageLiveSimGridOverlayProps> = ({
  state,
  selectedVillagerId,
  onSelectVillager,
  onStepSimulation,
  onResetSimulation,
  onToggleDrought,
  onConvokeSabha,
  loading,
}) => {
  const gridW = state.grid?.width || 40;
  const gridH = state.grid?.height || 40;
  const villagers = state.villagers || [];
  const zones = state.zones || [];

  return (
    <div className="space-y-6 text-slate-100 font-sans">
      {/* 1. Header Toolbar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-4 rounded-2xl bg-black/40 border border-white/10 backdrop-blur-md">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-lg font-bold text-white tracking-wide flex items-center gap-2">
              <span>Village 2D Spatial Matrix</span>
              <span className="px-2 py-0.5 rounded text-xs font-mono bg-cyan-950 text-cyan-300 border border-cyan-800">
                Step: {state.step}
              </span>
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Synchronized 2D grid matrix of the 3D cinematic world ({gridW} × {gridH} cells)
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => onStepSimulation(1)}
            disabled={loading}
            className="px-3.5 py-1.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs flex items-center gap-1.5 transition disabled:opacity-50"
          >
            <Play size={13} className="fill-current" /> Step +1 Tick
          </button>
          <button
            onClick={() => onStepSimulation(5)}
            disabled={loading}
            className="px-3 py-1.5 rounded-xl bg-cyan-500/20 hover:bg-cyan-500/35 border border-cyan-400/40 text-cyan-200 text-xs font-semibold transition disabled:opacity-50"
          >
            +5 Ticks
          </button>
          <button
            onClick={onToggleDrought}
            disabled={loading}
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold flex items-center gap-1.5 border transition disabled:opacity-50 ${
              state.drought_active
                ? 'bg-rose-500/30 text-rose-200 border-rose-500/60'
                : 'bg-white/5 hover:bg-white/10 text-slate-300 border-white/10'
            }`}
          >
            {state.drought_active ? <CloudRain size={13} /> : <CloudSun size={13} />}
            {state.drought_active ? 'Drought ACTIVE' : 'Trigger Drought'}
          </button>
          <button
            onClick={onConvokeSabha}
            disabled={loading}
            className="px-3 py-1.5 rounded-xl bg-amber-500/20 hover:bg-amber-500/35 border border-amber-400/40 text-amber-200 text-xs font-semibold flex items-center gap-1.5 transition disabled:opacity-50"
          >
            <Users size={13} /> Convoke Sabha
          </button>
          <button
            onClick={onResetSimulation}
            disabled={loading}
            className="px-3 py-1.5 rounded-xl bg-rose-500/10 hover:bg-rose-500/25 border border-rose-400/30 text-rose-300 text-xs font-semibold flex items-center gap-1.5 transition disabled:opacity-50"
          >
            <RotateCcw size={13} /> Reset
          </button>
        </div>
      </div>

      {/* 2. Grid & Telemetry Split */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: 2D Spatial SVG Grid */}
        <div className="lg:col-span-2 p-5 rounded-2xl bg-black/45 border border-white/10 flex flex-col items-center">
          <div className="w-full flex items-center justify-between mb-3 text-xs text-slate-400 font-mono">
            <span>SPATIAL MATRIX (40x40 Grid)</span>
            <span>Click any villager dot to inspect</span>
          </div>

          <div className="relative w-full max-w-[540px] aspect-square rounded-2xl bg-[#091522] border border-cyan-500/30 overflow-hidden shadow-2xl p-2">
            <svg viewBox={`0 0 ${gridW} ${gridH}`} className="w-full h-full">
              {/* Grid Lines */}
              <defs>
                <pattern id="gridPattern" width="1" height="1" patternUnits="userSpaceOnUse">
                  <path d="M 1 0 L 0 0 0 1" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="0.05" />
                </pattern>
              </defs>
              <rect width={gridW} height={gridH} fill="url(#gridPattern)" />

              {/* Render Zones */}
              {zones.map((z) => {
                const w = z.x_max - z.x_min;
                const h = z.y_max - z.y_min;
                let fillCol = 'rgba(34, 211, 238, 0.08)';
                let strokeCol = 'rgba(34, 211, 238, 0.35)';

                if (z.zone_id === 'village_commons') {
                  fillCol = 'rgba(234, 179, 8, 0.12)';
                  strokeCol = 'rgba(234, 179, 8, 0.45)';
                } else if (z.zone_id === 'town_well') {
                  fillCol = 'rgba(56, 189, 248, 0.15)';
                  strokeCol = 'rgba(56, 189, 248, 0.6)';
                } else if (z.zone_id === 'agricultural_fields') {
                  fillCol = 'rgba(16, 185, 129, 0.12)';
                  strokeCol = 'rgba(16, 185, 129, 0.45)';
                } else if (z.zone_id === 'wild_forest') {
                  fillCol = 'rgba(5, 150, 105, 0.18)';
                  strokeCol = 'rgba(5, 150, 105, 0.5)';
                } else if (z.zone_id === 'central_market') {
                  fillCol = 'rgba(244, 63, 94, 0.12)';
                  strokeCol = 'rgba(244, 63, 94, 0.45)';
                }

                return (
                  <g key={z.zone_id}>
                    <rect
                      x={z.x_min}
                      y={z.y_min}
                      width={w}
                      height={h}
                      rx="0.8"
                      fill={fillCol}
                      stroke={strokeCol}
                      strokeWidth="0.15"
                    />
                    <text
                      x={z.x_min + w / 2}
                      y={z.y_min + h / 2}
                      fill={strokeCol}
                      fontSize="1.1"
                      textAnchor="middle"
                      dominantBaseline="middle"
                      className="font-mono font-bold select-none"
                    >
                      {z.label}
                    </text>
                  </g>
                );
              })}

              {/* Render Villagers */}
              {villagers.map((v) => {
                const isSelected = v.agent_id === selectedVillagerId;
                const gx = v.position?.x ?? 20;
                const gy = v.position?.y ?? 20;
                let dotCol = '#38bdf8';
                if (v.role === 'Farmer') dotCol = '#eab308';
                else if (v.role === 'Merchant') dotCol = '#f43f5e';
                else if (v.role === 'Guard') dotCol = '#94a3b8';
                else if (v.role === 'Wanderer') dotCol = '#c084fc';

                return (
                  <g
                    key={v.agent_id}
                    className="cursor-pointer transition-transform"
                    onClick={() => onSelectVillager(v.agent_id)}
                  >
                    {isSelected && (
                      <circle
                        cx={gx}
                        cy={gy}
                        r="2.2"
                        fill="none"
                        stroke="#22d3ee"
                        strokeWidth="0.25"
                        strokeDasharray="0.5 0.3"
                      />
                    )}
                    <circle cx={gx} cy={gy} r={isSelected ? 1.4 : 1.0} fill={dotCol} />
                    <text
                      x={gx}
                      y={gy - 1.4}
                      fill="#ffffff"
                      fontSize="0.9"
                      textAnchor="middle"
                      className="font-mono font-bold select-none"
                    >
                      {v.name}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        </div>

        {/* Right Col: Live Economic & Institutional Status */}
        <div className="space-y-4">
          {/* Economy Ticker */}
          <div className="p-4 rounded-2xl bg-black/40 border border-white/10 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono uppercase tracking-wider text-amber-400 font-bold">
              <span className="flex items-center gap-1.5">
                <ShoppingBag size={14} /> Clearing Prices
              </span>
              <span className="text-slate-400">Double Auction</span>
            </div>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="p-2.5 rounded-xl bg-slate-900/70 border border-white/5">
                <div className="text-slate-400 text-[10px]">WHEAT</div>
                <div className="text-amber-300 font-bold text-sm">
                  {state.economy.clearing_prices.wheat?.toFixed(2) || state.economy.clearing_prices.WHEAT?.toFixed(2) || '3.00'} c
                </div>
              </div>
              <div className="p-2.5 rounded-xl bg-slate-900/70 border border-white/5">
                <div className="text-slate-400 text-[10px]">WATER</div>
                <div className="text-cyan-300 font-bold text-sm">
                  {state.economy.clearing_prices.water?.toFixed(2) || state.economy.clearing_prices.WATER?.toFixed(2) || '1.50'} c
                </div>
              </div>
            </div>
            <div className="pt-2 border-t border-white/5 text-xs font-mono space-y-1">
              <div className="flex justify-between text-slate-300">
                <span className="flex items-center gap-1"><Wheat size={12} className="text-amber-400" /> Granary Wheat:</span>
                <span className="font-bold text-amber-300">
                  {state.economy.granary_storage.wheat || state.economy.granary_storage.WHEAT || 75.0} bu
                </span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span className="flex items-center gap-1"><Droplets size={12} className="text-cyan-400" /> Well Level:</span>
                <span className="font-bold text-cyan-300">{state.economy.well_water_level?.toFixed(0) || 300} L</span>
              </div>
            </div>
          </div>

          {/* Villagers List Quick Inspect */}
          <div className="p-4 rounded-2xl bg-black/40 border border-white/10 space-y-3">
            <div className="text-xs font-mono uppercase tracking-wider text-cyan-400 font-bold">
              Living Citizens ({villagers.length})
            </div>
            <div className="space-y-1.5 max-h-[220px] overflow-y-auto pr-1">
              {villagers.map((v) => {
                const isSelected = v.agent_id === selectedVillagerId;
                return (
                  <div
                    key={v.agent_id}
                    onClick={() => onSelectVillager(v.agent_id)}
                    className={`p-2 rounded-xl border flex items-center justify-between cursor-pointer transition text-xs ${
                      isSelected
                        ? 'bg-cyan-500/25 border-cyan-400/60 font-semibold'
                        : 'bg-white/5 hover:bg-white/10 border-white/5'
                    }`}
                  >
                    <div>
                      <div className="text-slate-200">{v.name} ({v.role})</div>
                      <div className="text-[10px] text-slate-400">
                        Goal: {v.current_goal_zone} · Act: {v.last_action}
                      </div>
                    </div>
                    <div className="text-right font-mono text-[11px]">
                      <div className="text-amber-300">{v.coins.toFixed(0)} coins</div>
                      <div className="text-purple-300">S_i {v.status_pct.toFixed(0)}%</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
