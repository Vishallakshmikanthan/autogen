import React from 'react';
import { Share2, Award } from 'lucide-react';
import { SocialGraphPayloadDTO, VillagerCardDTO } from '../../../types/village';

interface SocialGraphViewProps {
  socialGraph: SocialGraphPayloadDTO;
  villagers: VillagerCardDTO[];
  selectedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
}

export const SocialGraphView: React.FC<SocialGraphViewProps> = ({
  socialGraph,
  villagers,
  selectedAgentId,
  onSelectAgent,
}) => {
  const cx = 160;
  const cy = 140;
  const radius = 98;
  const n = Math.max(1, villagers.length);

  const nodeCoords: Record<string, { x: number; y: number; v: VillagerCardDTO }> = {};
  villagers.forEach((v, i) => {
    const angle = (2 * Math.PI * i) / n - Math.PI / 2;
    nodeCoords[v.agent_id] = {
      x: cx + Math.cos(angle) * radius,
      y: cy + Math.sin(angle) * radius,
      v,
    };
  });

  // Filter notable edges or edges connected to selected agent for visual clarity
  const visibleEdges = socialGraph.edges.filter((e) => {
    if (selectedAgentId) {
      return e.source === selectedAgentId || e.target === selectedAgentId;
    }
    return Math.abs(e.trust - 0.5) > 0.04 || e.trust >= 0.56;
  });

  return (
    <div className="bg-research-panel border border-research-border rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between border-b border-research-border pb-2">
        <div className="flex items-center space-x-2">
          <Share2 size={15} className="text-cyan-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Directed Trust Graph &amp; PageRank Status S_i(t)
          </h3>
        </div>
        <span className="text-[10px] font-mono text-slate-400">
          d=0.85 · iter={socialGraph.pagerank_iterations}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3 items-center">
        {/* SVG Network Graph */}
        <div className="bg-slate-950/70 rounded border border-slate-800 flex items-center justify-center p-1">
          <svg width={320} height={280} className="overflow-visible">
            {visibleEdges.slice(0, 45).map((edge, idx) => {
              const src = nodeCoords[edge.source];
              const dst = nodeCoords[edge.target];
              if (!src || !dst) return null;
              const isHighTrust = edge.trust >= 0.52;
              const stroke = isHighTrust ? '#10b981' : '#f43f5e';
              const opacity = Math.min(0.85, Math.max(0.18, Math.abs(edge.trust - 0.4) * 1.4));
              return (
                <line
                  key={`${edge.source}-${edge.target}-${idx}`}
                  x1={src.x}
                  y1={src.y}
                  x2={dst.x}
                  y2={dst.y}
                  stroke={stroke}
                  strokeOpacity={opacity}
                  strokeWidth={edge.source === selectedAgentId ? 2.0 : 1.1}
                />
              );
            })}

            {Object.values(nodeCoords).map(({ x, y, v }) => {
              const isSelected = v.agent_id === selectedAgentId;
              const nodeR = Math.max(11, Math.min(20, 8 + v.societal_status * 65));
              const hasCrimes = v.crimes_detected > 0;
              return (
                <g
                  key={v.agent_id}
                  onClick={() => onSelectAgent(v.agent_id)}
                  className="cursor-pointer"
                >
                  <circle
                    cx={x}
                    cy={y}
                    r={nodeR}
                    fill={hasCrimes ? '#881337' : isSelected ? '#f59e0b' : '#0f172a'}
                    stroke={isSelected ? '#fbbf24' : hasCrimes ? '#f43f5e' : '#38bdf8'}
                    strokeWidth={isSelected ? 2.5 : 1.5}
                  />
                  <text
                    x={x}
                    y={y + 3}
                    textAnchor="middle"
                    className="fill-slate-100 font-mono text-[9px] font-bold select-none"
                  >
                    {v.name.slice(0, 4)}
                  </text>
                  <text
                    x={x}
                    y={y + nodeR + 10}
                    textAnchor="middle"
                    className="fill-amber-300 font-mono text-[8px] select-none"
                  >
                    {(v.societal_status * 100).toFixed(1)}%
                  </text>
                </g>
              );
            })}
          </svg>
        </div>

        {/* Societal Status Leaderboard */}
        <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
          <div className="flex items-center space-x-1.5 text-[11px] font-mono text-amber-400 mb-1">
            <Award size={13} />
            <span className="font-semibold">Societal Status Leaderboard (Σ S_i = 1.0)</span>
          </div>
          {socialGraph.leaderboard.map((entry, idx) => {
            const v = villagers.find((vil) => vil.agent_id === entry.agent_id);
            if (!v) return null;
            const isSel = v.agent_id === selectedAgentId;
            return (
              <div
                key={entry.agent_id}
                onClick={() => onSelectAgent(v.agent_id)}
                className={`cursor-pointer flex items-center justify-between px-2.5 py-1.5 rounded border text-xs font-mono transition ${
                  isSel
                    ? 'bg-amber-500/15 border-amber-400 text-amber-200'
                    : 'bg-slate-900/60 border-slate-800 text-slate-300 hover:bg-slate-800/60'
                }`}
              >
                <div className="flex items-center space-x-2">
                  <span className="text-slate-500 w-4">{idx + 1}.</span>
                  <span className="font-bold text-slate-100">{v.name}</span>
                  <span className="text-[10px] text-slate-400">({v.role})</span>
                </div>
                <div className="flex items-center space-x-3">
                  <span className="text-[10px] text-emerald-400">{v.coins.toFixed(0)}c</span>
                  <span className="font-bold text-amber-400">
                    {(entry.status * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

