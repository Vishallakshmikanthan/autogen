import React, { useEffect, useRef, useState } from 'react';
import { Box, Grid, Download, Sun, Moon } from 'lucide-react';
import { VillageStatePayloadDTO } from '../../../types/village';

interface VillageCanvasProps {
  villageState: VillageStatePayloadDTO;
  selectedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
}

const ROLE_COLORS: Record<string, string> = {
  Farmer: '#6aa84f',
  'Water-Carrier': '#4a90d9',
  Merchant: '#d9534f',
  Guard: '#9ca3af',
  Wanderer: '#b07acb',
  Woodcutter: '#d97706',
};

const ZONE_COLORS: Record<string, string> = {
  wild_forest: 'rgba(34, 84, 47, 0.45)',
  private_huts: 'rgba(180, 142, 92, 0.35)',
  central_market: 'rgba(212, 160, 23, 0.38)',
  town_well: 'rgba(56, 189, 248, 0.45)',
  village_commons: 'rgba(168, 85, 247, 0.35)',
  agricultural_fields: 'rgba(132, 204, 22, 0.38)',
  windmill: 'rgba(245, 158, 11, 0.35)',
  tavern: 'rgba(239, 68, 68, 0.35)',
  smithy: 'rgba(148, 163, 184, 0.35)',
};

export const VillageCanvas: React.FC<VillageCanvasProps> = ({
  villageState,
  selectedAgentId,
  onSelectAgent,
}) => {
  const [viewMode, setViewMode] = useState<'3D_CINEMATIC' | '2D_TACTICAL'>('3D_CINEMATIC');
  const iframeRef = useRef<HTMLIFrameElement | null>(null);
  const canvas2dRef = useRef<HTMLCanvasElement | null>(null);

  // Sync live AUTOGEN Village state into the 3D Cinematic Three.js scene
  useEffect(() => {
    if (viewMode === '3D_CINEMATIC' && iframeRef.current?.contentWindow) {
      const selectedVillager = villageState.villagers.find(
        (v) => v.agent_id === selectedAgentId
      );
      iframeRef.current.contentWindow.postMessage(
        {
          type: 'AUTOGEN_VILLAGE_SYNC',
          payload: villageState,
          selectedVillagerName: selectedVillager?.name || null,
        },
        '*'
      );
    }
  }, [villageState, selectedAgentId, viewMode]);

  // Listen for villager click selection events from inside the 3D Cinematic scene
  useEffect(() => {
    const handleMessage = (e: MessageEvent) => {
      if (e.data && e.data.type === 'AUTOGEN_VILLAGER_SELECTED' && e.data.name) {
        const match = villageState.villagers.find(
          (v) => v.name.toLowerCase() === String(e.data.name).toLowerCase()
        );
        if (match) {
          onSelectAgent(match.agent_id);
        }
      }
    };
    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, [villageState.villagers, onSelectAgent]);

  // Render 2D Tactical Municipal Grid when in 2D_TACTICAL mode
  useEffect(() => {
    if (viewMode !== '2D_TACTICAL' || !canvas2dRef.current) return;
    const canvas = canvas2dRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const gw = villageState.grid.width;
    const gh = villageState.grid.height;
    const cellW = W / gw;
    const cellH = H / gh;

    // Background modulated by ambient light
    const light = villageState.ambient_light;
    ctx.fillStyle = light > 0.5 ? '#111c16' : '#090d16';
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let x = 0; x <= gw; x++) {
      ctx.beginPath();
      ctx.moveTo(x * cellW, 0);
      ctx.lineTo(x * cellW, H);
      ctx.stroke();
    }
    for (let y = 0; y <= gh; y++) {
      ctx.beginPath();
      ctx.moveTo(0, y * cellH);
      ctx.lineTo(W, y * cellH);
      ctx.stroke();
    }

    // Draw Municipal Zones
    villageState.zones.forEach((z) => {
      const zx = z.x_min * cellW;
      const zy = z.y_min * cellH;
      const zw = (z.x_max - z.x_min + 1) * cellW;
      const zh = (z.y_max - z.y_min + 1) * cellH;
      ctx.fillStyle = ZONE_COLORS[z.zone_type] || 'rgba(100,116,139,0.25)';
      ctx.fillRect(zx, zy, zw, zh);
      ctx.strokeStyle = 'rgba(255,255,255,0.18)';
      ctx.strokeRect(zx, zy, zw, zh);

      ctx.fillStyle = '#e2e8f0';
      ctx.font = 'bold 10px monospace';
      ctx.fillText(z.label, zx + 5, zy + 13);
    });

    // Draw Villagers
    villageState.villagers.forEach((v) => {
      const vx = (v.position.x + 0.5) * cellW;
      const vy = (v.position.y + 0.5) * cellH;
      const isSel = v.agent_id === selectedAgentId;

      if (isSel) {
        ctx.beginPath();
        ctx.arc(vx, vy, 13, 0, Math.PI * 2);
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(vx, vy, 7.5, 0, Math.PI * 2);
      ctx.fillStyle = ROLE_COLORS[v.role] || '#38bdf8';
      ctx.fill();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = '#f8fafc';
      ctx.font = 'bold 10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(v.name, vx, vy - 10);
    });
    ctx.textAlign = 'start';
  }, [villageState, selectedAgentId, viewMode]);

  const handleCanvas2dClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvas2dRef.current) return;
    const rect = canvas2dRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;
    const cellW = canvas2dRef.current.width / villageState.grid.width;
    const cellH = canvas2dRef.current.height / villageState.grid.height;

    for (const v of villageState.villagers) {
      const vx = (v.position.x + 0.5) * cellW;
      const vy = (v.position.y + 0.5) * cellH;
      if (Math.hypot(clickX - vx, clickY - vy) <= 16) {
        onSelectAgent(v.agent_id);
        return;
      }
    }
  };

  const handleExportJson = () => {
    const blob = new Blob([JSON.stringify(villageState.cinematic_export, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'village_world.json';
    a.click();
  };

  return (
    <div className="bg-research-panel border border-research-border rounded-lg overflow-hidden flex flex-col">
      {/* Canvas Header Bar */}
      <div className="flex flex-wrap items-center justify-between px-4 py-2.5 border-b border-research-border bg-slate-900/60 gap-2">
        <div className="flex items-center space-x-3">
          <span className="text-xs font-bold uppercase tracking-wider text-amber-400">
            AUTOGEN Village World
          </span>
          <span className="flex items-center space-x-1 text-xs font-mono text-slate-300">
            {villageState.ambient_light > 0.45 ? (
              <Sun size={14} className="text-amber-400" />
            ) : (
              <Moon size={14} className="text-indigo-400" />
            )}
            <span>
              {Math.floor(villageState.time_of_day)
                .toString()
                .padStart(2, '0')}
              :
              {Math.floor((villageState.time_of_day % 1) * 60)
                .toString()
                .padStart(2, '0')}{' '}
              (Light {(villageState.ambient_light * 100).toFixed(0)}%)
            </span>
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setViewMode('3D_CINEMATIC')}
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-mono transition ${
              viewMode === '3D_CINEMATIC'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 font-semibold'
                : 'bg-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Box size={13} />
            <span>3D Cinematic World</span>
          </button>

          <button
            onClick={() => setViewMode('2D_TACTICAL')}
            className={`flex items-center space-x-1.5 px-2.5 py-1 rounded text-xs font-mono transition ${
              viewMode === '2D_TACTICAL'
                ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-semibold'
                : 'bg-slate-800 text-slate-400 hover:text-slate-200'
            }`}
          >
            <Grid size={13} />
            <span>2D Tactical Grid (40x40)</span>
          </button>

          <button
            onClick={handleExportJson}
            className="flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-mono bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700"
          >
            <Download size={13} />
            <span>village_world.json</span>
          </button>
        </div>
      </div>

      {/* Viewport */}
      <div className="relative w-full h-[500px] bg-slate-950 flex items-center justify-center">
        {viewMode === '3D_CINEMATIC' ? (
          <iframe
            ref={iframeRef}
            src="/village_cinematic.html"
            title="AUTOGEN Village 3D Cinematic Edition"
            className="w-full h-full border-0"
          />
        ) : (
          <canvas
            ref={canvas2dRef}
            width={680}
            height={480}
            onClick={handleCanvas2dClick}
            className="cursor-pointer rounded border border-slate-800"
          />
        )}
      </div>
    </div>
  );
};

