import React, { useRef, useEffect } from 'react';
import { ViewerStateDTO } from '../../../types/viewer';
import { Eye, EyeOff } from 'lucide-react';

interface TruthVsBeliefSplitProps {
  viewerState: ViewerStateDTO | null;
  selectedAgentId: string | null;
}

export const TruthVsBeliefSplit: React.FC<TruthVsBeliefSplitProps> = ({
  viewerState,
  selectedAgentId,
}) => {
  const truthCanvasRef = useRef<HTMLCanvasElement>(null);
  const beliefCanvasRef = useRef<HTMLCanvasElement>(null);

  const renderCanvases = () => {
    if (!viewerState) return;

    const truthCanvas = truthCanvasRef.current;
    const beliefCanvas = beliefCanvasRef.current;
    if (!truthCanvas || !beliefCanvas) return;

    const tCtx = truthCanvas.getContext('2d');
    const bCtx = beliefCanvas.getContext('2d');
    if (!tCtx || !bCtx) return;

    const w = truthCanvas.width;
    const h = truthCanvas.height;
    const gridW = viewerState.width || 20;
    const gridH = viewerState.height || 20;
    const cellSize = Math.min(w / gridW, h / gridH);

    // Selected agent position
    const selectedAgent = viewerState.agents.find((a) => a.agent_id === selectedAgentId) || viewerState.agents[0];
    const agentPos = selectedAgent ? { x: selectedAgent.x, y: selectedAgent.y } : { x: 0, y: 0 };

    // Clear
    tCtx.clearRect(0, 0, w, h);
    bCtx.clearRect(0, 0, w, h);

    // 1. Render Ground Truth (Left)
    for (const cell of viewerState.grid_cells) {
      const px = cell.x * cellSize;
      const py = cell.y * cellSize;

      tCtx.fillStyle = cell.terrain === 'obstacle' ? '#47181E' : '#14251D';
      tCtx.fillRect(px, py, cellSize, cellSize);
      tCtx.strokeStyle = '#1E293B33';
      tCtx.strokeRect(px, py, cellSize, cellSize);

      if (cell.food > 0.1) {
        tCtx.fillStyle = '#10B981';
        tCtx.beginPath();
        tCtx.arc(px + cellSize / 2, py + cellSize / 2, cellSize * 0.3, 0, Math.PI * 2);
        tCtx.fill();
      }
    }

    // 2. Render Agent Internal Belief (Right) with Fog of War (Manhattan distance <= 3)
    const visionRadius = 3;
    for (const cell of viewerState.grid_cells) {
      const px = cell.x * cellSize;
      const py = cell.y * cellSize;
      const dist = Math.abs(cell.x - agentPos.x) + Math.abs(cell.y - agentPos.y);

      if (dist <= visionRadius) {
        // Visible and known with full confidence
        bCtx.fillStyle = cell.terrain === 'obstacle' ? '#47181E' : '#0B2936';
        bCtx.fillRect(px, py, cellSize, cellSize);
        bCtx.strokeStyle = '#06B6D444';
        bCtx.strokeRect(px, py, cellSize, cellSize);

        if (cell.food > 0.1) {
          bCtx.fillStyle = '#06B6D4';
          bCtx.beginPath();
          bCtx.arc(px + cellSize / 2, py + cellSize / 2, cellSize * 0.3, 0, Math.PI * 2);
          bCtx.fill();
        }
      } else {
        // Fog of war / unobserved / degraded confidence
        bCtx.fillStyle = '#05070B';
        bCtx.fillRect(px, py, cellSize, cellSize);
        bCtx.strokeStyle = '#0F172A44';
        bCtx.strokeRect(px, py, cellSize, cellSize);
      }
    }

    // Draw agents
    if (selectedAgent) {
      const ax = selectedAgent.x * cellSize + cellSize / 2;
      const ay = selectedAgent.y * cellSize + cellSize / 2;

      // Truth
      tCtx.fillStyle = '#10B981';
      tCtx.beginPath();
      tCtx.arc(ax, ay, cellSize * 0.4, 0, Math.PI * 2);
      tCtx.fill();

      // Belief avatar
      bCtx.fillStyle = '#06B6D4';
      bCtx.beginPath();
      bCtx.arc(ax, ay, cellSize * 0.4, 0, Math.PI * 2);
      bCtx.fill();
    }
  };

  useEffect(() => {
    renderCanvases();
  }, [viewerState, selectedAgentId]);

  return (
    <div className="bg-research-panel border border-research-border rounded-lg p-4 font-mono text-xs space-y-3">
      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
        <span className="font-semibold text-slate-200 uppercase tracking-wider">
          Epistemic Split Screen (Truth vs Belief)
        </span>
        <span className="text-slate-400 text-[11px]">Agent: {selectedAgentId || 'None'}</span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Left: Actual World State */}
        <div className="flex flex-col items-center space-y-2">
          <div className="flex items-center space-x-1.5 text-emerald-400 font-semibold text-[11px]">
            <Eye size={14} />
            <span>ACTUAL WORLD STATE (GROUND TRUTH)</span>
          </div>
          <div className="border border-emerald-900/40 rounded overflow-hidden">
            <canvas ref={truthCanvasRef} width={340} height={340} className="block" />
          </div>
          <span className="text-[10px] text-slate-500">Includes all unobserved terrain & hazards</span>
        </div>

        {/* Right: Agent Internal Belief */}
        <div className="flex flex-col items-center space-y-2">
          <div className="flex items-center space-x-1.5 text-cyan-400 font-semibold text-[11px]">
            <EyeOff size={14} />
            <span>AGENT BELIEF STATE (FOG OF WAR)</span>
          </div>
          <div className="border border-cyan-900/40 rounded overflow-hidden">
            <canvas ref={beliefCanvasRef} width={340} height={340} className="block" />
          </div>
          <span className="text-[10px] text-slate-500">Filtered by sensory horizon & spatial memory</span>
        </div>
      </div>
    </div>
  );
};

