import React, { useEffect, useRef, useCallback } from 'react';
import { ViewerStateDTO } from '../../../types/viewer';

interface CanvasRendererProps {
  viewerState: ViewerStateDTO | null;
  selectedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
  width?: number;
  height?: number;
}

const TERRAIN_COLORS: Record<string, string> = {
  grassland: '#14251D',
  forest: '#0D3320',
  desert: '#362D1C',
  water: '#0C2646',
  mountain: '#2A303C',
  obstacle: '#47181E',
};

const AGENT_COLORS = ['#06B6D4', '#10B981', '#8B5CF6', '#F59E0B', '#EC4899', '#3B82F6'];

export const CanvasRenderer: React.FC<CanvasRendererProps> = ({
  viewerState,
  selectedAgentId,
  onSelectAgent,
  width = 540,
  height = 540,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    if (!viewerState) {
      ctx.fillStyle = '#111827';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#64748B';
      ctx.font = '12px monospace';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting Simulation Stream...', width / 2, height / 2);
      return;
    }

    const gridW = viewerState.width || 20;
    const gridH = viewerState.height || 20;
    const cellSize = Math.min(width / gridW, height / gridH);

    // 1. Render Terrain Grid Cells
    for (const cell of viewerState.grid_cells) {
      const px = cell.x * cellSize;
      const py = cell.y * cellSize;

      ctx.fillStyle = TERRAIN_COLORS[cell.terrain] || '#111827';
      ctx.fillRect(px, py, cellSize, cellSize);

      // Fine grid borders
      ctx.strokeStyle = '#1E293B33';
      ctx.lineWidth = 0.5;
      ctx.strokeRect(px, py, cellSize, cellSize);

      // Render Food Resource nodes
      if (cell.food > 0.1) {
        ctx.fillStyle = '#10B981';
        ctx.beginPath();
        const r = Math.min(cellSize * 0.35, Math.max(2, (cell.food / 10.0) * cellSize * 0.35));
        ctx.arc(px + cellSize / 2, py + cellSize / 2, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 2. Render Movement Trails
    if (viewerState.overlays && viewerState.overlays.movement_trails) {
      for (const [aid, trail] of Object.entries(viewerState.overlays.movement_trails)) {
        if (!trail || trail.length < 2) continue;
        const colorIdx = Math.abs(aid.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0)) % AGENT_COLORS.length;
        ctx.strokeStyle = `${AGENT_COLORS[colorIdx]}66`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < trail.length; i++) {
          const [tx, ty] = trail[i];
          const cx = tx * cellSize + cellSize / 2;
          const cy = ty * cellSize + cellSize / 2;
          if (i === 0) ctx.moveTo(cx, cy);
          else ctx.lineTo(cx, cy);
        }
        ctx.stroke();
      }
    }

    // 3. Render Agents
    for (let idx = 0; idx < viewerState.agents.length; idx++) {
      const agent = viewerState.agents[idx];
      const ax = agent.x * cellSize + cellSize / 2;
      const ay = agent.y * cellSize + cellSize / 2;
      const radius = cellSize * 0.4;

      if (!agent.is_alive) {
        // Skull or tombstone marker for dead agent
        ctx.fillStyle = '#EF4444';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('x', ax, ay + 3);
        continue;
      }

      const color = AGENT_COLORS[idx % AGENT_COLORS.length];

      // Pulsating aura if selected
      if (agent.agent_id === selectedAgentId) {
        ctx.fillStyle = `${color}44`;
        ctx.beginPath();
        ctx.arc(ax, ay, radius * 1.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Agent body
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(ax, ay, radius, 0, Math.PI * 2);
      ctx.fill();

      // Energy ring
      const energyFraction = Math.max(0, Math.min(1.0, agent.energy / 100.0));
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(ax, ay, radius, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * energyFraction);
      ctx.stroke();
    }
  }, [viewerState, selectedAgentId, width, height]);

  useEffect(() => {
    draw();
  }, [draw]);

  // Handle click on canvas to select agent
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!viewerState) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const gridW = viewerState.width || 20;
    const gridH = viewerState.height || 20;
    const cellSize = Math.min(width / gridW, height / gridH);

    const cellX = Math.floor(clickX / cellSize);
    const cellY = Math.floor(clickY / cellSize);

    // Check if an agent occupies this cell
    const clickedAgent = viewerState.agents.find((a) => a.x === cellX && a.y === cellY);
    if (clickedAgent) {
      onSelectAgent(clickedAgent.agent_id);
    }
  };

  return (
    <div className="relative border border-research-border rounded-lg overflow-hidden bg-research-panel shadow-2xl">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        onClick={handleCanvasClick}
        className="cursor-crosshair block"
      />
    </div>
  );
};
