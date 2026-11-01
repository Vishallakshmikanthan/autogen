import React, { useEffect } from 'react';
import { Play, Pause, StepForward, RotateCcw, FastForward } from 'lucide-react';

interface ControlToolbarProps {
  isPaused: boolean;
  speed: number;
  onPlay: () => void;
  onPause: () => void;
  onStep: () => void;
  onReset: () => void;
  onSpeedChange: (speed: number) => void;
  disabled?: boolean;
}

export const ControlToolbar: React.FC<ControlToolbarProps> = ({
  isPaused,
  speed,
  onPlay,
  onPause,
  onStep,
  onReset,
  onSpeedChange,
  disabled = false,
}) => {
  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't intercept if user is typing in an input
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes((e.target as HTMLElement).tagName)) {
        return;
      }

      if (e.code === 'Space') {
        e.preventDefault();
        if (isPaused) onPlay();
        else onPause();
      } else if (e.code === 'ArrowRight' || e.key === '.') {
        e.preventDefault();
        onStep();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        onReset();
      } else if (e.key === ']') {
        e.preventDefault();
        onSpeedChange(Math.min(5.0, speed + 0.5));
      } else if (e.key === '[') {
        e.preventDefault();
        onSpeedChange(Math.max(0.5, speed - 0.5));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isPaused, speed, onPlay, onPause, onStep, onReset, onSpeedChange]);

  return (
    <div className="flex items-center space-x-2 bg-research-panel border border-research-border px-3 py-1.5 rounded-lg text-xs font-mono">
      {/* Play / Pause */}
      {isPaused ? (
        <button
          onClick={onPlay}
          disabled={disabled}
          title="Play (Space)"
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-research-truth/20 text-research-truth hover:bg-research-truth/30 transition disabled:opacity-40"
        >
          <Play size={14} className="fill-current" />
          <span>PLAY</span>
        </button>
      ) : (
        <button
          onClick={onPause}
          disabled={disabled}
          title="Pause (Space)"
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-research-warning/20 text-research-warning hover:bg-research-warning/30 transition disabled:opacity-40"
        >
          <Pause size={14} className="fill-current" />
          <span>PAUSE</span>
        </button>
      )}

      {/* Step Forward */}
      <button
        onClick={onStep}
        disabled={disabled}
        title="Step forward (.)"
        className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded bg-slate-800 text-slate-200 hover:bg-slate-700 transition disabled:opacity-40"
      >
        <StepForward size={14} />
        <span>STEP</span>
      </button>

      {/* Reset */}
      <button
        onClick={onReset}
        disabled={disabled}
        title="Reset simulation (R)"
        className="flex items-center space-x-1.5 px-2.5 py-1.5 rounded bg-slate-800 text-slate-200 hover:bg-slate-700 transition disabled:opacity-40"
      >
        <RotateCcw size={14} />
        <span>RESET</span>
      </button>

      <div className="h-4 w-px bg-slate-700 mx-1" />

      {/* Speed Selector */}
      <div className="flex items-center space-x-1.5 text-slate-300">
        <FastForward size={14} className="text-slate-400" />
        <span>{speed.toFixed(1)}x</span>
        <input
          type="range"
          min="0.5"
          max="4.0"
          step="0.5"
          value={speed}
          onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
          disabled={disabled}
          className="w-16 accent-cyan-500 cursor-pointer"
        />
      </div>
    </div>
  );
};

