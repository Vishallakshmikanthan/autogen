import React from 'react';
import { Eye, Brain, Sliders, BookOpen, FlaskConical, Zap } from 'lucide-react';

export type StudioTab = 'LIVE_SIMULATION' | 'DECISION_TRACES' | 'WORLD_BUILDER' | 'ALGORITHM_LAB' | 'EXPERIMENTS';

interface NavigationSidebarProps {
  activeTab: StudioTab;
  onSelectTab: (tab: StudioTab) => void;
}

export const NavigationSidebar: React.FC<NavigationSidebarProps> = ({ activeTab, onSelectTab }) => {
  const items: Array<{ id: StudioTab; label: string; icon: React.ReactNode; badge?: string }> = [
    { id: 'LIVE_SIMULATION', label: 'Live Simulation', icon: <Eye size={16} /> },
    { id: 'DECISION_TRACES', label: 'Decision Traces', icon: <Brain size={16} /> },
    { id: 'WORLD_BUILDER', label: 'World Builder', icon: <Sliders size={16} /> },
    { id: 'ALGORITHM_LAB', label: 'Algorithm Lab', icon: <BookOpen size={16} /> },
    { id: 'EXPERIMENTS', label: 'Experiment Manager', icon: <FlaskConical size={16} /> },
  ];

  return (
    <aside className="w-56 bg-research-panel border-r border-research-border flex flex-col font-mono text-xs">
      <div className="p-4 border-b border-research-border flex items-center space-x-2">
        <Zap size={18} className="text-cyan-400 fill-current" />
        <span className="font-bold text-sm tracking-wider text-slate-100">AUTOGEN AI</span>
      </div>

      <nav className="flex-1 p-2 space-y-1">
        {items.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onSelectTab(item.id)}
              className={`w-full flex items-center space-x-2.5 px-3 py-2 rounded text-left transition ${
                isActive
                  ? 'bg-cyan-500/15 text-cyan-400 font-semibold border-l-2 border-cyan-400'
                  : 'text-slate-400 hover:bg-slate-800/40 hover:text-slate-200'
              }`}
            >
              {item.icon}
              <span className="flex-1 truncate">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-3 border-t border-research-border text-[10px] text-slate-500">
        AUTOGEN Research Platform v0.1.0
      </div>
    </aside>
  );
};
