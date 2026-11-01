import React from 'react';

interface MetricBadgeProps {
  label: string;
  value: string | number;
  unit?: string;
  ci?: [number, number];
  colorScheme?: 'truth' | 'belief' | 'memory' | 'veto' | 'neutral';
}

export const MetricBadge: React.FC<MetricBadgeProps> = ({
  label,
  value,
  unit = '',
  ci,
  colorScheme = 'neutral',
}) => {
  const colorMap = {
    truth: 'text-research-truth border-research-truth/30 bg-research-truth/10',
    belief: 'text-research-belief border-research-belief/30 bg-research-belief/10',
    memory: 'text-research-memory border-research-memory/30 bg-research-memory/10',
    veto: 'text-research-veto border-research-veto/30 bg-research-veto/10',
    neutral: 'text-slate-300 border-slate-700 bg-slate-800/40',
  };

  return (
    <div className={`px-2.5 py-1.5 rounded border flex flex-col font-mono text-xs ${colorMap[colorScheme]}`}>
      <span className="text-[10px] uppercase tracking-wider text-slate-400">{label}</span>
      <div className="flex items-baseline space-x-1 mt-0.5">
        <span className="text-sm font-semibold">{value}</span>
        {unit && <span className="text-[10px] text-slate-400">{unit}</span>}
      </div>
      {ci && (
        <span className="text-[9px] text-slate-500 mt-0.5">
          95% CI: [{ci[0].toFixed(2)}, {ci[1].toFixed(2)}]
        </span>
      )}
    </div>
  );
};

