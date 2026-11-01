import React from 'react';
import { MessageSquare, AlertTriangle, Megaphone, BookOpen } from 'lucide-react';
import { VillageDialogueEventDTO } from '../../../types/village';

interface SpeechBubbleOverlayProps {
  events: VillageDialogueEventDTO[];
  selectedAgentId: string | null;
  onSelectAgent: (agentId: string) => void;
}

export const SpeechBubbleOverlay: React.FC<SpeechBubbleOverlayProps> = ({
  events,
  selectedAgentId,
  onSelectAgent,
}) => {
  const recent = events.slice(-8).reverse();

  const getBadge = (ev: VillageDialogueEventDTO) => {
    if (ev.event_type === 'GossipEvent') {
      return {
        icon: <AlertTriangle size={12} className="text-amber-400" />,
        label: 'GOSSIP',
        cls: 'border-amber-500/40 bg-amber-950/30 text-amber-200',
      };
    }
    if (ev.event_type === 'PublicDeclaration') {
      return {
        icon: <Megaphone size={12} className="text-emerald-400" />,
        label: 'PUBLIC',
        cls: 'border-emerald-500/40 bg-emerald-950/30 text-emerald-200',
      };
    }
    if (ev.event_type === 'DiaryEntry') {
      return {
        icon: <BookOpen size={12} className="text-rose-400" />,
        label: 'COVERT DIARY',
        cls: 'border-rose-500/40 bg-rose-950/30 text-rose-200',
      };
    }
    return {
      icon: <MessageSquare size={12} className="text-cyan-400" />,
      label: 'SPEECH',
      cls: 'border-cyan-500/30 bg-slate-900/80 text-slate-200',
    };
  };

  return (
    <div className="bg-research-panel border border-research-border rounded-lg p-3.5 space-y-2.5">
      <div className="flex items-center justify-between border-b border-research-border pb-2">
        <div className="flex items-center space-x-2">
          <MessageSquare size={15} className="text-amber-400" />
          <h3 className="text-xs font-bold uppercase tracking-wider text-slate-200">
            Spatial Dialogue &amp; Gossip Bus
          </h3>
        </div>
        <span className="text-[10px] font-mono text-slate-400">
          Dual-Mode Persona Engine (Symbolic / LLM)
        </span>
      </div>

      {recent.length === 0 ? (
        <div className="text-xs text-slate-500 py-4 text-center font-mono">
          Step the village simulation to broadcast spatial villager dialogue and gossip.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 max-h-48 overflow-y-auto pr-1">
          {recent.map((ev, idx) => {
            const badge = getBadge(ev);
            const isSelected = ev.speaker_id === selectedAgentId;
            return (
              <div
                key={`${ev.step}-${ev.speaker_id}-${idx}`}
                onClick={() => onSelectAgent(ev.speaker_id)}
                className={`cursor-pointer border rounded-md p-2.5 transition ${badge.cls} ${
                  isSelected ? 'ring-1 ring-amber-400' : 'hover:border-slate-500'
                }`}
              >
                <div className="flex items-center justify-between text-[11px] font-mono mb-1">
                  <div className="flex items-center space-x-1.5">
                    {badge.icon}
                    <span className="font-bold text-slate-100">{ev.speaker_name}</span>
                    <span className="text-slate-400">({ev.speaker_role})</span>
                    {ev.recipient_name && (
                      <span className="text-slate-400">&rarr; {ev.recipient_name}</span>
                    )}
                  </div>
                  <div className="flex items-center space-x-1.5">
                    <span className="px-1.5 py-0.5 rounded bg-black/30 text-[9px] font-semibold">
                      {badge.label}
                    </span>
                    <span className="text-[10px] text-slate-400">t={ev.step}</span>
                  </div>
                </div>
                <p className="text-xs italic leading-snug">&ldquo;{ev.utterance}&rdquo;</p>
                <div className="mt-1.5 flex items-center justify-between text-[10px] font-mono text-slate-400">
                  <span>Action: {ev.action_chosen}</span>
                  <span>
                    Trust: {(ev.trust_score * 100).toFixed(0)}% · Mode: {ev.mode}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

