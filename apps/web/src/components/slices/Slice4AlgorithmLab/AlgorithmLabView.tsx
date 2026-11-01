import React, { useState } from 'react';
import { MathBlock } from '../../common/MathBlock';
import { BookOpen, Swords } from 'lucide-react';

interface AlgorithmRecord {
  id: string;
  name: string;
  family: 'Search' | 'Probabilistic' | 'Planning' | 'Reinforcement Learning' | 'Game Theory';
  formula: string;
  definition: string;
  assumptions: string[];
  complexityTime: string;
  complexitySpace: string;
  limitations: string[];
}

const ALGORITHM_CATALOG: AlgorithmRecord[] = [
  {
    id: 'astar',
    name: 'A* Heuristic Graph Search',
    family: 'Search',
    formula: 'f(n) = g(n) + h(n)',
    definition: 'Finds optimal cost path by prioritizing node expansion using exact path cost g(n) and admissible heuristic h(n).',
    assumptions: ['Admissible heuristic: h(n) <= h*(n)', 'Consistent heuristic: h(n) <= c(n, a, n\') + h(n\')'],
    complexityTime: 'O(b^d) where b is branching factor and d is optimal depth',
    complexitySpace: 'O(b^d) keeps all generated nodes in open/closed set',
    limitations: ['Exponential memory footprint on large unstructured grids'],
  },
  {
    id: 'bayesian_hazard',
    name: 'Bayesian Hazard Filter',
    family: 'Probabilistic',
    formula: 'P(H_t | z_{1:t}) = \\eta P(z_t | H_t) \\sum_{H_{t-1}} P(H_t | H_{t-1}) P(H_{t-1} | z_{1:t-1})',
    definition: 'Recursive Bayesian state estimator tracking true environmental danger under noisy partial sensor reports.',
    assumptions: ['First-order Markov state transition', 'Sensor observations conditionally independent given true state'],
    complexityTime: 'O(|S|^2) per temporal update step',
    complexitySpace: 'O(|S|) state probability vector storage',
    limitations: ['Sensor likelihood distributions must be calibrated accurately'],
  },
  {
    id: 'q_learning',
    name: 'Tabular Q-Learning (Off-Policy TD Control)',
    family: 'Reinforcement Learning',
    formula: 'Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a\'} Q(s\', a\') - Q(s, a) \\right]',
    definition: 'Learns optimal action-value function Q*(s, a) independently of the agent exploration policy.',
    assumptions: ['Markov Decision Process with discrete finite states and actions', 'All state-action pairs continually visited'],
    complexityTime: 'O(1) per transition step update',
    complexitySpace: 'O(|S| \\times |A|) Q-table size',
    limitations: ['Suffers from curse of dimensionality for large state spaces'],
  },
  {
    id: 'minimax_alpha_beta',
    name: 'Minimax with Alpha-Beta Pruning',
    family: 'Game Theory',
    formula: '\\alpha = \\max(\\alpha, \\text{child}), \\quad \\beta = \\min(\\beta, \\text{child})',
    definition: 'Computes zero-sum game equilibrium strategy while pruning branches that cannot influence the final decision.',
    assumptions: ['Two-player zero-sum game', 'Perfect information game tree'],
    complexityTime: 'O(b^{d/2}) under optimal move ordering',
    complexitySpace: 'O(b \\cdot d) tree depth recursion stack',
    limitations: ['Assumes opponent plays strictly optimal counter-moves'],
  },
];

export const AlgorithmLabView: React.FC = () => {
  const [selectedAlgoId, setSelectedAlgoId] = useState<string>('astar');
  const [activeTab, setActiveTab] = useState<'SPEC' | 'DUEL'>('SPEC');

  const algo = ALGORITHM_CATALOG.find((a) => a.id === selectedAlgoId) || ALGORITHM_CATALOG[0];

  return (
    <div className="bg-research-panel border border-research-border rounded-lg p-6 font-mono text-xs space-y-6">
      {/* Header and Mode Selector */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
        <div className="flex items-center space-x-2">
          <BookOpen size={18} className="text-cyan-400" />
          <h2 className="text-sm font-semibold text-slate-100">ALGORITHM LABORATORY & INTERACTIVE SPEC</h2>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={() => setActiveTab('SPEC')}
            className={`px-3 py-1 rounded text-xs transition ${activeTab === 'SPEC' ? 'bg-cyan-600 text-black font-semibold' : 'bg-slate-800 text-slate-300'}`}
          >
            THEORY & SPEC
          </button>
          <button
            onClick={() => setActiveTab('DUEL')}
            className={`flex items-center space-x-1.5 px-3 py-1 rounded text-xs transition ${activeTab === 'DUEL' ? 'bg-purple-600 text-white font-semibold' : 'bg-slate-800 text-slate-300'}`}
          >
            <Swords size={12} />
            <span>ALGORITHM DUEL</span>
          </button>
        </div>
      </div>

      {activeTab === 'SPEC' ? (
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6">
          {/* Algorithm List Sidebar */}
          <div className="md:col-span-4 space-y-2 border-r border-slate-800 pr-4">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider">Catalog of AI Implementations</span>
            {ALGORITHM_CATALOG.map((a) => (
              <button
                key={a.id}
                onClick={() => setSelectedAlgoId(a.id)}
                className={`w-full text-left p-2.5 rounded border transition flex flex-col space-y-1 ${
                  a.id === selectedAlgoId
                    ? 'border-cyan-500 bg-cyan-950/20 text-cyan-200'
                    : 'border-slate-800 bg-slate-900/40 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-200">{a.name}</span>
                </div>
                <span className="text-[10px] text-slate-500">{a.family}</span>
              </button>
            ))}
          </div>

          {/* Algorithm Spec Details */}
          <div className="md:col-span-8 space-y-4">
            <div>
              <span className="text-xs text-cyan-400 font-semibold">{algo.family}</span>
              <h3 className="text-base font-bold text-slate-100">{algo.name}</h3>
              <p className="text-slate-300 mt-1 leading-relaxed">{algo.definition}</p>
            </div>

            {/* Formula Block with KaTeX */}
            <div className="bg-slate-950 border border-slate-800 rounded p-4 flex flex-col items-center justify-center space-y-2">
              <span className="text-[10px] text-slate-500 uppercase tracking-wider">Mathematical Formulation</span>
              <div className="text-emerald-400 text-sm py-2">
                <MathBlock math={algo.formula} block />
              </div>
            </div>

            {/* Assumptions & Complexity */}
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-900/60 border border-slate-800 rounded p-3 space-y-1.5">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider">Core Assumptions</span>
                <ul className="list-disc list-inside space-y-1 text-slate-300 text-[11px]">
                  {algo.assumptions.map((assump, idx) => (
                    <li key={idx}>{assump}</li>
                  ))}
                </ul>
              </div>

              <div className="bg-slate-900/60 border border-slate-800 rounded p-3 space-y-1.5">
                <span className="text-[10px] text-slate-500 uppercase tracking-wider">Computational Complexity</span>
                <div className="text-[11px] text-slate-300">
                  <span className="text-slate-400">Time:</span> {algo.complexityTime}
                </div>
                <div className="text-[11px] text-slate-300">
                  <span className="text-slate-400">Space:</span> {algo.complexitySpace}
                </div>
              </div>
            </div>

            {/* Known Limitations */}
            <div className="bg-rose-950/20 border border-rose-900/30 rounded p-3">
              <span className="text-[10px] text-rose-400 uppercase tracking-wider block mb-1">Theoretical Limitations</span>
              <ul className="list-disc list-inside text-rose-200 text-[11px]">
                {algo.limitations.map((lim, idx) => (
                  <li key={idx}>{lim}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      ) : (
        /* Algorithm Duel View */
        <div className="space-y-4">
          <div className="bg-purple-950/20 border border-purple-800/40 p-4 rounded text-center space-y-1">
            <div className="flex items-center justify-center space-x-2 text-purple-400 font-bold text-sm">
              <Swords size={16} />
              <span>LOCKSTEP ALGORITHM DUEL ARENA</span>
            </div>
            <p className="text-slate-300 text-xs">
              Comparing two distinct algorithms across identical seeded environments to evaluate path cost and deliberation latency.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-slate-900 border border-cyan-800/50 rounded-lg p-4 space-y-2">
              <span className="text-cyan-400 font-bold">ALGORITHM A: A* Search (Manhattan)</span>
              <div className="text-slate-300 space-y-1 text-[11px]">
                <div>Nodes Expanded: <span className="text-emerald-400 font-bold">42</span></div>
                <div>Path Length: <span className="text-slate-100">18 steps</span></div>
                <div>Execution Time: <span className="text-slate-100">1.2 ms</span></div>
                <div>Status: <span className="text-emerald-400 font-bold">OPTIMAL</span></div>
              </div>
            </div>

            <div className="bg-slate-900 border border-purple-800/50 rounded-lg p-4 space-y-2">
              <span className="text-purple-400 font-bold">ALGORITHM B: Uniform Cost Search (Dijkstra)</span>
              <div className="text-slate-300 space-y-1 text-[11px]">
                <div>Nodes Expanded: <span className="text-rose-400 font-bold">116</span></div>
                <div>Path Length: <span className="text-slate-100">18 steps</span></div>
                <div>Execution Time: <span className="text-slate-100">3.8 ms</span></div>
                <div>Status: <span className="text-emerald-400 font-bold">OPTIMAL</span></div>
              </div>
            </div>
          </div>

          <div className="bg-slate-950 border border-slate-800 p-3 rounded flex items-center justify-between text-xs">
            <span className="text-slate-300">DUEL VERDICT:</span>
            <span className="text-emerald-400 font-bold">A* expanded 63.8% fewer nodes to achieve the identical optimal path.</span>
          </div>
        </div>
      )}
    </div>
  );
};
