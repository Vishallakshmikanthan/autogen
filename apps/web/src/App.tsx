import React, { useState, useEffect } from 'react';
import { NavigationSidebar, StudioTab } from './components/layout/NavigationSidebar';
import { TelemetryHeader } from './components/layout/TelemetryHeader';
import { LiveSimulationView } from './components/slices/Slice1LiveSimulation/LiveSimulationView';
import { DecisionTraceViewer } from './components/slices/Slice2DecisionTraces/DecisionTraceViewer';
import { TruthVsBeliefSplit } from './components/slices/Slice2DecisionTraces/TruthVsBeliefSplit';
import { WorldBuilderView } from './components/slices/Slice3WorldBuilder/WorldBuilderView';
import { AlgorithmLabView } from './components/slices/Slice4AlgorithmLab/AlgorithmLabView';
import { ExperimentManagerView } from './components/slices/Slice5ExperimentAnalytics/ExperimentManagerView';
import { useSimulationStream } from './api/useSimulationStream';
import { createSimulation } from './api/client';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<StudioTab>('LIVE_SIMULATION');
  const [activeSimId, setActiveSimId] = useState<string | null>(null);

  // Initialize a default simulation session on mount if none active
  useEffect(() => {
    let mounted = true;
    async function initDefaultSim() {
      try {
        const sim = await createSimulation({
          name: 'research_studio_primary',
          seed: 42,
          width: 20,
          height: 20,
          max_steps: 200,
          agents: [
            { agent_id: 'hybrid_agent_1', agent_type: 'simple_reflex' },
            { agent_id: 'random_forager_2', agent_type: 'random' },
          ],
        });
        if (mounted) {
          setActiveSimId(sim.id);
        }
      } catch (err) {
        console.warn('Could not auto-connect to backend simulation. Running in offline/spec mode.', err);
      }
    }
    initDefaultSim();
    return () => {
      mounted = false;
    };
  }, []);

  const { viewerState } = useSimulationStream(activeSimId);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-research-bg text-slate-100 font-sans">
      {/* Navigation Sidebar */}
      <NavigationSidebar activeTab={activeTab} onSelectTab={setActiveTab} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <TelemetryHeader activeSimId={activeSimId} />

        <main className="flex-1 overflow-y-auto p-6">
          {activeTab === 'LIVE_SIMULATION' && (
            <LiveSimulationView simId={activeSimId} />
          )}

          {activeTab === 'DECISION_TRACES' && (
            <div className="space-y-6">
              <TruthVsBeliefSplit
                viewerState={viewerState}
                selectedAgentId={viewerState?.agents[0]?.agent_id || null}
              />
              <DecisionTraceViewer trace={viewerState?.decision_trace} />
            </div>
          )}

          {activeTab === 'WORLD_BUILDER' && (
            <WorldBuilderView
              onSimulationCreated={(newSimId) => {
                setActiveSimId(newSimId);
                setActiveTab('LIVE_SIMULATION');
              }}
            />
          )}

          {activeTab === 'ALGORITHM_LAB' && (
            <AlgorithmLabView />
          )}

          {activeTab === 'EXPERIMENTS' && (
            <ExperimentManagerView />
          )}
        </main>
      </div>
    </div>
  );
};

export default App;

