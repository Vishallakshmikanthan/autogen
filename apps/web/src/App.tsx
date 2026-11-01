import React, { useState, useEffect } from 'react';
import { StudioTab } from './components/layout/NavigationSidebar';
import { LiveSimulationView } from './components/slices/Slice1LiveSimulation/LiveSimulationView';
import { DecisionTraceViewer } from './components/slices/Slice2DecisionTraces/DecisionTraceViewer';
import { TruthVsBeliefSplit } from './components/slices/Slice2DecisionTraces/TruthVsBeliefSplit';
import { WorldBuilderView } from './components/slices/Slice3WorldBuilder/WorldBuilderView';
import { AlgorithmLabView } from './components/slices/Slice4AlgorithmLab/AlgorithmLabView';
import { ExperimentManagerView } from './components/slices/Slice5ExperimentAnalytics/ExperimentManagerView';
import { VillageSocietyView } from './components/slices/Slice6VillageSociety/VillageSocietyView';
import { useSimulationStream } from './api/useSimulationStream';
import { createSimulation } from './api/client';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<StudioTab>('VILLAGE_SOCIETY');
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

  const renderOverlayTabContent = () => {
    switch (activeTab) {
      case 'LIVE_SIMULATION':
        return <LiveSimulationView simId={activeSimId} />;
      case 'DECISION_TRACES':
        return (
          <div className="space-y-6">
            <TruthVsBeliefSplit
              viewerState={viewerState}
              selectedAgentId={viewerState?.agents[0]?.agent_id || null}
            />
            <DecisionTraceViewer trace={viewerState?.decision_trace} />
          </div>
        );
      case 'WORLD_BUILDER':
        return (
          <WorldBuilderView
            onSimulationCreated={(newSimId) => {
              setActiveSimId(newSimId);
              setActiveTab('LIVE_SIMULATION');
            }}
          />
        );
      case 'ALGORITHM_LAB':
        return <AlgorithmLabView />;
      case 'EXPERIMENTS':
        return <ExperimentManagerView />;
      default:
        return null;
    }
  };

  return (
    <VillageSocietyView
      activeTab={activeTab}
      onSelectTab={setActiveTab}
      overlayContent={renderOverlayTabContent()}
    />
  );
};

export default App;
