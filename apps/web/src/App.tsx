import React, { useState } from 'react';
import { StudioTab } from './components/layout/NavigationSidebar';
import { VillageSocietyView } from './components/slices/Slice6VillageSociety/VillageSocietyView';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<StudioTab>('VILLAGE_SOCIETY');

  return (
    <VillageSocietyView
      activeTab={activeTab}
      onSelectTab={setActiveTab}
    />
  );
};

export default App;
