import { useState, useEffect } from 'react';
import { Navbar } from './components/Navbar';
import { DashboardPage } from './pages/DashboardPage';
import { PredictionPage } from './pages/PredictionPage';
import { BatchPage } from './pages/BatchPage';
import { ComparisonPage } from './pages/ComparisonPage';
import { DatasetPage } from './pages/DatasetPage';
import { MethodologyPage } from './pages/MethodologyPage';
import { TrainingModal } from './components/TrainingModal';
import { apiService } from './services/api';
import type { ModelComparisonData, DatasetStats, TrainingStatus } from './types';

export function App() {
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [comparisonData, setComparisonData] = useState<ModelComparisonData | null>(null);
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isTrainModalOpen, setIsTrainModalOpen] = useState<boolean>(false);

  const fetchAppData = async () => {
    setLoading(true);
    try {
      const [compRes, statsRes, statusRes] = await Promise.allSettled([
        apiService.getModelComparison(),
        apiService.getDatasetStats(),
        apiService.getTrainingStatus()
      ]);

      if (compRes.status === 'fulfilled') setComparisonData(compRes.value);
      if (statsRes.status === 'fulfilled') setDatasetStats(statsRes.value);
      if (statusRes.status === 'fulfilled') setTrainingStatus(statusRes.value);
    } catch (err) {
      console.error("Error loading initial data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAppData();
  }, []);

  // Poll training status if training is active
  useEffect(() => {
    let interval: any;
    if (trainingStatus?.status === 'training') {
      interval = setInterval(async () => {
        try {
          const status = await apiService.getTrainingStatus();
          setTrainingStatus(status);
          if (status.status === 'completed') {
            fetchAppData();
          }
        } catch (err) {
          console.error("Failed to poll status:", err);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [trainingStatus?.status]);

  return (
    <div className="min-h-screen bg-black text-white flex flex-col font-sans selection:bg-white selection:text-black relative">
      <div className="fixed inset-0 bg-grid-pattern opacity-40 pointer-events-none z-0"></div>

      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        trainingStatus={trainingStatus}
        onOpenTrainingModal={() => setIsTrainModalOpen(true)}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 relative z-10">
        {activeTab === 'dashboard' && (
          <DashboardPage
            data={comparisonData}
            loading={loading}
            onNavigate={setActiveTab}
            onOpenTrainModal={() => setIsTrainModalOpen(true)}
          />
        )}
        {activeTab === 'prediction' && <PredictionPage />}
        {activeTab === 'batch' && <BatchPage />}
        {activeTab === 'comparison' && (
          <ComparisonPage data={comparisonData} loading={loading} />
        )}
        {activeTab === 'dataset' && (
          <DatasetPage stats={datasetStats} loading={loading} />
        )}
        {activeTab === 'methodology' && <MethodologyPage />}
      </main>

      {/* Training Progress Modal */}
      <TrainingModal
        isOpen={isTrainModalOpen}
        onClose={() => setIsTrainModalOpen(false)}
        status={trainingStatus}
        onRefreshData={fetchAppData}
      />

      {/* Footer */}
      <footer className="border-t border-neutral-900/90 bg-black/90 backdrop-blur-md py-6 text-center text-xs text-neutral-400 relative z-10">
        <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="font-medium text-neutral-300">Sentimix Lab</span>
            <span className="text-neutral-600">—</span>
            <span className="text-neutral-400">Sentiment140 Benchmark System</span>
          </div>
          <p className="text-neutral-500">Random Forest (Bagging) vs. AdaBoost (Boosting) • TF-IDF Vectorization</p>
        </div>
      </footer>

    </div>
  );
}

export default App;
