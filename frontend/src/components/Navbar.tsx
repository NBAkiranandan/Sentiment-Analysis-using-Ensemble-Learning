import React from 'react';
import { 
  BarChart3, 
  MessageSquareText, 
  FileSpreadsheet, 
  GitCompare, 
  Database, 
  Workflow, 
  Play, 
  RefreshCw
} from 'lucide-react';
import type { TrainingStatus } from '../types';

interface NavbarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  trainingStatus: TrainingStatus | null;
  onOpenTrainingModal: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  activeTab,
  setActiveTab,
  trainingStatus,
  onOpenTrainingModal,
}) => {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'prediction', label: 'Predict Sentiment', icon: MessageSquareText },
    { id: 'batch', label: 'Batch Analysis', icon: FileSpreadsheet },
    { id: 'comparison', label: 'Model Comparison', icon: GitCompare },
    { id: 'dataset', label: 'Dataset', icon: Database },
    { id: 'methodology', label: 'Pipeline Architecture', icon: Workflow },
  ];

  const isTraining = trainingStatus?.status === 'training';

  return (
    <header className="sticky top-0 z-40 obsidian-panel border-b border-neutral-900 shadow-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          
          {/* Logo & Title */}
          <div className="flex items-center gap-3 cursor-pointer select-none group" onClick={() => setActiveTab('dashboard')}>
            <div className="w-9 h-9 rounded-xl bg-neutral-900 border border-neutral-700/80 flex items-center justify-center text-white group-hover:border-neutral-500 transition-colors shadow-inner">
              <span className="font-heading font-extrabold text-sm tracking-tighter">SX</span>
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-heading font-bold text-base text-white tracking-wide">SENTIMIX</span>
                <span className="px-1.5 py-0.5 text-[10px] font-mono font-medium bg-neutral-900 text-neutral-300 border border-neutral-800 rounded">
                  v3.0
                </span>
              </div>
              <p className="text-[11px] text-neutral-400 font-sans">Baseline vs. Bagging vs. Boosting</p>
            </div>
          </div>

          {/* Nav items */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-150 ${
                    isActive
                      ? 'bg-neutral-900 text-white border border-neutral-700 shadow-sm'
                      : 'text-neutral-400 hover:text-white hover:bg-neutral-900/60'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-white' : 'text-neutral-500'}`} />
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Action Button: Train Model */}
          <div className="flex items-center gap-3">
            <button
              onClick={onOpenTrainingModal}
              disabled={isTraining}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-lg font-medium text-xs transition-all duration-150 active:scale-95 ${
                isTraining
                  ? 'bg-neutral-900 border border-neutral-700 text-neutral-300 cursor-wait'
                  : 'bg-white hover:bg-neutral-200 text-black font-semibold shadow-sm'
              }`}
            >
              {isTraining ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-neutral-300" />
                  <span>Training ({trainingStatus?.progress || 0}%)</span>
                </>
              ) : (
                <>
                  <Play className="w-3 h-3 fill-current" />
                  <span>Train Models</span>
                </>
              )}
            </button>
          </div>

        </div>

        {/* Mobile Navigation Bar */}
        <div className="md:hidden flex overflow-x-auto py-2 space-x-2 border-t border-neutral-900 scrollbar-none">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-colors ${
                  isActive
                    ? 'bg-neutral-900 text-white border border-neutral-700'
                    : 'text-neutral-400 hover:bg-neutral-900'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                {item.label}
              </button>
            );
          })}
        </div>
      </div>
    </header>
  );
};
