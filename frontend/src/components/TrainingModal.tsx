import React, { useState } from 'react';
import { X, Play, RefreshCw, CheckCircle2, AlertCircle, Cpu, Database } from 'lucide-react';
import type { TrainingStatus } from '../types';
import { apiService } from '../services/api';

interface TrainingModalProps {
  isOpen: boolean;
  onClose: () => void;
  status: TrainingStatus | null;
  onRefreshData: () => void;
  onTrainingStarted?: (status: TrainingStatus) => void;
}

export const TrainingModal: React.FC<TrainingModalProps> = ({
  isOpen,
  onClose,
  status,
  onRefreshData,
  onTrainingStarted,
}) => {
  const [sampleSize, setSampleSize] = useState<number>(50000);
  const [starting, setStarting] = useState<boolean>(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [showConfig, setShowConfig] = useState<boolean>(false);

  if (!isOpen) return null;

  const isTraining = status?.status === 'training';
  const isCompleted = status?.status === 'completed' && !showConfig;
  const isFailed = status?.status === 'failed' && !showConfig;
  const isConfiguring = !isTraining && (!isCompleted && !isFailed || showConfig);

  const handleStart = async () => {
    setStarting(true);
    setLocalError(null);
    setShowConfig(false);
    try {
      const res = await apiService.startTraining(sampleSize);
      if (onTrainingStarted) {
        onTrainingStarted(res);
      }
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message;
      setLocalError(
        msg && msg !== "Network Error"
          ? msg
          : "Failed to connect to backend server (http://localhost:8000). Please ensure Uvicorn is running."
      );
    } finally {
      setStarting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-fadeIn">
      <div className="obsidian-card w-full max-w-lg rounded-3xl p-6 border border-neutral-800 shadow-2xl space-y-6 relative">
        
        {/* Header */}
        <div className="flex items-center justify-between border-b border-neutral-800 pb-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-neutral-900 border border-neutral-700/80 text-white flex items-center justify-center">
              <Cpu className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-base font-bold font-heading text-white">Train Machine Learning Models</h3>
              <p className="text-xs text-neutral-400">Logistic Regression, Random Forest & AdaBoost</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-neutral-400 hover:text-white hover:bg-neutral-900 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Content Body: Config View */}
        {isConfiguring && (
          <div className="space-y-4">
            <p className="text-xs text-neutral-400 leading-relaxed">
              Configure the training corpus sample size. The pipeline will partition a stratified 80/20 train/test split, vectorize n-gram TF-IDF matrices, fit all 3 classifiers (Logistic Regression, Random Forest, AdaBoost), compute metrics, and dynamically select the winner.
            </p>

            <div>
              <label className="text-xs font-mono text-neutral-400 uppercase tracking-wider block mb-2 flex items-center gap-2">
                <Database className="w-3.5 h-3.5 text-neutral-400" />
                Corpus Subsample Configuration
              </label>
              <select
                value={sampleSize}
                onChange={(e) => setSampleSize(Number(e.target.value))}
                className="w-full bg-neutral-950 text-white border border-neutral-700 rounded-xl p-3 text-xs font-medium focus:outline-none focus:border-white transition-colors"
              >
                <option value={20000}>20,000 samples (Fastest dev run)</option>
                <option value={50000}>50,000 samples (Recommended high performance)</option>
                <option value={100000}>100,000 samples (High accuracy evaluation)</option>
                <option value={1600000}>1,600,000 samples (Full Sentiment140 Dataset)</option>
              </select>
            </div>

            {localError && (
              <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-950/40 border border-rose-800/80 text-rose-300 text-xs font-mono">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{localError}</span>
              </div>
            )}
          </div>
        )}

        {/* Training Progress Bar */}
        {isTraining && (
          <div className="space-y-4 text-center py-4">
            <div className="w-12 h-12 bg-neutral-900 border border-neutral-700 text-white rounded-2xl flex items-center justify-center mx-auto">
              <RefreshCw className="w-5 h-5 animate-spin" />
            </div>

            <div>
              <h4 className="text-sm font-semibold font-heading text-white mb-1">
                Training Pipeline in Progress...
              </h4>
              <p className="text-xs text-neutral-400 font-mono">
                {status?.message || "Fitting ML classifiers..."}
              </p>
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between text-xs text-neutral-400 font-mono">
                <span>Stage Progress</span>
                <span className="text-white font-bold">{status?.progress || 0}%</span>
              </div>
              <div className="w-full bg-neutral-950 h-2 rounded-full overflow-hidden p-0.5 border border-neutral-800">
                <div
                  className="bg-white h-full rounded-full transition-all duration-300"
                  style={{ width: `${status?.progress || 0}%` }}
                />
              </div>
            </div>
          </div>
        )}

        {/* Training Completed */}
        {isCompleted && (
          <div className="text-center py-4 space-y-3">
            <div className="w-12 h-12 bg-emerald-950/40 text-emerald-400 rounded-2xl flex items-center justify-center mx-auto border border-emerald-500/30">
              <CheckCircle2 className="w-6 h-6" />
            </div>
            <h4 className="text-base font-bold font-heading text-white">Model Training Complete!</h4>
            <p className="text-xs text-neutral-400 max-w-xs mx-auto leading-relaxed">
              {status?.message || "Logistic Regression, Random Forest, and AdaBoost trained successfully."}
            </p>
          </div>
        )}

        {/* Footer Actions */}
        <div className="flex flex-col gap-2 pt-2 border-t border-neutral-800">
          {isConfiguring && (
            <button
              onClick={handleStart}
              disabled={starting}
              className="w-full py-3 bg-white hover:bg-neutral-200 text-black font-semibold rounded-xl text-xs transition-all shadow-md active:scale-95 flex items-center justify-center gap-2"
            >
              {starting ? <RefreshCw className="w-3.5 h-3.5 animate-spin text-black" /> : <Play className="w-3.5 h-3.5 fill-current" />}
              <span>Start 3-Model Training ({sampleSize.toLocaleString()} samples)</span>
            </button>
          )}

          {isCompleted && (
            <div className="flex flex-col sm:flex-row gap-2">
              <button
                onClick={() => setShowConfig(true)}
                className="flex-1 py-3 bg-neutral-900 hover:bg-neutral-800 text-white font-medium rounded-xl text-xs border border-neutral-700 transition-all"
              >
                Change Data Count & Retrain
              </button>
              <button
                onClick={() => { onRefreshData(); onClose(); }}
                className="flex-1 py-3 bg-white hover:bg-neutral-200 text-black font-semibold rounded-xl text-xs transition-all shadow-md active:scale-95"
              >
                View Updated Dashboard
              </button>
            </div>
          )}

          {isFailed && (
            <div className="flex flex-col sm:flex-row gap-2">
              <button
                onClick={() => setShowConfig(true)}
                className="flex-1 py-3 bg-neutral-900 hover:bg-neutral-800 text-white font-medium rounded-xl text-xs border border-neutral-700"
              >
                Change Data Count
              </button>
              <button
                onClick={handleStart}
                className="flex-1 py-3 bg-rose-600 hover:bg-rose-500 text-white font-semibold rounded-xl text-xs transition-all"
              >
                Retry Training
              </button>
            </div>
          )}
        </div>

      </div>
    </div>
  );
};
