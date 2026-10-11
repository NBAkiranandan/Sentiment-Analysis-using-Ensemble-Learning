import React from 'react';
import { 
  Trophy, 
  Target, 
  BarChart, 
  Zap, 
  Layers, 
  CheckCircle2, 
  PieChart as PieIcon,
  Activity,
  ArrowUpRight,
  Cpu,
  GitBranch
} from 'lucide-react';
import { 
  BarChart as ReBarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  PieChart, 
  Pie, 
  Cell, 
  Legend 
} from 'recharts';
import type { ModelComparisonData } from '../types';

interface DashboardProps {
  data: ModelComparisonData | null;
  loading: boolean;
  onNavigate: (tab: string) => void;
  onOpenTrainModal: () => void;
}

export const DashboardPage: React.FC<DashboardProps> = ({
  data,
  loading,
  onNavigate,
  onOpenTrainModal,
}) => {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh] gap-4">
        <div className="w-10 h-10 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
        <p className="text-neutral-400 text-xs font-mono">Loading model comparison metrics...</p>
      </div>
    );
  }

  if (!data || (!data.logistic_regression && !data.random_forest)) {
    return (
      <div className="obsidian-card rounded-2xl p-10 text-center max-w-2xl mx-auto my-12 border border-neutral-800">
        <div className="w-14 h-14 bg-neutral-900 border border-neutral-700/80 rounded-2xl flex items-center justify-center mx-auto mb-4 text-white">
          <Layers className="w-7 h-7" />
        </div>
        <h3 className="text-2xl font-bold font-heading text-white mb-2">Models Are Not Trained Yet</h3>
        <p className="text-neutral-400 mb-6 text-sm max-w-md mx-auto">
          Train Logistic Regression (Baseline), Random Forest (Bagging), and AdaBoost (Boosting) on the Sentiment140 corpus to view comparative metrics.
        </p>
        <button
          onClick={onOpenTrainModal}
          className="px-6 py-3 bg-white hover:bg-neutral-200 text-black font-semibold rounded-xl text-xs transition-all shadow-md active:scale-95"
        >
          Train ML Models
        </button>
      </div>
    );
  }

  const { dataset_stats, logistic_regression: lr, random_forest: rf, adaboost: ada, selected_model } = data;
  
  // Identify winning model metric object
  let bestModel = lr || rf || ada;
  if (selected_model === 'Logistic Regression' && lr) bestModel = lr;
  else if (selected_model === 'Random Forest' && rf) bestModel = rf;
  else if (selected_model === 'AdaBoost' && ada) bestModel = ada;

  // Pie chart data
  const pieData = [
    { name: 'Positive Sentiment', value: dataset_stats?.positive_samples || 800000, color: '#22c55e' },
    { name: 'Negative Sentiment', value: dataset_stats?.negative_samples || 800000, color: '#f43f5e' },
  ];

  // Grouped Bar chart data
  const metricsChartData = [
    { metric: 'Accuracy', 'Logistic Regression': lr?.accuracy || 0, 'Random Forest': rf?.accuracy || 0, AdaBoost: ada?.accuracy || 0 },
    { metric: 'Precision', 'Logistic Regression': lr?.precision || 0, 'Random Forest': rf?.precision || 0, AdaBoost: ada?.precision || 0 },
    { metric: 'Recall', 'Logistic Regression': lr?.recall || 0, 'Random Forest': rf?.recall || 0, AdaBoost: ada?.recall || 0 },
    { metric: 'F1-Score', 'Logistic Regression': lr?.f1_score || 0, 'Random Forest': rf?.f1_score || 0, AdaBoost: ada?.f1_score || 0 },
  ];

  return (
    <div className="space-y-8 animate-fadeIn">
      
      {/* Hero Banner */}
      <div className="relative overflow-hidden obsidian-card rounded-3xl p-8 sm:p-10 border border-neutral-800 shadow-2xl">
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-neutral-900 border border-neutral-800 text-neutral-300 text-xs font-mono mb-4">
            <Activity className="w-3.5 h-3.5 text-neutral-400" />
            Machine Learning & Ensemble Learning System
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold font-heading text-white tracking-tight mb-3">
            Sentiment Analysis & Model Comparison System
          </h1>
          <p className="text-neutral-400 text-sm sm:text-base leading-relaxed mb-6">
            Comprehensive evaluation comparing <span className="text-white font-semibold">Logistic Regression (Baseline)</span>, <span className="text-white font-semibold font-mono">Random Forest (Bagging)</span>, and <span className="text-white font-semibold font-mono">AdaBoost (Boosting)</span> trained on TF-IDF extracted features from the Sentiment140 dataset.
          </p>

          <div className="flex flex-wrap gap-3">
            <button
              onClick={() => onNavigate('prediction')}
              className="flex items-center gap-2 px-5 py-2.5 bg-white hover:bg-neutral-200 text-black font-semibold rounded-xl text-xs transition-all shadow-md active:scale-95"
            >
              Single & Multi-Model Predictor
              <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => onNavigate('comparison')}
              className="flex items-center gap-2 px-5 py-2.5 bg-neutral-900 hover:bg-neutral-800 text-white font-medium rounded-xl text-xs transition-all border border-neutral-700/80 active:scale-95"
            >
              Full Model Benchmark
            </button>
          </div>
        </div>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        
        {/* Total Samples */}
        <div className="obsidian-card p-6 rounded-2xl obsidian-card-hover border border-neutral-800">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">Dataset Records</span>
            <div className="w-8 h-8 rounded-lg bg-neutral-900 border border-neutral-800 text-neutral-300 flex items-center justify-center">
              <Layers className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl sm:text-3xl font-bold font-heading text-white mb-1">
            {dataset_stats?.total_samples.toLocaleString() || '1,600,000'}
          </div>
          <p className="text-xs text-neutral-400">
            {dataset_stats?.training_mode || (dataset_stats?.is_sampled ? `Sample (${dataset_stats.sample_size.toLocaleString()})` : 'Full Sentiment140 Dataset')}
          </p>
        </div>

        {/* Selected Best Model */}
        <div className="obsidian-card p-6 rounded-2xl obsidian-card-hover border border-neutral-800">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">Selected Model</span>
            <div className="w-8 h-8 rounded-lg bg-neutral-900 border border-neutral-800 text-amber-300 flex items-center justify-center">
              <Trophy className="w-4 h-4" />
            </div>
          </div>
          <div className="text-xl sm:text-2xl font-bold font-heading text-white mb-1">
            {selected_model || 'Logistic Regression'}
          </div>
          <p className="text-xs text-neutral-400">Dynamically selected via F1-Score</p>
        </div>

        {/* F1-Score */}
        <div className="obsidian-card p-6 rounded-2xl obsidian-card-hover border border-neutral-800">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider font-semibold">Top F1-Score</span>
            <div className="w-8 h-8 rounded-lg bg-neutral-900 border border-neutral-800 text-emerald-400 flex items-center justify-center">
              <Target className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl sm:text-3xl font-bold font-heading text-emerald-400 mb-1">
            {bestModel ? (bestModel.f1_score * 100).toFixed(2) : '0.00'}%
          </div>
          <p className="text-xs text-neutral-400">Harmonic mean on test split</p>
        </div>

        {/* Best Model Accuracy */}
        <div className="obsidian-card p-6 rounded-2xl obsidian-card-hover border border-neutral-800">
          <div className="flex items-center justify-between mb-3">
            <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">Top Accuracy</span>
            <div className="w-8 h-8 rounded-lg bg-neutral-900 border border-neutral-800 text-white flex items-center justify-center">
              <Zap className="w-4 h-4" />
            </div>
          </div>
          <div className="text-2xl sm:text-3xl font-bold font-heading text-white mb-1">
            {bestModel ? (bestModel.accuracy * 100).toFixed(2) : '0.00'}%
          </div>
          <p className="text-xs text-neutral-400">Test set accuracy score</p>
        </div>

      </div>

      {/* Model Cards Grid: Logistic Regression, Random Forest, AdaBoost */}
      <div className="space-y-4">
        <h3 className="text-lg font-bold font-heading text-white flex items-center gap-2">
          <Cpu className="w-5 h-5 text-indigo-400" />
          Model Architecture Summary Cards
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          
          {/* Logistic Regression Card */}
          <div className={`obsidian-card p-6 rounded-2xl border transition-all ${selected_model === 'Logistic Regression' ? 'border-emerald-500/60 shadow-lg shadow-emerald-950/20' : 'border-neutral-800'}`}>
            <div className="flex items-center justify-between mb-3">
              <span className="px-2.5 py-1 rounded-md bg-blue-950/80 border border-blue-800/60 text-blue-300 text-xs font-mono font-semibold">
                Baseline Classifier
              </span>
              {selected_model === 'Logistic Regression' && (
                <span className="flex items-center gap-1 text-xs font-mono text-emerald-400 font-semibold">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Selected
                </span>
              )}
            </div>
            <h4 className="text-lg font-bold font-heading text-white mb-1">Logistic Regression</h4>
            <p className="text-xs text-neutral-400 mb-4">Linear Model Baseline for TF-IDF Text Vectors</p>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-neutral-950 p-3 rounded-xl border border-neutral-800">
              <div><span className="text-neutral-500 block">Accuracy</span> <span className="text-white font-bold">{lr ? (lr.accuracy * 100).toFixed(2) : '--'}%</span></div>
              <div><span className="text-neutral-500 block">Precision</span> <span className="text-white font-bold">{lr ? (lr.precision * 100).toFixed(2) : '--'}%</span></div>
              <div><span className="text-neutral-500 block">Recall</span> <span className="text-white font-bold">{lr ? (lr.recall * 100).toFixed(2) : '--'}%</span></div>
              <div><span className="text-neutral-500 block">F1-Score</span> <span className="text-emerald-400 font-bold">{lr ? (lr.f1_score * 100).toFixed(2) : '--'}%</span></div>
            </div>
            <div className="mt-3 text-[11px] text-neutral-400 font-mono flex justify-between">
              <span>Train time: {lr ? `${lr.training_time.toFixed(2)}s` : '--'}</span>
              <span>Pred time: {lr ? `${(lr.prediction_time * 1000).toFixed(1)}ms` : '--'}</span>
            </div>
          </div>

          {/* Random Forest Card */}
          <div className={`obsidian-card p-6 rounded-2xl border transition-all ${selected_model === 'Random Forest' ? 'border-emerald-500/60 shadow-lg shadow-emerald-950/20' : 'border-neutral-800'}`}>
            <div className="flex items-center justify-between mb-3">
              <span className="px-2.5 py-1 rounded-md bg-purple-950/80 border border-purple-800/60 text-purple-300 text-xs font-mono font-semibold">
                Bagging Ensemble
              </span>
              {selected_model === 'Random Forest' && (
                <span className="flex items-center gap-1 text-xs font-mono text-emerald-400 font-semibold">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Selected
                </span>
              )}
            </div>
            <h4 className="text-lg font-bold font-heading text-white mb-1">Random Forest</h4>
            <p className="text-xs text-neutral-400 mb-4">Bootstrap Aggregating Ensemble (100 Decision Trees)</p>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-neutral-950 p-3 rounded-xl border border-neutral-800">
              <div><span className="text-neutral-500 block">Accuracy</span> <span className="text-white font-bold">{rf ? (rf.accuracy * 100).toFixed(2) : '--'}%</span></div>
              <div><span className="text-neutral-500 block">Precision</span> <span className="text-white font-bold">{rf ? (rf.precision * 100).toFixed(2) : '--'}%</span></div>
              <div><span className="text-neutral-500 block">Recall</span> <span className="text-white font-bold">{rf ? (rf.recall * 100).toFixed(2) : '--'}%</span></div>
              <div><span className="text-neutral-500 block">F1-Score</span> <span className="text-emerald-400 font-bold">{rf ? (rf.f1_score * 100).toFixed(2) : '--'}%</span></div>
            </div>
            <div className="mt-3 text-[11px] text-neutral-400 font-mono flex justify-between">
              <span>Train time: {rf ? `${rf.training_time.toFixed(2)}s` : '--'}</span>
              <span>Pred time: {rf ? `${(rf.prediction_time * 1000).toFixed(1)}ms` : '--'}</span>
            </div>
          </div>

          {/* AdaBoost Card */}
          <div className={`obsidian-card p-6 rounded-2xl border transition-all ${selected_model === 'AdaBoost' ? 'border-emerald-500/60 shadow-lg shadow-emerald-950/20' : 'border-neutral-800'}`}>
            <div className="flex items-center justify-between mb-3">
              <span className="px-2.5 py-1 rounded-md bg-amber-950/80 border border-amber-800/60 text-amber-300 text-xs font-mono font-semibold">
                Boosting Ensemble
              </span>
              {selected_model === 'AdaBoost' && (
                <span className="flex items-center gap-1 text-xs font-mono text-emerald-400 font-semibold">
                  <CheckCircle2 className="w-3.5 h-3.5" /> Selected
                </span>
              )}
            </div>
            <h4 className="text-lg font-bold font-heading text-white mb-1">AdaBoost</h4>
            <p className="text-xs text-neutral-400 mb-4">Adaptive Boosting Sequential Weak Learner Ensemble</p>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono bg-neutral-950 p-3 rounded-xl border border-neutral-800">
              <div><span className="text-neutral-500 block">Accuracy</span> <span className="text-white font-bold">{ada ? (ada.accuracy * 100).toFixed(2) : '--'}%</span></div>
              <div><span className="text-neutral-500 block">Precision</span> <span className="text-white font-bold">{ada ? (ada.precision * 100).toFixed(2) : '--'}%</span></div>
              <div><span className="text-neutral-500 block">Recall</span> <span className="text-white font-bold">{ada ? (ada.recall * 100).toFixed(2) : '--'}%</span></div>
              <div><span className="text-neutral-500 block">F1-Score</span> <span className="text-emerald-400 font-bold">{ada ? (ada.f1_score * 100).toFixed(2) : '--'}%</span></div>
            </div>
            <div className="mt-3 text-[11px] text-neutral-400 font-mono flex justify-between">
              <span>Train time: {ada ? `${ada.training_time.toFixed(2)}s` : '--'}</span>
              <span>Pred time: {ada ? `${(ada.prediction_time * 1000).toFixed(1)}ms` : '--'}</span>
            </div>
          </div>

        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Chart 1: 3-Model Comparison Bar Chart */}
        <div className="lg:col-span-7 obsidian-card p-6 rounded-2xl border border-neutral-800">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <BarChart className="w-4 h-4 text-neutral-300" />
                Comparative Model Benchmark
              </h3>
              <p className="text-xs text-neutral-400">Baseline vs. Bagging vs. Boosting classification performance</p>
            </div>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ReBarChart data={metricsChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#222225" />
                <XAxis dataKey="metric" stroke="#71717a" fontSize={11} fontFamily="var(--font-sans)" />
                <YAxis domain={[0, 1]} stroke="#71717a" fontSize={11} fontFamily="var(--font-sans)" tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', borderRadius: '10px', color: '#fff', fontSize: '12px' }}
                  formatter={(val: any) => [`${(Number(val) * 100).toFixed(2)}%`, '']}
                />
                <Legend wrapperStyle={{ fontSize: '11px', color: '#a1a1aa' }} />
                <Bar dataKey="Logistic Regression" fill="#60a5fa" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Random Forest" fill="#c084fc" radius={[4, 4, 0, 0]} />
                <Bar dataKey="AdaBoost" fill="#fbbf24" radius={[4, 4, 0, 0]} />
              </ReBarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart 2: Dataset Distribution Donut Chart */}
        <div className="lg:col-span-5 obsidian-card p-6 rounded-2xl border border-neutral-800">
          <div className="flex items-center justify-between mb-6">
            <div>
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <PieIcon className="w-4 h-4 text-emerald-400" />
                Binary Sentiment Distribution
              </h3>
              <p className="text-xs text-neutral-400">Sentiment140 positive vs. negative ground truth balance</p>
            </div>
          </div>
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={4}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', borderRadius: '10px', color: '#fff', fontSize: '12px' }}
                  formatter={(val: any) => [Number(val).toLocaleString(), 'Samples']}
                />
                <Legend wrapperStyle={{ fontSize: '11px', color: '#a1a1aa' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

      </div>

    </div>
  );
};
