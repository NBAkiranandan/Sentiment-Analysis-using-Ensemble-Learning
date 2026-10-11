import React, { useState } from 'react';
import { 
  GitCompare, 
  Trophy, 
  BarChart2,
  CheckCircle2,
  Clock,
  Zap,
  ArrowUpDown
} from 'lucide-react';
import { 
  BarChart as ReBarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend 
} from 'recharts';
import type { ModelComparisonData, ModelMetric } from '../types';

interface ComparisonProps {
  data: ModelComparisonData | null;
  loading: boolean;
}

type SortField = 'name' | 'accuracy' | 'precision' | 'recall' | 'f1_score' | 'training_time' | 'prediction_time';

export const ComparisonPage: React.FC<ComparisonProps> = ({ data, loading }) => {
  const [sortField, setSortField] = useState<SortField>('f1_score');
  const [sortAsc, setSortAsc] = useState<boolean>(false);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] gap-4">
        <div className="w-10 h-10 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
        <p className="text-neutral-400 text-xs font-mono">Loading model comparison metrics...</p>
      </div>
    );
  }

  if (!data || (!data.logistic_regression && !data.random_forest)) {
    return (
      <div className="obsidian-card p-8 rounded-2xl text-center max-w-lg mx-auto my-12 border border-neutral-800">
        <p className="text-white font-medium">No trained models available for comparison.</p>
        <p className="text-xs text-neutral-400 mt-1">Please train the models first using the top header button.</p>
      </div>
    );
  }

  const { logistic_regression: lr, random_forest: rf, adaboost: ada, selected_model } = data;

  const modelsList = [
    { key: 'logistic_regression', name: 'Logistic Regression', type: 'Baseline', metric: lr, color: '#60a5fa' },
    { key: 'random_forest', name: 'Random Forest', type: 'Bagging', metric: rf, color: '#c084fc' },
    { key: 'adaboost', name: 'AdaBoost', type: 'Boosting', metric: ada, color: '#fbbf24' },
  ].filter(m => m.metric != null);

  // Sorting logic for table
  const sortedModels = [...modelsList].sort((a, b) => {
    let valA: any = 0;
    let valB: any = 0;

    if (sortField === 'name') {
      valA = a.name;
      valB = b.name;
    } else {
      valA = (a.metric as any)?.[sortField] || 0;
      valB = (b.metric as any)?.[sortField] || 0;
    }

    if (valA < valB) return sortAsc ? -1 : 1;
    if (valA > valB) return sortAsc ? 1 : -1;
    return 0;
  });

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const chartData = [
    { metric: 'Accuracy', 'Logistic Regression': lr?.accuracy || 0, 'Random Forest': rf?.accuracy || 0, AdaBoost: ada?.accuracy || 0 },
    { metric: 'Precision', 'Logistic Regression': lr?.precision || 0, 'Random Forest': rf?.precision || 0, AdaBoost: ada?.precision || 0 },
    { metric: 'Recall', 'Logistic Regression': lr?.recall || 0, 'Random Forest': rf?.recall || 0, AdaBoost: ada?.recall || 0 },
    { metric: 'F1-Score', 'Logistic Regression': lr?.f1_score || 0, 'Random Forest': rf?.f1_score || 0, AdaBoost: ada?.f1_score || 0 },
  ];

  const renderConfusionMatrix = (matrix: number[][], modelName: string, modelType: string) => {
    // [[TN, FP], [FN, TP]]
    const tn = matrix?.[0]?.[0] || 0;
    const fp = matrix?.[0]?.[1] || 0;
    const fn = matrix?.[1]?.[0] || 0;
    const tp = matrix?.[1]?.[1] || 0;
    const total = tn + fp + fn + tp;

    return (
      <div className="obsidian-card p-5 rounded-2xl border border-neutral-800 space-y-4">
        <div className="flex items-center justify-between">
          <h4 className="text-sm font-bold text-white flex items-center gap-2">
            <BarChart2 className="w-4 h-4 text-indigo-400" />
            {modelName}
          </h4>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-neutral-900 border border-neutral-800 text-neutral-400">
            {modelType}
          </span>
        </div>
        
        {/* Confusion Matrix Visual Grid with Explicit Labels */}
        <div className="bg-neutral-950 p-3 rounded-xl border border-neutral-800/90 space-y-3">
          <div className="text-center text-[11px] font-mono text-neutral-400 tracking-wider uppercase">
            Predicted Class → (Negative vs Positive)
          </div>

          <div className="grid grid-cols-2 gap-2 text-center text-xs">
            {/* True Negative */}
            <div className="bg-neutral-900/90 p-3 rounded-lg border border-emerald-500/30">
              <span className="text-neutral-400 font-mono text-[10px] block mb-1">Actual Negative → Predicted Negative</span>
              <span className="text-emerald-400 font-bold font-mono text-lg block">{tn.toLocaleString()}</span>
              <span className="text-[10px] text-neutral-400 font-mono">True Negative (TN)</span>
            </div>

            {/* False Positive */}
            <div className="bg-neutral-900/90 p-3 rounded-lg border border-rose-500/30">
              <span className="text-neutral-400 font-mono text-[10px] block mb-1">Actual Negative → Predicted Positive</span>
              <span className="text-rose-400 font-bold font-mono text-lg block">{fp.toLocaleString()}</span>
              <span className="text-[10px] text-neutral-400 font-mono">False Positive (FP)</span>
            </div>

            {/* False Negative */}
            <div className="bg-neutral-900/90 p-3 rounded-lg border border-rose-500/30">
              <span className="text-neutral-400 font-mono text-[10px] block mb-1">Actual Positive → Predicted Negative</span>
              <span className="text-rose-400 font-bold font-mono text-lg block">{fn.toLocaleString()}</span>
              <span className="text-[10px] text-neutral-400 font-mono">False Negative (FN)</span>
            </div>

            {/* True Positive */}
            <div className="bg-neutral-900/90 p-3 rounded-lg border border-emerald-500/30">
              <span className="text-neutral-400 font-mono text-[10px] block mb-1">Actual Positive → Predicted Positive</span>
              <span className="text-emerald-400 font-bold font-mono text-lg block">{tp.toLocaleString()}</span>
              <span className="text-[10px] text-neutral-400 font-mono">True Positive (TP)</span>
            </div>
          </div>
        </div>

        <div className="flex justify-between text-[11px] font-mono text-neutral-400">
          <span>Total Test Samples: {total.toLocaleString()}</span>
          <span>Correct: {((tn + tp) / (total || 1) * 100).toFixed(1)}%</span>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-8 animate-fadeIn max-w-6xl mx-auto">
      
      {/* Page Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold font-heading text-white tracking-tight flex items-center gap-3">
          <GitCompare className="w-6 h-6 text-white" />
          Three-Model Benchmark & Comparison
        </h1>
        <p className="text-neutral-400 text-sm mt-1">
          Fair empirical evaluation of <span className="text-white font-medium">Logistic Regression (Baseline)</span>, <span className="text-white font-medium">Random Forest (Bagging)</span>, and <span className="text-white font-medium">AdaBoost (Boosting)</span> trained on identical TF-IDF feature splits.
        </p>
      </div>

      {/* Dynamic Model Winner Banner */}
      <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className="w-11 h-11 rounded-xl bg-neutral-900 border border-neutral-700 text-amber-300 flex items-center justify-center shrink-0">
            <Trophy className="w-5 h-5" />
          </div>
          <div>
            <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">Top Performing Model (Selected)</span>
            <h3 className="text-xl font-bold font-heading text-white">{selected_model}</h3>
            <p className="text-xs text-neutral-400">Determined automatically by highest test set F1-Score</p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          {modelsList.map(m => (
            <div key={m.key} className={`px-3 py-2 rounded-xl border ${selected_model === m.name ? 'bg-emerald-950/60 border-emerald-500/60 text-white' : 'bg-neutral-950 border-neutral-800 text-neutral-400'}`}>
              <span className="font-mono block text-[10px]">{m.name}</span>
              <span className="font-bold font-mono text-sm">{(m.metric.f1_score * 100).toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Model Comparison Table */}
      <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-base font-bold font-heading text-white">Model Comparison Matrix</h3>
            <p className="text-xs text-neutral-400">Click column headers to sort models by performance metric</p>
          </div>
          <span className="text-xs font-mono text-neutral-400 bg-neutral-900 px-3 py-1 rounded-lg border border-neutral-800">
            Same 80/20 Stratified Split (random_state=42)
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400 font-mono uppercase tracking-wider">
                <th onClick={() => handleSort('name')} className="p-3 cursor-pointer hover:text-white">
                  <div className="flex items-center gap-1">Model <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th className="p-3">Type</th>
                <th onClick={() => handleSort('accuracy')} className="p-3 cursor-pointer hover:text-white text-right">
                  <div className="flex items-center justify-end gap-1">Accuracy <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th onClick={() => handleSort('precision')} className="p-3 cursor-pointer hover:text-white text-right">
                  <div className="flex items-center justify-end gap-1">Precision <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th onClick={() => handleSort('recall')} className="p-3 cursor-pointer hover:text-white text-right">
                  <div className="flex items-center justify-end gap-1">Recall <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th onClick={() => handleSort('f1_score')} className="p-3 cursor-pointer hover:text-white text-right">
                  <div className="flex items-center justify-end gap-1">F1-Score <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th onClick={() => handleSort('training_time')} className="p-3 cursor-pointer hover:text-white text-right">
                  <div className="flex items-center justify-end gap-1">Train Time <ArrowUpDown className="w-3 h-3" /></div>
                </th>
                <th onClick={() => handleSort('prediction_time')} className="p-3 cursor-pointer hover:text-white text-right">
                  <div className="flex items-center justify-end gap-1">Pred Time <ArrowUpDown className="w-3 h-3" /></div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-900 font-mono">
              {sortedModels.map((item) => {
                const isSelected = selected_model === item.name;
                return (
                  <tr key={item.key} className={`hover:bg-neutral-900/50 transition-colors ${isSelected ? 'bg-emerald-950/20' : ''}`}>
                    <td className="p-3 font-semibold text-white flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: item.color }} />
                      {item.name}
                      {isSelected && (
                        <span className="text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800 px-1.5 py-0.5 rounded">
                          Selected Winner
                        </span>
                      )}
                    </td>
                    <td className="p-3 text-neutral-400">{item.type}</td>
                    <td className="p-3 text-right text-white font-bold">{(item.metric.accuracy * 100).toFixed(2)}%</td>
                    <td className="p-3 text-right text-white">{(item.metric.precision * 100).toFixed(2)}%</td>
                    <td className="p-3 text-right text-white">{(item.metric.recall * 100).toFixed(2)}%</td>
                    <td className="p-3 text-right text-emerald-400 font-bold">{(item.metric.f1_score * 100).toFixed(2)}%</td>
                    <td className="p-3 text-right text-neutral-300">{item.metric.training_time.toFixed(2)}s</td>
                    <td className="p-3 text-right text-neutral-300">{(item.metric.prediction_time * 1000).toFixed(1)}ms</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Metrics Grouped Bar Chart */}
      <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-4">
        <h3 className="text-base font-bold font-heading text-white">Direct Model Metric Comparison</h3>
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ReBarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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

      {/* Confusion Matrices Grid for All 3 Models */}
      <div className="space-y-4">
        <h3 className="text-base font-bold font-heading text-white flex items-center gap-2">
          <BarChart2 className="w-4 h-4 text-emerald-400" />
          Model Confusion Matrices
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {lr && renderConfusionMatrix(lr.confusion_matrix, "Logistic Regression", "Baseline")}
          {rf && renderConfusionMatrix(rf.confusion_matrix, "Random Forest", "Bagging")}
          {ada && renderConfusionMatrix(ada.confusion_matrix, "AdaBoost", "Boosting")}
        </div>
      </div>

    </div>
  );
};
