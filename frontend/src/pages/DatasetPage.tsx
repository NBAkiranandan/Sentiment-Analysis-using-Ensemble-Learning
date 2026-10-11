import React from 'react';
import { 
  Database, 
  Sliders, 
  FileText, 
  PieChart as PieIcon
} from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import type { DatasetStats } from '../types';

interface DatasetProps {
  stats: DatasetStats | null;
  loading: boolean;
}

export const DatasetPage: React.FC<DatasetProps> = ({ stats, loading }) => {
  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] gap-4">
        <div className="w-10 h-10 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
        <p className="text-neutral-400 text-xs font-mono">Loading Sentiment140 corpus statistics...</p>
      </div>
    );
  }

  const rawTotal = stats?.records_raw || stats?.total_samples || 1600000;
  const total = stats?.total_samples || 1600000;
  const pos = stats?.positive_samples || 800000;
  const neg = stats?.negative_samples || 800000;
  const trainSize = stats?.train_size || Math.round(total * 0.8);
  const testSize = stats?.test_size || Math.round(total * 0.2);
  const trainingMode = stats?.training_mode || (stats?.is_sampled ? `Sample (${total.toLocaleString()})` : 'Full Dataset');
  const isSampled = stats?.is_sampled ?? false;

  const pieData = [
    { name: 'Positive Sentiment (Target 1)', value: pos, color: '#22c55e' },
    { name: 'Negative Sentiment (Target 0)', value: neg, color: '#f43f5e' },
  ];

  const preprocessingSteps = [
    { step: 1, title: "Lowercasing", desc: "Converts all input characters to lowercase for uniform string comparison." },
    { step: 2, title: "URL Stripping", desc: "Strips http://, https://, and www. links via regex token filters." },
    { step: 3, title: "User Mention Cleaning", desc: "Filters out Twitter @username handles." },
    { step: 4, title: "Hashtag Normalization", desc: "Converts #topic tags to raw linguistic tokens." },
    { step: 5, title: "Punctuation & Special Chars", desc: "Strips non-alphabetic symbols while retaining semantic negation markers." },
    { step: 6, title: "Whitespace Compression", desc: "Collapses redundant tabs, returns, and spaces into single spaces." },
    { step: 7, title: "Lexical Tokenization", desc: "Splits raw tweet strings into discrete lexical token arrays." },
    { step: 8, title: "Selective Stopword Removal", desc: "Removes standard English stopwords while strictly retaining sentiment polarity negators ('not', 'never', 'no')." },
    { step: 9, title: "Morphological Lemmatization", desc: "Utilizes NLTK WordNetLemmatizer to reduce inflected words to root dictionary lemmas." },
    { step: 10, title: "TF-IDF Vectorization", desc: "Transforms lemmatized text into sparse n-gram (1-2) feature matrices (sublinear term frequency scaling)." }
  ];

  return (
    <div className="space-y-8 animate-fadeIn max-w-6xl mx-auto">
      
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold font-heading text-white tracking-tight flex items-center gap-3">
          <Database className="w-6 h-6 text-white" />
          Stanford Sentiment140 Corpus
        </h1>
        <p className="text-neutral-400 text-sm mt-1">
          Kaggle & Stanford Sentiment140 dataset containing 1,600,000 annotated social media posts for binary sentiment classification.
        </p>
      </div>

      {/* Dataset KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3.5">
        
        <div className="obsidian-card p-5 rounded-2xl border border-neutral-800 xl:col-span-2">
          <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider block mb-1">Total Corpus Records</span>
          <span className="text-2xl font-bold font-heading text-white">{rawTotal.toLocaleString()}</span>
          <span className="text-[11px] text-neutral-400 font-mono block mt-1">Raw Tweet Records (1.6M)</span>
        </div>

        <div className="obsidian-card p-5 rounded-2xl border border-neutral-800">
          <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider block mb-1">Active Split</span>
          <span className="text-2xl font-bold font-heading text-white">{total.toLocaleString()}</span>
          <span className={`text-[11px] font-mono block mt-1 font-semibold ${isSampled ? 'text-amber-400' : 'text-emerald-400'}`}>
            {isSampled ? 'Sampled Subset' : 'Full Dataset ✓'}
          </span>
        </div>

        <div className="obsidian-card p-5 rounded-2xl border border-neutral-800">
          <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider block mb-1">Positive Class</span>
          <span className="text-2xl font-bold font-heading text-emerald-400">{pos.toLocaleString()}</span>
          <span className="text-[11px] text-neutral-400 font-mono block mt-1">Target = 1</span>
        </div>

        <div className="obsidian-card p-5 rounded-2xl border border-neutral-800">
          <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider block mb-1">Negative Class</span>
          <span className="text-2xl font-bold font-heading text-rose-400">{neg.toLocaleString()}</span>
          <span className="text-[11px] text-neutral-400 font-mono block mt-1">Target = 0</span>
        </div>

        <div className="obsidian-card p-5 rounded-2xl border border-neutral-800">
          <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider block mb-1">Train Split (80%)</span>
          <span className="text-2xl font-bold font-heading text-white">{trainSize.toLocaleString()}</span>
          <span className="text-[11px] text-neutral-400 font-mono block mt-1">Stratified</span>
        </div>

        <div className="obsidian-card p-5 rounded-2xl border border-neutral-800">
          <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider block mb-1">Test Split (20%)</span>
          <span className="text-2xl font-bold font-heading text-white">{testSize.toLocaleString()}</span>
          <span className="text-[11px] text-neutral-400 font-mono block mt-1">Stratified</span>
        </div>

      </div>

      {/* Training Mode Banner */}
      <div className={`flex items-center gap-3 px-5 py-3 rounded-xl border text-xs font-mono ${
        isSampled 
          ? 'bg-neutral-950 border-amber-500/30 text-amber-300' 
          : 'bg-neutral-950 border-neutral-800 text-neutral-300'
      }`}>
        <span className="text-base">{isSampled ? '⚡' : '✓'}</span>
        <span>
          <strong className="text-white">Active Training Mode:</strong> {trainingMode}
          {isSampled && ` — Fast development mode. Full benchmark evaluation utilizes complete 1.6M dataset.`}
          {!isSampled && ` — Model trained on the complete Sentiment140 corpus.`}
        </span>
      </div>

      {/* Class Balance Donut Chart */}
      <div className="obsidian-card p-6 rounded-2xl border border-neutral-800">
        <h3 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
          <PieIcon className="w-4 h-4 text-neutral-300" />
          Balanced Sentiment Class Distribution
        </h3>
        <div className="h-64 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={85}
                paddingAngle={4}
                dataKey="value"
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ backgroundColor: '#09090b', borderColor: '#27272a', borderRadius: '10px', color: '#fff', fontSize: '12px' }}
                formatter={(val: any) => [Number(val).toLocaleString(), 'Records']}
              />
              <Legend wrapperStyle={{ fontSize: '11px', color: '#a1a1aa' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sample Records Table */}
      {stats?.samples && stats.samples.length > 0 && (
        <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-4">
          <h3 className="text-sm font-semibold text-white flex items-center gap-2">
            <FileText className="w-4 h-4 text-neutral-300" />
            Corpus Sample Records
          </h3>
          <div className="overflow-x-auto border border-neutral-800 rounded-xl">
            <table className="w-full text-left text-xs text-neutral-300">
              <thead className="bg-neutral-900 text-neutral-400 font-mono text-[11px] border-b border-neutral-800">
                <tr>
                  <th className="p-3">ID</th>
                  <th className="p-3">Author</th>
                  <th className="p-3">Raw Content</th>
                  <th className="p-3">Label</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/80 bg-black">
                {stats.samples.map((sample, idx) => (
                  <tr key={idx} className="hover:bg-neutral-900/40 transition-colors">
                    <td className="p-3 font-mono text-neutral-500">{sample.id}</td>
                    <td className="p-3 text-white font-mono">@{sample.user}</td>
                    <td className="p-3 text-neutral-300">{sample.text}</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded-full font-mono text-[11px] font-semibold border ${
                        sample.sentiment === 'Positive'
                          ? 'bg-emerald-950/40 text-emerald-400 border-emerald-500/30'
                          : 'bg-rose-950/40 text-rose-400 border-rose-500/30'
                      }`}>
                        {sample.sentiment} ({sample.target})
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Preprocessing Pipeline Steps */}
      <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-4">
        <h3 className="text-sm font-semibold text-white flex items-center gap-2">
          <Sliders className="w-4 h-4 text-neutral-300" />
          Text Preprocessing Protocol
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {preprocessingSteps.map((item) => (
            <div key={item.step} className="bg-neutral-950 p-4 rounded-xl border border-neutral-800/90 flex gap-3">
              <div className="w-6 h-6 rounded-lg bg-neutral-900 border border-neutral-700/80 text-white font-mono font-bold text-xs flex items-center justify-center shrink-0">
                {item.step}
              </div>
              <div>
                <h4 className="text-xs font-semibold text-white font-heading">{item.title}</h4>
                <p className="text-xs text-neutral-400 mt-0.5 leading-relaxed">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};
