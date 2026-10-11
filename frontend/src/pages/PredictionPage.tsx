import React, { useState } from 'react';
import { 
  Send, 
  Terminal, 
  ThumbsUp, 
  ThumbsDown, 
  Cpu, 
  AlignLeft, 
  BarChart2, 
  AlertCircle,
  HelpCircle,
  GitCompare
} from 'lucide-react';
import { apiService } from '../services/api';
import type { SinglePredictionResponse, ComparePredictionResponse } from '../types';

export const PredictionPage: React.FC = () => {
  const [text, setText] = useState('');
  const [selectedModel, setSelectedModel] = useState<string>('compare_all');
  const [loading, setLoading] = useState(false);
  const [singleResult, setSingleResult] = useState<SinglePredictionResponse | null>(null);
  const [compareResult, setCompareResult] = useState<ComparePredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const sampleInputs = [
    { label: "Positive Movie Review", text: "I absolutely loved this movie! The cinematography and acting were brilliant." },
    { label: "Negative Product Review", text: "Worst customer support ever. The product broke down after just two days!" },
    { label: "Positive Tech Tweet", text: "Super excited about the new software release! Everything runs so fast and smoothly." },
    { label: "Negative Service Tweet", text: "Extremely frustrated with the constant delays and terrible communication." }
  ];

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please enter a sentence, tweet, or text review to analyze.");
      return;
    }
    setError(null);
    setLoading(true);
    setSingleResult(null);
    setCompareResult(null);

    try {
      if (selectedModel === 'compare_all') {
        const res = await apiService.predictCompare(text);
        setCompareResult(res);
      } else {
        const res = await apiService.predictSingle(text, selectedModel);
        setSingleResult(res);
      }
    } catch (err: any) {
      const msg = err.response?.data?.detail || "Prediction failed. Ensure models are trained.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 animate-fadeIn max-w-6xl mx-auto">
      
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold font-heading text-white tracking-tight flex items-center gap-3">
          <Terminal className="w-6 h-6 text-white" />
          Real-Time Sentiment Predictor & Model Evaluator
        </h1>
        <p className="text-neutral-400 text-sm mt-1">
          Evaluate text input using individual models or compare predictions side-by-side across Logistic Regression (Baseline), Random Forest (Bagging), and AdaBoost (Boosting).
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Input Form & Sample Presets */}
        <div className="lg:col-span-6 space-y-5">
          
          <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-4">
            
            {/* Model Selector */}
            <div className="flex items-center justify-between">
              <label className="text-xs font-mono text-neutral-400 uppercase tracking-wider flex items-center gap-2">
                <Cpu className="w-3.5 h-3.5 text-neutral-400" />
                Prediction Model Mode
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="bg-neutral-900 text-white border border-neutral-700 rounded-lg px-3 py-1.5 text-xs font-medium focus:outline-none focus:border-white transition-colors"
              >
                <option value="compare_all">✨ Compare All 3 Models Side-by-Side</option>
                <option value="Logistic Regression">Logistic Regression (Baseline)</option>
                <option value="Random Forest">Random Forest (Bagging)</option>
                <option value="AdaBoost">AdaBoost (Boosting)</option>
              </select>
            </div>

            {/* Textarea */}
            <div>
              <textarea
                rows={5}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Enter tweet, review, or sentence (e.g., 'I absolutely love this product!')..."
                className="w-full bg-neutral-950 text-white placeholder-neutral-500 rounded-xl p-4 text-sm border border-neutral-800 focus:outline-none focus:border-neutral-500 focus:ring-1 focus:ring-neutral-500 transition-all resize-none font-sans"
              />
              <div className="flex items-center justify-between mt-2 text-xs text-neutral-400 font-mono">
                <span>{text.length} characters</span>
                <button
                  onClick={() => { setText(''); setSingleResult(null); setCompareResult(null); setError(null); }}
                  className="hover:text-white transition-colors"
                >
                  Clear text
                </button>
              </div>
            </div>

            {/* Error banner */}
            {error && (
              <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-950/40 border border-rose-800/80 text-rose-300 text-xs font-mono">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{error}</span>
              </div>
            )}

            {/* Submit Button */}
            <button
              onClick={handleAnalyze}
              disabled={loading || !text.trim()}
              className="w-full py-3.5 bg-white hover:bg-neutral-200 disabled:opacity-40 text-black font-semibold rounded-xl text-xs transition-all shadow-md active:scale-95 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-black border-t-transparent rounded-full animate-spin"></div>
                  <span>Extracting Features & Classifying...</span>
                </>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  <span>{selectedModel === 'compare_all' ? 'Evaluate Across All 3 Models' : 'Analyze Sentiment'}</span>
                </>
              )}
            </button>
          </div>

          {/* Sample Prompts */}
          <div className="obsidian-card p-5 rounded-2xl border border-neutral-800">
            <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider block mb-3">
              Standard Test Benchmarks:
            </span>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {sampleInputs.map((sample, idx) => (
                <button
                  key={idx}
                  onClick={() => { setText(sample.text); setSingleResult(null); setCompareResult(null); }}
                  className="text-left p-3 rounded-xl bg-neutral-950 hover:bg-neutral-900 border border-neutral-800/80 hover:border-neutral-700 transition-all text-xs group"
                >
                  <span className="font-semibold text-white block mb-1 group-hover:text-neutral-200 font-sans">
                    {sample.label}
                  </span>
                  <p className="text-neutral-400 line-clamp-2 leading-relaxed">{sample.text}</p>
                </button>
              ))}
            </div>
          </div>

        </div>

        {/* Right Column: Results Display */}
        <div className="lg:col-span-6">
          
          {/* Comparison View across 3 Models */}
          {compareResult ? (
            <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-5 animate-fadeIn">
              
              <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
                <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider flex items-center gap-1.5">
                  <GitCompare className="w-4 h-4 text-emerald-400" />
                  3-Model Evaluation Comparison
                </span>
                <span className="text-[11px] font-mono text-neutral-400 bg-neutral-900 px-2.5 py-1 rounded border border-neutral-800">
                  Selected Model: {compareResult.selected_model}
                </span>
              </div>

              {/* Cleaned Tokenized Input */}
              <div className="bg-neutral-950 p-3.5 rounded-xl border border-neutral-800">
                <span className="text-[11px] font-mono text-neutral-400 block mb-1">Processed TF-IDF Tokens:</span>
                <p className="text-xs font-mono text-neutral-200 bg-black p-2 rounded border border-neutral-800 break-words">
                  {compareResult.processed_text || "(No tokens remaining)"}
                </p>
              </div>

              {/* Side-by-side Cards for 3 Models */}
              <div className="space-y-3">
                {[
                  { key: 'logistic_regression', item: compareResult.logistic_regression, badgeColor: 'bg-blue-950 border-blue-800 text-blue-300' },
                  { key: 'random_forest', item: compareResult.random_forest, badgeColor: 'bg-purple-950 border-purple-800 text-purple-300' },
                  { key: 'adaboost', item: compareResult.adaboost, badgeColor: 'bg-amber-950 border-amber-800 text-amber-300' }
                ].map(({ key, item, badgeColor }) => (
                  <div key={key} className="bg-neutral-950 p-4 rounded-xl border border-neutral-800 flex items-center justify-between gap-4">
                    <div>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="font-bold text-sm text-white">{item.name}</span>
                        <span className={`text-[10px] font-mono px-2 py-0.5 rounded border ${badgeColor}`}>{item.type}</span>
                      </div>
                      <div className="text-xs text-neutral-400 font-mono">
                        Model Confidence: <span className="text-white font-bold">{(item.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>

                    <div className="text-right">
                      <span className={`inline-block px-3 py-1 rounded-lg text-xs font-bold font-mono border ${
                        item.sentiment === 'Positive'
                          ? 'bg-emerald-950/80 border-emerald-600 text-emerald-400'
                          : 'bg-rose-950/80 border-rose-600 text-rose-400'
                      }`}>
                        {item.sentiment}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

            </div>
          ) : singleResult ? (
            /* Single Model Result View */
            <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-5 animate-fadeIn">
              
              {/* Main Sentiment Badge */}
              <div className="text-center py-4 border-b border-neutral-800">
                <div className={`w-14 h-14 rounded-2xl mx-auto flex items-center justify-center mb-3 shadow-md border ${
                  singleResult.sentiment === 'Positive'
                    ? 'bg-emerald-950/40 text-emerald-400 border-emerald-500/30'
                    : 'bg-rose-950/40 text-rose-400 border-rose-500/30'
                }`}>
                  {singleResult.sentiment === 'Positive' ? (
                    <ThumbsUp className="w-6 h-6" />
                  ) : (
                    <ThumbsDown className="w-6 h-6" />
                  )}
                </div>
                
                <span className="text-xs font-mono text-neutral-400 uppercase tracking-wider">Classification Result</span>
                <h2 className={`text-3xl font-extrabold font-heading mt-1 ${
                  singleResult.sentiment === 'Positive' ? 'text-emerald-400' : 'text-rose-400'
                }`}>
                  {singleResult.sentiment}
                </h2>
              </div>

              {/* Model Confidence Progress Bar */}
              <div className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-neutral-400 font-mono">Model Confidence</span>
                  <span className="font-bold font-mono text-white">{(singleResult.confidence * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-neutral-950 h-2.5 rounded-full overflow-hidden p-0.5 border border-neutral-800">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      singleResult.sentiment === 'Positive' ? 'bg-emerald-500' : 'bg-rose-500'
                    }`}
                    style={{ width: `${(singleResult.confidence * 100).toFixed(1)}%` }}
                  />
                </div>
              </div>

              {/* Probability Breakdown */}
              <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800/90 space-y-3">
                <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                  <BarChart2 className="w-3.5 h-3.5 text-neutral-400" />
                  Posterior Probability Distribution
                </span>
                
                <div className="space-y-2 text-xs font-mono">
                  <div className="flex items-center justify-between">
                    <span className="text-emerald-400">Positive Class</span>
                    <span className="text-white font-bold">{(singleResult.probabilities.Positive * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-rose-400">Negative Class</span>
                    <span className="text-white font-bold">{(singleResult.probabilities.Negative * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>

              {/* Preprocessed Text Tokenization Details */}
              <div className="bg-neutral-950 p-4 rounded-xl border border-neutral-800/90 space-y-2">
                <span className="text-xs font-semibold text-white flex items-center gap-1.5">
                  <AlignLeft className="w-3.5 h-3.5 text-neutral-400" />
                  Cleaned TF-IDF Input Tokens
                </span>
                <p className="text-xs text-neutral-300 font-mono bg-black p-2.5 rounded-lg border border-neutral-800/80 break-words leading-relaxed">
                  {singleResult.processed_text || "(No tokens remaining after filtering)"}
                </p>
              </div>

              {/* Model Info */}
              <div className="flex items-center justify-between text-xs text-neutral-400 pt-2 border-t border-neutral-800/80">
                <span>Model Architecture Used:</span>
                <span className="font-mono text-neutral-200 bg-neutral-900 px-2 py-0.5 rounded border border-neutral-800">
                  {singleResult.model}
                </span>
              </div>

            </div>
          ) : (
            /* Empty State */
            <div className="obsidian-card p-8 rounded-2xl border border-neutral-800 text-center h-full flex flex-col items-center justify-center min-h-[380px]">
              <div className="w-12 h-12 bg-neutral-950 rounded-xl flex items-center justify-center mb-3 text-neutral-500 border border-neutral-800">
                <HelpCircle className="w-6 h-6" />
              </div>
              <h3 className="text-sm font-semibold font-heading text-white mb-1">Awaiting Text Input</h3>
              <p className="text-xs text-neutral-400 max-w-xs leading-relaxed">
                Type text or select a preset benchmark on the left to evaluate sentiment across Logistic Regression, Random Forest, and AdaBoost.
              </p>
            </div>
          )}

        </div>

      </div>

    </div>
  );
};
