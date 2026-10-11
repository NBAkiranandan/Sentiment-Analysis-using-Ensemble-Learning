import React, { useState } from 'react';
import { 
  FileSpreadsheet, 
  Upload, 
  Download, 
  Play, 
  AlertCircle, 
  ThumbsUp, 
  ThumbsDown,
  RefreshCw,
  Search
} from 'lucide-react';
import { apiService } from '../services/api';
import type { BatchPredictionResponse } from '../types';

export const BatchPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('Random Forest');
  const [textColumn, setTextColumn] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<BatchPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      if (!selected.name.toLowerCase().endsWith('.csv')) {
        setError("Only .csv files are supported.");
        setFile(null);
        return;
      }
      setFile(selected);
      setError(null);
      setResponse(null);
    }
  };

  const handleUploadAndRun = async () => {
    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const res = await apiService.predictBatch(file, selectedModel, textColumn || undefined);
      setResponse(res);
    } catch (err: any) {
      const msg = err.response?.data?.detail || "Batch prediction failed.";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadCSV = () => {
    if (!response || !response.results.length) return;
    
    const headers = ["Original Text", "Predicted Sentiment", "Confidence", "Model"];
    const csvRows = [headers.join(',')];

    response.results.forEach(item => {
      const cleanText = `"${item.original_text.replace(/"/g, '""')}"`;
      csvRows.push([cleanText, item.predicted_sentiment, item.confidence, item.model].join(','));
    });

    const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `sentiment_predictions_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const filteredResults = response?.results.filter(r =>
    r.original_text.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  return (
    <div className="space-y-8 animate-fadeIn max-w-6xl mx-auto">
      
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold font-heading text-white tracking-tight flex items-center gap-3">
          <FileSpreadsheet className="w-6 h-6 text-white" />
          Batch CSV Sentiment Classifier
        </h1>
        <p className="text-neutral-400 text-sm mt-1">
          Upload any CSV file to execute high-throughput batch classification across thousands of text records simultaneously.
        </p>
      </div>

      {/* Upload Controls Box */}
      <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-6">
        
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-center">
          
          {/* Drag & Drop File Zone */}
          <div className="md:col-span-6">
            <label className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-neutral-700 hover:border-neutral-400 rounded-2xl cursor-pointer bg-neutral-950 hover:bg-neutral-900 transition-all text-center">
              <Upload className="w-7 h-7 text-neutral-300 mb-2" />
              <span className="text-xs font-semibold text-white">
                {file ? file.name : "Select or drag a CSV file"}
              </span>
              <span className="text-[11px] text-neutral-500 font-mono mt-1">
                {file ? `${(file.size / 1024).toFixed(1)} KB` : "Supports standard CSV datasets"}
              </span>
              <input type="file" accept=".csv" onChange={handleFileChange} className="hidden" />
            </label>
          </div>

          {/* Options */}
          <div className="md:col-span-6 space-y-3.5">
            
            {/* Model Selector */}
            <div>
              <label className="text-xs font-mono text-neutral-400 uppercase tracking-wider block mb-1">
                Classifier Architecture
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-neutral-950 text-white border border-neutral-700 rounded-xl px-3.5 py-2 text-xs font-medium focus:outline-none focus:border-white transition-colors"
              >
                <option value="Logistic Regression">Logistic Regression (Baseline)</option>
                <option value="Random Forest">Random Forest (Bagging)</option>
                <option value="AdaBoost">AdaBoost (Boosting)</option>
              </select>
            </div>

            {/* Custom Text Column Name */}
            <div>
              <label className="text-xs font-mono text-neutral-400 uppercase tracking-wider block mb-1">
                Text Column Name (Optional auto-detect)
              </label>
              <input
                type="text"
                placeholder="e.g., text, tweet, review..."
                value={textColumn}
                onChange={(e) => setTextColumn(e.target.value)}
                className="w-full bg-neutral-950 text-white placeholder-neutral-500 border border-neutral-700 rounded-xl px-3.5 py-2 text-xs focus:outline-none focus:border-white transition-colors"
              />
            </div>

            {/* Submit Button */}
            <button
              onClick={handleUploadAndRun}
              disabled={loading || !file}
              className="w-full py-3 bg-white hover:bg-neutral-200 disabled:opacity-40 text-black font-semibold rounded-xl text-xs transition-all shadow-md active:scale-95 flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin text-black" />
                  <span>Processing Batch Predictions...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Run Batch Predictions</span>
                </>
              )}
            </button>

          </div>

        </div>

        {error && (
          <div className="flex items-center gap-2 p-3 rounded-xl bg-rose-950/40 border border-rose-800/80 text-rose-300 text-xs font-mono">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

      </div>

      {/* Results Section */}
      {response && (
        <div className="space-y-6 animate-fadeIn">
          
          {/* Summary Stats Header */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="obsidian-card p-4 rounded-xl border border-neutral-800">
              <span className="text-xs text-neutral-400 font-mono block mb-1">Total Processed</span>
              <span className="text-2xl font-bold font-heading text-white">{response.total_records.toLocaleString()}</span>
            </div>
            <div className="obsidian-card p-4 rounded-xl border border-neutral-800">
              <span className="text-xs text-neutral-400 font-mono block mb-1">Positive Predictions</span>
              <span className="text-2xl font-bold font-heading text-emerald-400">{response.positive_count.toLocaleString()}</span>
            </div>
            <div className="obsidian-card p-4 rounded-xl border border-neutral-800">
              <span className="text-xs text-neutral-400 font-mono block mb-1">Negative Predictions</span>
              <span className="text-2xl font-bold font-heading text-rose-400">{response.negative_count.toLocaleString()}</span>
            </div>
          </div>

          {/* Table Header & Download Action */}
          <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-4">
            
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              
              {/* Search Box */}
              <div className="relative flex-1 max-w-md">
                <Search className="w-3.5 h-3.5 absolute left-3.5 top-3 text-neutral-500" />
                <input
                  type="text"
                  placeholder="Filter rows by text..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full bg-neutral-950 text-white placeholder-neutral-500 text-xs pl-9 pr-4 py-2.5 rounded-xl border border-neutral-800 focus:outline-none focus:border-white transition-colors"
                />
              </div>

              {/* Export Button */}
              <button
                onClick={handleDownloadCSV}
                className="flex items-center gap-2 px-4 py-2 bg-neutral-900 hover:bg-neutral-800 text-white font-medium rounded-xl text-xs transition-all border border-neutral-700/80 shrink-0"
              >
                <Download className="w-3.5 h-3.5" />
                Export Predicted CSV
              </button>

            </div>

            {/* Results Table */}
            <div className="overflow-x-auto border border-neutral-800 rounded-xl">
              <table className="w-full text-left text-xs text-neutral-300">
                <thead className="bg-neutral-900 text-neutral-400 font-mono text-[11px] border-b border-neutral-800">
                  <tr>
                    <th className="p-3">#</th>
                    <th className="p-3">Original Text</th>
                    <th className="p-3">Predicted Sentiment</th>
                    <th className="p-3">Confidence</th>
                    <th className="p-3">Model</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800/80 bg-black">
                  {filteredResults.slice(0, 100).map((row, idx) => (
                    <tr key={idx} className="hover:bg-neutral-900/40 transition-colors">
                      <td className="p-3 text-neutral-500 font-mono">{idx + 1}</td>
                      <td className="p-3 max-w-md truncate text-neutral-200" title={row.original_text}>
                        {row.original_text}
                      </td>
                      <td className="p-3">
                        <span className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full font-mono text-[11px] font-semibold border ${
                          row.predicted_sentiment === 'Positive'
                            ? 'bg-emerald-950/40 text-emerald-400 border-emerald-500/30'
                            : 'bg-rose-950/40 text-rose-400 border-rose-500/30'
                        }`}>
                          {row.predicted_sentiment === 'Positive' ? <ThumbsUp className="w-2.5 h-2.5" /> : <ThumbsDown className="w-2.5 h-2.5" />}
                          {row.predicted_sentiment}
                        </span>
                      </td>
                      <td className="p-3 font-bold font-mono text-white">
                        {(row.confidence * 100).toFixed(1)}%
                      </td>
                      <td className="p-3 text-neutral-400 font-mono text-[11px]">
                        {row.model}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {filteredResults.length > 100 && (
                <div className="p-3 text-center text-xs text-neutral-500 border-t border-neutral-800 font-mono">
                  Displaying top 100 of {filteredResults.length} records. Download CSV for complete dataset.
                </div>
              )}
            </div>

          </div>

        </div>
      )}

    </div>
  );
};
