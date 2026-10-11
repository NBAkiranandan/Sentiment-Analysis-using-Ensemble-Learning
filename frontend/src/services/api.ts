import axios from 'axios';
import type {
  DatasetStats,
  ModelComparisonData,
  SinglePredictionResponse,
  ComparePredictionResponse,
  BatchPredictionResponse,
  TrainingStatus,
  MethodologyInfo
} from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000,
});

export const apiService = {
  async getHealth() {
    const res = await apiClient.get('/api/health');
    return res.data;
  },

  async getDatasetStats(): Promise<DatasetStats> {
    const res = await apiClient.get('/api/dataset/stats');
    return res.data;
  },

  async getModels() {
    const res = await apiClient.get('/api/models');
    return res.data;
  },

  async getMetrics(): Promise<any> {
    const res = await apiClient.get('/api/metrics');
    return res.data;
  },

  async getModelComparison(): Promise<ModelComparisonData> {
    const res = await apiClient.get('/api/model-comparison');
    return res.data;
  },

  async getMethodology(): Promise<MethodologyInfo> {
    const res = await apiClient.get('/api/methodology');
    return res.data;
  },

  async predictSingle(text: string, model?: string): Promise<SinglePredictionResponse> {
    const res = await apiClient.post('/api/predict', { text, model });
    return res.data;
  },

  async predictCompare(text: string): Promise<ComparePredictionResponse> {
    const res = await apiClient.post('/api/predict/compare', { text });
    return res.data;
  },

  async predictBatch(file: File, model?: string, textColumn?: string): Promise<BatchPredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);
    if (model) formData.append('model', model);
    if (textColumn) formData.append('text_column', textColumn);

    const res = await apiClient.post('/api/predict/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  async startTraining(sampleSize?: number): Promise<TrainingStatus> {
    const res = await apiClient.post('/api/train', { sample_size: sampleSize });
    return res.data;
  },

  async getTrainingStatus(): Promise<TrainingStatus> {
    const res = await apiClient.get('/api/training/status');
    return res.data;
  }
};
