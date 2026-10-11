export interface DatasetStats {
  records_raw?: number;
  total_samples: number;
  positive_samples: number;
  negative_samples: number;
  train_size: number;
  test_size: number;
  is_sampled: boolean;
  sample_size: number;
  sample_size_used?: number;
  training_mode?: string;
  dataset_source?: string;
  dataset_name?: string;
  samples?: SampleRecord[];
}

export interface SampleRecord {
  id: string;
  user: string;
  text: string;
  target: number;
  sentiment: string;
}

export interface ModelMetric {
  model_name: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: number[][]; // [[TN, FP], [FN, TP]]
  training_time: number;
  prediction_time: number;
  classification_report?: Record<string, any>;
}

export interface ModelComparisonData {
  dataset_stats: DatasetStats;
  models?: string[];
  logistic_regression: ModelMetric;
  random_forest: ModelMetric;
  adaboost: ModelMetric;
  selected_model: string;
  primary_metric: string;
}

export interface SinglePredictionRequest {
  text: string;
  model?: string;
}

export interface SinglePredictionResponse {
  text: string;
  processed_text: string;
  sentiment: 'Positive' | 'Negative';
  label: number;
  confidence: number;
  model: string;
  probabilities: {
    Positive: number;
    Negative: number;
  };
}

export interface ModelPredictionDetail {
  name: string;
  type: string;
  sentiment: 'Positive' | 'Negative';
  label: number;
  confidence: number;
  probabilities: {
    Positive: number;
    Negative: number;
  };
}

export interface ComparePredictionResponse {
  text: string;
  processed_text: string;
  selected_model: string;
  logistic_regression: ModelPredictionDetail;
  random_forest: ModelPredictionDetail;
  adaboost: ModelPredictionDetail;
}

export interface BatchPredictionItem {
  original_text: string;
  processed_text: string;
  predicted_sentiment: 'Positive' | 'Negative';
  label: number;
  confidence: number;
  model: string;
}

export interface BatchPredictionResponse {
  total_records: number;
  positive_count: number;
  negative_count: number;
  model_used: string;
  results: BatchPredictionItem[];
}

export interface TrainingStatus {
  status: 'idle' | 'training' | 'completed' | 'failed';
  step: string;
  progress: number;
  message: string;
  error?: string | null;
}

export interface MethodologyStep {
  step: number;
  name: string;
  description: string;
}

export interface MethodologyInfo {
  title: string;
  steps: MethodologyStep[];
}
