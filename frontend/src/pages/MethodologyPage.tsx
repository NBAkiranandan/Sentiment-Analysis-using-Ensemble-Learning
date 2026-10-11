import React from 'react';
import { 
  Workflow, 
  Database, 
  Scissors, 
  ListFilter, 
  Binary, 
  Layers, 
  GitCompare, 
  Send,
  Cpu
} from 'lucide-react';

export const MethodologyPage: React.FC = () => {
  const steps = [
    { title: "Corpus Ingestion", icon: Database, desc: "Ingestion of 1.6 Million Sentiment140 social media text records annotated as Positive (1) or Negative (0)." },
    { title: "Deterministic Cleaning", icon: Scissors, desc: "Normalized lowercasing, regex URL scrubbing, author handle filtering, and special punctuation normalization." },
    { title: "Negation-Preserving Tokenization", icon: ListFilter, desc: "Splitting into lexical tokens with custom stopword removal that explicitly protects negation markers ('not', 'no', 'never', 'bad', 'great')." },
    { title: "TF-IDF Feature Extraction & Anti-Data Leakage", icon: Binary, desc: "Generating high-dimensional unigram and bigram numerical vectors (sublinear term frequency scaling, up to 25,000 features). Vectorizer is fitted ONLY on training split (X_train)." },
    { title: "Model 1: Baseline Logistic Regression", icon: Cpu, desc: "Fitting a linear decision boundary classifier with L2 regularization for fast, interpretable baseline text classification." },
    { title: "Model 2: Bagging Random Forest", icon: Layers, desc: "Bootstrap Aggregation (100 decorrelated decision trees) built in parallel using random feature subsets to reduce variance." },
    { title: "Model 3: Boosting AdaBoost", icon: Layers, desc: "Sequential Adaptive Boosting fitting decision stumps iteratively to penalize misclassifications and lower bias." },
    { title: "Stratified Model Evaluation & Dynamic Selection", icon: GitCompare, desc: "Computing Accuracy, Precision, Recall, F1 Score, and Confusion Matrices on the identical 20% test split. Dynamic selection picks the top F1 Score model." },
    { title: "Low-Latency REST Inference API", icon: Send, desc: "Real-time single prediction, 3-model comparison, and vectorized CSV batch prediction endpoints via FastAPI." },
  ];

  return (
    <div className="space-y-8 animate-fadeIn max-w-5xl mx-auto">
      
      {/* Header */}
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold font-heading text-white tracking-tight flex items-center gap-3">
          <Workflow className="w-6 h-6 text-white" />
          Machine Learning Pipeline & Methodology
        </h1>
        <p className="text-neutral-400 text-sm mt-1">
          End-to-end technical pipeline comparing Baseline (Logistic Regression), Bagging (Random Forest), and Boosting (AdaBoost).
        </p>
      </div>

      {/* Interactive Visual Diagram Flow */}
      <div className="obsidian-card p-8 rounded-3xl border border-neutral-800 space-y-6">
        <h3 className="text-sm font-semibold text-white mb-6">Execution Pipeline Sequence</h3>

        <div className="relative border-l border-neutral-800 ml-4 space-y-7 pl-6">
          {steps.map((item, index) => {
            const Icon = item.icon;
            return (
              <div key={index} className="relative group">
                
                {/* Node Bullet */}
                <div className="absolute -left-[37px] top-0 w-8 h-8 rounded-xl bg-neutral-900 border border-neutral-700/80 flex items-center justify-center text-white shadow-md transition-transform group-hover:scale-105">
                  <Icon className="w-3.5 h-3.5" />
                </div>

                {/* Content Box */}
                <div className="bg-neutral-950 p-5 rounded-2xl border border-neutral-800/80 hover:border-neutral-700 transition-all">
                  <span className="text-[10px] font-mono font-bold tracking-wider text-neutral-400 uppercase">
                    Stage 0{index + 1}
                  </span>
                  <h4 className="text-sm font-bold font-heading text-white mt-0.5">{item.title}</h4>
                  <p className="text-xs text-neutral-400 mt-1 leading-relaxed">{item.desc}</p>
                </div>

              </div>
            );
          })}
        </div>
      </div>

      {/* 3 Model Architectural Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        
        {/* Baseline Card */}
        <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-3">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-950 border border-blue-800 text-blue-300 flex items-center justify-center font-mono font-bold text-xs">
              LR
            </div>
            <div>
              <h3 className="text-base font-bold font-heading text-white">Baseline: Logistic Reg</h3>
              <span className="text-[11px] font-mono text-neutral-400">Linear Boundary Model</span>
            </div>
          </div>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Logistic Regression models class log-odds as a linear combination of TF-IDF feature weights. Serves as a fast, highly scalable traditional Machine Learning baseline for binary text classification.
          </p>
        </div>

        {/* Bagging Card */}
        <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-3">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-purple-950 border border-purple-800 text-purple-300 flex items-center justify-center font-mono font-bold text-xs">
              RF
            </div>
            <div>
              <h3 className="text-base font-bold font-heading text-white">Bagging: Random Forest</h3>
              <span className="text-[11px] font-mono text-neutral-400">Variance Reduction</span>
            </div>
          </div>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Bootstrap Aggregation (Bagging) constructs an ensemble of decorrelated decision trees in parallel across random bootstrap samples. Random feature subsets per node split reduce variance and overfitting.
          </p>
        </div>

        {/* Boosting Card */}
        <div className="obsidian-card p-6 rounded-2xl border border-neutral-800 space-y-3">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-amber-950 border border-amber-800 text-amber-300 flex items-center justify-center font-mono font-bold text-xs">
              ADA
            </div>
            <div>
              <h3 className="text-base font-bold font-heading text-white">Boosting: AdaBoost</h3>
              <span className="text-[11px] font-mono text-neutral-400">Bias Reduction</span>
            </div>
          </div>
          <p className="text-xs text-neutral-400 leading-relaxed">
            Adaptive Boosting (AdaBoost) fits decision tree stumps sequentially. Iterative sample reweighting heavily penalizes misclassified instances, systematically reducing bias to build a strong classifier.
          </p>
        </div>

      </div>

    </div>
  );
};
