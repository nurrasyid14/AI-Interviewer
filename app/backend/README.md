# 🧠 AI Interviewer — Backend Architecture

This backend module powers the **AI Interviewer**, a conversational system designed for automated evaluation of applicant responses using **Machine Learning**, **Text Mining**, and **I/O Psychology** metrics (based on Lawshe’s Reliability thresholds).

---

## ⚙️ Core Objective

To simulate an intelligent interviewer that:
- Receives multilingual interview responses (speech or text).
- Translates, cleans, and normalizes linguistic data.
- Evaluates semantic **relevance**, **sentiment**, and **reliability**.
- Aggregates decision metrics for final **acceptance classification**.

---

## 🧩 Backend Folder Overview

backend/
├── api_call/ # Handles API endpoints & communication layer
│
├── context_classifier/ # Classifies context/topic of responses
│
├── decisionmaker/ # Core logic for acceptance/rejection based on metrics
│
├── learning_evaluation/ # Evaluates model learning, metrics, and reliability
│
├── preprocessor/ # Complete text preprocessing and translation pipeline
│ ├── ingestion/ # Input handling (speech/text), multilingual translation
│ ├── text_cleaning/ # Normalization, stopword removal, tokenization
│ ├── linguistics/ # Lemmatization & stemming
│ ├── feature_extraction/# TF-IDF, BoW, and word embeddings
│ └── preprocessor_pipeline.py # Orchestrates preprocessing flow
│
├── relevance/ # Determines semantic & topical relevance of responses
│
├── reporter/ # Summarizes and reports evaluation results
│
├── sentiment/ # Sentiment and affective tone analysis
│
└── init.py

---

## 🔁 System Pipeline

[Input Speech/Text]
↓
🧩 Preprocessing
├─ Translation & Cleaning
├─ Tokenization & Lemmatization
└─ Feature Vectorization (TF-IDF / Embeddings)
↓
📊 Subsetting & Ensemble Learning
├─ Model Partitioning (K-Fold / Leave-One-Out)
├─ Ensemble Voting (RF / SVM / NB / DTree)
↓
🧠 Decision Making
├─ Reliability Scoring (0–1 scale)
├─ Acceptance if R >= 0.75 (Lawshe threshold)
↓
💬 Sentiment & Relevance Evaluation
├─ Sentiment tone consistency
└─ Contextual match with question intent
↓
📄 Reporting
├─ Generates reliability report
├─ Stores results under /data/reports
└─ Feeds data to frontend visualizations (Plotly)

---

## 🧱 Key Components

### 🔹 **1. Preprocessor**
Handles all text ingestion, normalization, and linguistic transformation:
- `translator.py` → Converts all responses to English for model consistency.
- `tokenizer.py`, `lemmatizer_stemmer.py` → Standardize linguistic inputs.
- `tf_idf.py`, `word_embedding.py` → Vectorize text for learning models.

### 🔹 **2. Relevance**
Computes semantic similarity between candidate answers and the intended question using:
- Cosine similarity or sentence embeddings.
- Threshold tuning to align with I/O Psychology interpretability.

### 🔹 **3. Sentiment**
Analyzes applicant emotional tone and assertiveness for personality estimation:
- Polarity & subjectivity scoring.
- Emotional stability as an interpretive feature in reliability metrics.

### 🔹 **4. Decision Maker**
Aggregates outputs of Ensemble Models + Relevance + Sentiment + Reliability to decide:
> **ACCEPT** if Reliability ≥ 0.75  
> **REJECT** otherwise.

### 🔹 **5. Reporter**
Compiles all logs and analytical summaries:
- Reliability score report
- Sentiment polarity chart (Plotly)
- Data export to `data/reports/`

## Folder Structure

app/
└── backend/
    ├── __init__.py
    │
    ├── api_call/
    │   ├── __init__.py
    │   ├── external_services.py          # Optional: translation or verification APIs
    │   ├── fastapi_router.py             # REST endpoints for frontend
    │   ├── websocket_handler.py          # Real-time feedback or dashboard updates
    │   └── schema_models.py              # Pydantic models for request/response data
    │
    ├── context_classifier/
    │   ├── __init__.py
    │   └── context_classifier.py         # Topic/context detection of interview Q&A
    │
    ├── preprocessor/
    │   ├── __init__.py
    │   ├── [ingestion]/
    │   │   ├── __init__.py
    │   │   ├── input_handler.py
    │   │   ├── audio_transcriber.py
    │   │   └── translator.py
    │   ├── [text_cleaning]/
    │   │   ├── __init__.py
    │   │   ├── normalizer.py
    │   │   ├── stopword_remover.py
    │   │   └── tokenizer.py
    │   ├── [linguistics]/
    │   │   ├── __init__.py
    │   │   ├── lemmatizer_stemmer.py
    │   │   └── pos_tagger.py
    │   ├── [feature_extraction]/
    │   │   ├── __init__.py
    │   │   ├── bag_of_words.py
    │   │   ├── tf_idf.py
    │   │   └── word_embedding.py
    │   └── preprocessor_pipeline.py
    │
    ├── subset/
    │   ├── __init__.py
    │   ├── k_fold.py
    │   ├── leave_one_out.py
    │   ├── cluster_subset.py
    │   └── subset_utils.py
    │
    ├── ensemble/
    │   ├── __init__.py
    │   ├── random_forest.py
    │   ├── neural_network.py
    │   ├── svm.py
    │   ├── naive_bayes.py
    │   └── ensemble_controller.py
    │
    ├── learning_evaluation/
    │   ├── __init__.py
    │   ├── metrics.py                    # Accuracy, F1, precision, recall
    │   ├── calibration.py                # Probability calibration & reliability
    │   ├── bias_analysis.py              # Fairness metrics
    │   └── evaluation_pipeline.py        # Interfaces with ensemble & visualizer
    │
    ├── relevance/
    │   ├── __init__.py
    │   ├── relevance_evaluator.py        # Cosine similarity / semantic match
    │   └── topic_alignment.py            # Theme matching for question-answer pairs
    │
    ├── sentiment/
    │   ├── __init__.py
    │   ├── sentiment_analyzer.py         # Polarity & affect detection
    │   └── tone_model.py                 # Prosody or voice tone model
    │
    ├── reliability/
    │   ├── __init__.py
    │   ├── lawshes_cvr.py                # Content Validity Ratio (I/O Psychology)
    │   ├── cronbach_alpha.py             # Internal consistency
    │   ├── reliability_metric.py         # Aggregation of reliability scores
    │   └── reliability_reporter.py       # Sends summarized reliability results
    │
    ├── decisionmaker/
    │   ├── __init__.py
    │   ├── decision_rules.py             # Accept/Reject logic
    │   ├── threshold_logic.py
    │   ├── explainability.py             # Justifications for decisions
    │   └── decision_logger.py
    │
    ├── reporter/
    │   ├── __init__.py
    │   ├── db_connector.py               # PostgreSQL interface for metadata/results
    │   ├── result_logger.py              # Saves run summaries
    │   ├── report_generator.py           # Converts results → structured report
    │   └── dashboard_updater.py          # Updates live user dashboards
    │
    ├── visualizer/
    │   ├── __init__.py
    │   ├── [evaluation/]                 # Dev-focused visuals (learning/evaluation)
    │   │   ├── __init__.py
    │   │   ├── model_performance.py
    │   │   ├── confusion_matrix_plotter.py
    │   │   └── reliability_distribution.py
    │   ├── [user_dashboard/]             # End-user dashboards
    │   │   ├── __init__.py
    │   │   ├── reliability_overview.py
    │   │   ├── sentiment_chart.py
    │   │   ├── relevance_heatmap.py
    │   │   └── overall_dashboard.py
    │   ├── [shared_components/]          # Shared styling/utilities
    │   │   ├── __init__.py
    │   │   ├── plotly_theme.py
    │   │   └── export_utils.py
    │   └── visualizer_pipeline.py        # Connects plots to frontend & reporter
    │
    ├── pipeline_controller.py            # Orchestrates full AI Interviewer pipeline
    │
    └── utils/
        ├── __init__.py
        ├── config_loader.py
        ├── language_tools.py
        ├── io_utils.py
        └── logger.py


---

## 📈 Learning Evaluation & Subsetting

- **Subsetting:** K-Fold, Leave-One-Out, or Stratified Sampling to ensure robust reliability testing.
- **Ensemble Learning:** Random Forest, SVM, Naïve Bayes, Decision Tree — trained via cross-validation.
- **Reliability Evaluation:** Cronbach’s Alpha or Lawshe CVR for I/O Psychology validity.

---

## 🧾 Reliability Rule (I/O Psychology)

| Symbol | Description | Threshold |
|:-------|:-------------|:-----------|
| `R` | Reliability rate | 0–1 |
| `R >= 0.75` | Accept candidate (valid performance) |
| `R < 0.75` | Reject candidate (insufficient reliability) |

---

## 🧰 Dependencies (Installed via `requirements.txt`)

### --- CORE ENVIRONMENT ---
python-dotenv==1.0.1          # For .env configuration
numpy==1.26.4                 # Base math operations
pandas==2.2.3                 # Data handling (reports, logs, etc.)
scikit-learn==1.5.2           # ML models, ensemble learning, subsetting (KFold, LOOCV)
scipy==1.14.1                 # Statistical / psychometric computation

### --- NLP / TEXT MINING ---
nltk==3.9.1                   # Tokenization, stemming, stopwords
spacy==3.7.5                  # Linguistic parsing, POS tagging, NER
gensim==4.3.3                 # Word2Vec, TF-IDF, topic models
sentence-transformers==3.2.1  # Semantic embeddings (cross-lingual)
langdetect==1.0.9             # Auto-detect language
deep-translator==1.11.4       # Translate user input to English if needed

### --- AI / ENSEMBLE / MODEL HANDLING ---
xgboost==2.1.3                # Gradient boosting ensemble
lightgbm==4.3.0               # Efficient ensemble variant
joblib==1.4.2                 # Model persistence & caching
imbalanced-learn==0.12.3      # Handle dataset imbalance (psychology corpora often skewed)

### --- RELIABILITY / PSYCHOMETRICS ---
pingouin==0.5.4               # Reliability metrics (Cronbach α, ICC, etc.)
statsmodels==0.14.4           # Advanced statistical analysis
factor-analyzer==0.5.1        # Factor analysis for I/O psychology dimensions

### --- DATABASE / STORAGE ---
psycopg2-binary==2.9.10       # PostgreSQL connector
SQLAlchemy==2.0.36            # ORM for flexible database schema mapping

### --- VISUALIZATION (Plotly only) ---
plotly==5.24.1                # Interactive dashboard & reliability visualization
dash==2.18.2                  # Plotly Dash web interface (optional)
kaleido==0.2.1                # Static image export for Plotly

### --- SYSTEM / UTILITIES ---
tqdm==4.67.1                  # Progress bar for batch jobs
rich==13.9.3                  # Console visualization (pretty logs)
loguru==0.7.3                 # Logging handler (backend/reporting)

---



🔮 Future Extensions
Add Voice Sentiment Recognition (acoustic + linguistic fusion)

Expand Multilingual Support (local cultural tones)

Integrate Adaptive Questioning (context-aware probing)

Include Explainable AI layer for model transparency

🧭 “The reliability of an interview is the reliability of its evaluator — hence we build one that never drifts.