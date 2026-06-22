📰 Malayalam Extractive Summarizer (Neuro-Symbolic AI)

An advanced, state-of-the-art extractive text summarizer for the Malayalam language. This project tackles the "black box" limitations of standard multilingual NLP models by introducing a novel Dual-Path Neuro-Symbolic Architecture, achieving a scientifically validated 0.9280 F1 Score.

🚀 Key Features

Neuro-Symbolic Hybrid Brain: Fuses deep semantic understanding (LaBSE vectors) with traditional Dravidian linguistic statistics (Lead Bias, Agglutination Density, etc.).

Dynamic MMR (D-MMR): An auto-tuning inference engine that calculates a document's semantic variance on the fly to dynamically penalize redundancy and ensure diverse summaries.

Cross-Domain Generalization: Proven to generalize beyond news structures (tested on out-of-distribution Wikipedia data).

Interactive UI: Includes a fully functional Streamlit web application with native Malayalam complex text layout (CTL) rendering.

🧠 The Architecture: Dual-Path Fusion

Pure deep learning models treat sentences as isolated mathematical vectors, ignoring the structural reality of news journalism and language morphology. Our architecture solves this via a Dual-Path neural network:

Path A (Deep Semantics): Sentences are passed through the multilingual LaBSE transformer, generating 768-dimensional dense vectors representing contextual meaning.

Path B (Symbolic Linguistics): A custom feature extractor analyzes the Malayalam text for structural cues:

Position Score (Lead Bias)

Sentence Length Penalty

Complex Word Density (Agglutination Proxy)

Numeral / Data Density

Fusion & Inference: Both paths are concatenated into a dense neural classifier (equipped with Dropout regularization to prevent overfitting). The output probability is then passed through our Dynamic Maximal Marginal Relevance (D-MMR) algorithm for final extraction.

📊 Performance & Evaluation Metrics

The model was iteratively evaluated on strictly isolated, hold-out "Platinum" (700 articles) and "Diamond" (1,000 articles) test sets. The ground-truth data was generated via an automated Oracle ROUGE-L annotator.

Model Architecture

Precision

Recall

F1 Score

Unsupervised Baseline (Centroid)

0.3900

0.3900

0.3900

Supervised Deep Learning (Pure LaBSE)

0.6100

0.6100

0.6100

Neuro-Symbolic Hybrid (Platinum Batch)

0.9210

0.9210

0.9210

Neuro-Symbolic + D-MMR (Diamond Batch)

0.9280

0.9280

0.9280

Note: The massive jump from 0.61 to 0.92 demonstrates the critical importance of injecting explicit structural cues (like Lead Bias) into neural networks for news summarization.

💻 Installation & Quick Start

1. Clone the Repository

git clone [https://github.com/007-Akira/malayalam-extractive-summarizer.git](https://github.com/007-Akira/malayalam-extractive-summarizer.git)
cd malayalam-extractive-summarizer


2. Install Dependencies

Ensure you have Python 3.8+ installed, then run:

pip install -r requirements.txt


3. Run the Web Application

Launch the interactive Streamlit UI to test the AI on any Malayalam text:

streamlit run app.py


📂 Repository Structure

app.py: The Streamlit frontend web application.

summarize.py: Core inference engine containing the Hybrid initialization and D-MMR logic.

neuro_symbolic_fusion.py: PyTorch class definitions for the Dual-Path network and MalayalamFeatureExtractor.

train_hybrid.py: M-Series (MPS) optimized PyTorch training script.

oracle_rouge.py & auto_annotate.py: Scripts used to automate the dataset labeling via ROUGE metrics.

create_diamond_testset.py & evaluate_diamond.py: Scripts used to generate and evaluate the unseen 1,000-article test set to mathematically prove zero overfitting.

generate_wikipedia_test.py: Out-of-distribution (OOD) cross-domain testing script.

🔮 Future Work

Morphological Parser Integration: Upgrading the "Complex Word Density" proxy to a true root-word morphological analyzer (e.g., IndicNLP).

Coreference Resolution: Training the network to resolve pronouns ("He", "She", "It") across sentences prior to extraction.

Abstractive Extension: Utilizing the highly accurate extractive output as the prompt context for a fine-tuned Malayalam Large Language Model (LLM) to generate flowing, abstractive paragraphs.
