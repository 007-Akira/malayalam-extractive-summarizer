import pandas as pd
import torch
import numpy as np
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Assuming these are available in your environment
from neuro_symbolic_fusion import HybridFusionClassifier, MalayalamFeatureExtractor

def extract_clean_sentences(numbered_text):
    if not isinstance(numbered_text, str): return []
    return [re.sub(r'^\[\d+\]\s*', '', line).strip() for line in numbered_text.split('\n') if re.sub(r'^\[\d+\]\s*', '', line).strip()]

# ---------------------------------------------------------
# SILENT MMR LOGIC
# ---------------------------------------------------------
def compute_dynamic_diversity_silent(embeddings):
    if len(embeddings) <= 1:
        return 0.3
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, 0)
    avg_sim = sim_matrix.sum() / (len(embeddings) * (len(embeddings) - 1))
    return round(max(0.1, min(0.6, avg_sim)), 2)

def extract_with_mmr_silent(embeddings, probabilities, k=3):
    num_sentences = len(probabilities)
    if num_sentences <= k:
        return list(range(num_sentences))

    diversity = compute_dynamic_diversity_silent(embeddings)
    selected_indices = []
    unselected_indices = list(range(num_sentences))

    first_idx = int(np.argmax(probabilities))
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)

    while len(selected_indices) < k:
        mmr_scores = []
        for idx in unselected_indices:
            importance = probabilities[idx]
            sims = cosine_similarity(
                [embeddings[idx]], 
                [embeddings[s] for s in selected_indices]
            )[0]
            max_sim = np.max(sims)
            adjusted_score = importance - (diversity * max_sim)
            mmr_scores.append((adjusted_score, idx))
            
        mmr_scores.sort(reverse=True, key=lambda x: x[0])
        best_next_idx = mmr_scores[0][1]
        selected_indices.append(best_next_idx)
        unselected_indices.remove(best_next_idx)

    return selected_indices
# ---------------------------------------------------------

def main():
    print("="*50)
    print("🚀 EVALUATING CHOTTA BHEEM MODEL (Diamond Set)")
    print("="*50)

    try:
        df = pd.read_csv("data/diamond_batch_1000_ANNOTATED.csv")
    except FileNotFoundError:
        print("Error: data/diamond_batch_1000_ANNOTATED.csv not found!")
        return

    print("1. Loading Semantic Engine...")
    # Keeping the same embedder for consistency with training
    embedder = SentenceTransformer("l3cube-pune/indic-sentence-bert-nli") 
    
    print("2. Loading Linguistic Feature Extractor...")
    feature_extractor = MalayalamFeatureExtractor()
    
    print("3. Loading Chotta Bheem Model (chotta_bheem.pt)...")
    classifier = HybridFusionClassifier(labse_dim=768, symbolic_dim=4)
    # Loading the specific requested weights
    classifier.load_state_dict(torch.load("models/chotta_bheem.pt", map_location=torch.device('cpu'), weights_only=True))
    classifier.eval()

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    print("\nStarting Evaluation...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentences = extract_clean_sentences(row['Numbered_Source_Text'])
        true_labels_str = str(row['Your_Extractive_Labels'])
        
        if not sentences or true_labels_str == "nan" or not true_labels_str.strip():
            continue

        true_indices = set(map(int, [x.strip() for x in true_labels_str.split(",")]))
        k = len(true_indices)
        total_sentences = len(sentences)

        if total_sentences <= k: continue

        # --- PATH A: Semantics ---
        embeddings = embedder.encode(sentences)
        X_tensor = torch.tensor(embeddings, dtype=torch.float32)

        # --- PATH B: Linguistics ---
        symbolic_features = []
        for i, s in enumerate(sentences):
            feats = feature_extractor.extract_features(s, i, total_sentences)
            symbolic_features.append(feats)
            
        S_tensor = torch.tensor(symbolic_features, dtype=torch.float32)

        # --- FUSION & PREDICTION ---
        with torch.no_grad():
            probs = classifier(X_tensor, S_tensor).squeeze().numpy()
            if probs.ndim == 0: probs = np.array([probs])

        # --- DYNAMIC MMR EXTRACTION ---
        predicted_list = extract_with_mmr_silent(embeddings, probs, k=k)
        predicted_indices = set(predicted_list)

        true_positives += len(true_indices.intersection(predicted_indices))
        false_positives += len(predicted_indices - true_indices)
        false_negatives += len(true_indices - predicted_indices)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print("🏆 CHOTTA BHEEM PERFORMANCE METRICS")
    print("="*50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()