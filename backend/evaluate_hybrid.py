import pandas as pd
import torch
import numpy as np
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Import your Hybrid architecture and MMR logic
from neuro_symbolic_fusion import HybridFusionClassifier, MalayalamFeatureExtractor
from summarize import extract_with_mmr

def extract_clean_sentences(numbered_text):
    """Strips the '[0]' tags so the algorithm can evaluate the raw text."""
    if not isinstance(numbered_text, str): return []
    lines = numbered_text.split('\n')
    sentences = []
    for line in lines:
        clean_text = re.sub(r'^\[\d+\]\s*', '', line).strip()
        if clean_text:
            sentences.append(clean_text)
    return sentences

def main():
    print("="*50)
    print("🔬 RUNNING FINAL NEURO-SYMBOLIC EVALUATION")
    print("="*50)

    # 1. Load the Platinum 700 Batch (The Answer Key)
    test_file = "data/platinum_batch_700_ANNOTATED.csv"
    try:
        df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Error: Could not find {test_file}. Check your data folder.")
        return

    # 2. Spin up the Hybrid Engines
    print("Loading LaBSE Semantic Engine...")
    labse = SentenceTransformer("sentence-transformers/LaBSE")
    feature_extractor = MalayalamFeatureExtractor()
    
    print("Loading Neuro-Symbolic Neural Network...")
    classifier = HybridFusionClassifier(labse_dim=768, symbolic_dim=4)
    # Loading the exact weights you just got from Godly!
    classifier.load_state_dict(torch.load("models/malayalam_hybrid_classifier.pt", map_location=torch.device('cpu'), weights_only=True))
    classifier.eval()

    # 3. Evaluation Metrics Trackers
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    print("Evaluating 700 unseen articles...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentences = extract_clean_sentences(row['Numbered_Source_Text'])
        true_labels_str = str(row['Your_Extractive_Labels'])
        
        # Skip broken rows
        if not sentences or true_labels_str == "nan" or not true_labels_str.strip():
            continue

        # Get the "Ground Truth" answers (e.g., set of {0, 2, 4})
        true_indices = set(map(int, [x.strip() for x in true_labels_str.split(",")]))
        k = len(true_indices) # How many sentences we should extract
        total_sentences = len(sentences)

        if total_sentences <= k:
            continue

        # --- AI INFERENCE ---
        # Path A: Semantics
        embeddings = labse.encode(sentences)
        X_tensor = torch.tensor(embeddings, dtype=torch.float32)

        # Path B: Linguistics
        symbolic_features = [feature_extractor.extract_features(s, i, total_sentences) for i, s in enumerate(sentences)]
        S_tensor = torch.tensor(symbolic_features, dtype=torch.float32)

        # Brain Scoring
        with torch.no_grad():
            probs = classifier(X_tensor, S_tensor).squeeze().numpy()
            if probs.ndim == 0:  # Handle single sentence edge-case
                probs = np.array([probs])

        # MMR Extraction
        predicted_list = extract_with_mmr(embeddings, probs, k=k, diversity=0.3)
        predicted_indices = set(predicted_list)

        # --- CALCULATE OVERLAPS ---
        true_positives += len(true_indices.intersection(predicted_indices))
        false_positives += len(predicted_indices - true_indices)
        false_negatives += len(true_indices - predicted_indices)

    # 4. The Final Math
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print("🏆 FINAL SUPERVISED METRICS (NEURO-SYMBOLIC AI)")
    print("="*50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()