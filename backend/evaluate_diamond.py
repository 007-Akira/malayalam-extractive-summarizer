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
    print("🔬 RUNNING NEURO-SYMBOLIC EVALUATION ON DIAMOND BATCH")
    print("="*50)

    # 1. Load the Diamond 1000 Batch
    test_file = "data/diamond_batch_1000_ANNOTATED.csv"
    try:
        df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Error: Could not find {test_file}. Run create_diamond_testset.py first.")
        return

    # 2. Spin up the Hybrid Engines
    print("Loading LaBSE Semantic Engine...")
    labse = SentenceTransformer("sentence-transformers/LaBSE")
    feature_extractor = MalayalamFeatureExtractor()
    
    print("Loading Neuro-Symbolic Neural Network...")
    classifier = HybridFusionClassifier(labse_dim=768, symbolic_dim=4)
    # Using map_location='cpu' for laptop compatibility
    classifier.load_state_dict(torch.load("models/malayalam_hybrid_classifier.pt", map_location=torch.device('cpu'), weights_only=True))
    classifier.eval()

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    print("Evaluating 1,000 completely new articles...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentences = extract_clean_sentences(row['Numbered_Source_Text'])
        true_labels_str = str(row['Your_Extractive_Labels'])
        
        if not sentences or true_labels_str == "nan" or not true_labels_str.strip():
            continue

        true_indices = set(map(int, [x.strip() for x in true_labels_str.split(",")]))
        k = len(true_indices)
        total_sentences = len(sentences)

        if total_sentences <= k:
            continue

        # Inference
        embeddings = labse.encode(sentences)
        X_tensor = torch.tensor(embeddings, dtype=torch.float32)
        symbolic_features = [feature_extractor.extract_features(s, i, total_sentences) for i, s in enumerate(sentences)]
        S_tensor = torch.tensor(symbolic_features, dtype=torch.float32)

        with torch.no_grad():
            probs = classifier(X_tensor, S_tensor).squeeze().numpy()
            if probs.ndim == 0: probs = np.array([probs])

        # We use the D-MMR logic we just built!
        predicted_list = extract_with_mmr(embeddings, probs, k=k, diversity="auto")
        predicted_indices = set(predicted_list)

        true_positives += len(true_indices.intersection(predicted_indices))
        false_positives += len(predicted_indices - true_indices)
        false_negatives += len(true_indices - predicted_indices)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print("💎 FINAL METRICS (1,000 ARTICLE DIAMOND BATCH)")
    print("="*50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()