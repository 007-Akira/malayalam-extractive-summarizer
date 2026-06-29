import pandas as pd
import torch
import numpy as np
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from neuro_symbolic_fusion import HybridFusionClassifier

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
    print("🧠 RUNNING PHASE 2: PURE DEEP LEARNING (NO LINGUISTICS)")
    print("="*50)

    test_file = "data/platinum_batch_700_ANNOTATED.csv"
    try:
        df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Error: Could not find {test_file}.")
        return

    print("Loading AI Engines...")
    labse = SentenceTransformer("sentence-transformers/LaBSE")
    
    classifier = HybridFusionClassifier(labse_dim=768, symbolic_dim=4)
    classifier.load_state_dict(torch.load("models/malayalam_hybrid_classifier.pt", map_location=torch.device('cpu'), weights_only=True))
    classifier.eval()

    true_positives, false_positives, false_negatives = 0, 0, 0

    print("Evaluating 700 articles (Semantics Only, No MMR)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentences = extract_clean_sentences(row['Numbered_Source_Text'])
        true_labels_str = str(row['Your_Extractive_Labels'])
        
        if not sentences or true_labels_str == "nan" or not true_labels_str.strip(): continue

        true_indices = set(map(int, [x.strip() for x in true_labels_str.split(",")]))
        k = len(true_indices)
        if len(sentences) <= k: continue

        # --- PHASE 2 LOGIC: Deep Semantics Only ---
        embeddings = labse.encode(sentences)
        X_tensor = torch.tensor(embeddings, dtype=torch.float32)

        # ISOLATION: We completely zero out Path B (Linguistics)
        # The AI is completely blind to Lead Bias, Length, and Agglutination
        S_tensor = torch.zeros((len(sentences), 4), dtype=torch.float32)

        with torch.no_grad():
            probs = classifier(X_tensor, S_tensor).squeeze().numpy()
            if probs.ndim == 0: probs = np.array([probs])

        # We also don't use MMR here, because Phase 2 didn't have it!
        predicted_list = np.argsort(probs)[-k:]
        predicted_indices = set(predicted_list)

        true_positives += len(true_indices.intersection(predicted_indices))
        false_positives += len(predicted_indices - true_indices)
        false_negatives += len(true_indices - predicted_indices)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print("📈 PHASE 2: PURE DEEP LEARNING METRICS")
    print("="*50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()