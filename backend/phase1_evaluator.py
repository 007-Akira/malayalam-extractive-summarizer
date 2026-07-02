import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
    print("📉 RUNNING PHASE 1: UNSUPERVISED CENTROID BASELINE")
    print("="*50)

    test_file = "data/platinum_batch_700_ANNOTATED.csv"
    try:
        df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Error: Could not find {test_file}.")
        return

    print("Loading LaBSE (No Supervised Neural Network)...")
    labse = SentenceTransformer("sentence-transformers/LaBSE")

    true_positives, false_positives, false_negatives = 0, 0, 0

    print("Evaluating 700 articles using pure Cosine Similarity...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentences = extract_clean_sentences(row['Numbered_Source_Text'])
        true_labels_str = str(row['Your_Extractive_Labels'])
        
        if not sentences or true_labels_str == "nan" or not true_labels_str.strip(): continue

        true_indices = set(map(int, [x.strip() for x in true_labels_str.split(",")]))
        k = len(true_indices)
        if len(sentences) <= k: continue

        # --- PHASE 1 LOGIC: The Centroid Math ---
        embeddings = labse.encode(sentences)
        
        # Calculate the "Centroid" (average meaning of the whole article)
        centroid = np.mean(embeddings, axis=0).reshape(1, -1)
        
        # Calculate how close each sentence is to the centroid
        sims = cosine_similarity(embeddings, centroid).flatten()
        
        # Pick the top k sentences
        predicted_list = np.argsort(sims)[-k:]
        predicted_indices = set(predicted_list)

        true_positives += len(true_indices.intersection(predicted_indices))
        false_positives += len(predicted_indices - true_indices)
        false_negatives += len(true_indices - predicted_indices)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*50)
    print("📊 PHASE 1: UNSUPERVISED BASELINE METRICS")
    print("="*50)
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()