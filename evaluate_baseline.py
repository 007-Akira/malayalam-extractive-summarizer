import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def get_model_predictions(sentences, model, k):
    """Generates embeddings, calculates centroid, and extracts top K indices."""
    if len(sentences) <= k:
        return set(range(len(sentences)))
        
    embeddings = model.encode(sentences)
    centroid = np.mean(embeddings, axis=0)
    scores = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
    
    # argsort sorts ascending, so [-k:] gets the highest values
    top_k_indices = scores.argsort()[-k:]
    return set(top_k_indices)

def main():
    print("1. Loading Evaluation Data...")
    df = pd.read_csv("data/gold_dataset_unannotated.csv")
    
    # Filter for rows where you actually typed in labels
    df['Your_Extractive_Labels'] = df['Your_Extractive_Labels'].fillna('')
    annotated_df = df[df['Your_Extractive_Labels'].str.strip() != '']
    
    if len(annotated_df) == 0:
        print("Error: No labels found. Did you save your numbers in the CSV?")
        return
        
    print(f"   Found {len(annotated_df)} annotated articles.")
    
    print("\n2. Loading LaBSE Engine (from cache)...")
    model = SentenceTransformer("sentence-transformers/LaBSE")
    
    metrics = []
    
    print("\n3. Running Evaluation...")
    for idx, row in annotated_df.iterrows():
        # Parse your human labels safely
        raw_labels = str(row['Your_Extractive_Labels']).split(',')
        human_labels = set([int(x.strip()) for x in raw_labels if x.strip().isdigit()])
        k = len(human_labels)
        
        if k == 0: continue
        
        # Extract clean sentences directly from the numbered text to guarantee index matching
        raw_lines = str(row['Numbered_Source_Text']).split('\n')
        sentences = [re.sub(r'^\[\d+\]\s*', '', line).strip() for line in raw_lines if line.strip()]
        
        # Get what the model thinks are the top K
        model_labels = get_model_predictions(sentences, model, k)
        
        # Calculate overlap and metrics
        true_positives = len(human_labels.intersection(model_labels))
        
        precision = true_positives / len(model_labels) if len(model_labels) > 0 else 0
        recall = true_positives / len(human_labels) if len(human_labels) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics.append((precision, recall, f1))
        
        print(f"\n--- Article ID: {row['Article_ID']} ---")
        print(f"Human chose: {sorted(list(human_labels))}")
        print(f"Model chose: {sorted(list(model_labels))}")
        print(f"Overlap:     {true_positives} sentences")
        
    # Calculate Macro Averages
    avg_p = sum(m[0] for m in metrics) / len(metrics)
    avg_r = sum(m[1] for m in metrics) / len(metrics)
    avg_f1 = sum(m[2] for m in metrics) / len(metrics)
    
    print("\n==========================================")
    print("BASELINE EVALUATION METRICS (LaBSE Centroid)")
    print("==========================================")
    print(f"Precision : {avg_p:.2f}")
    print(f"Recall    : {avg_r:.2f}")
    print(f"F1 Score  : {avg_f1:.2f}")
    print("==========================================")

if __name__ == "__main__":
    main()