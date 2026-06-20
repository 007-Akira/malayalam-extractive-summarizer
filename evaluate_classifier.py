import pandas as pd
import torch
import torch.nn as nn
import re
from sentence_transformers import SentenceTransformer

# 1. Rebuild the Model Architecture exactly as it was during training
class SentenceClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super(SentenceClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)

def main():
    print("1. Loading Evaluation Data...")
    df = pd.read_csv("data/gold_dataset_unannotated.csv")
    df['Your_Extractive_Labels'] = df['Your_Extractive_Labels'].fillna('')
    annotated_df = df[df['Your_Extractive_Labels'].str.strip() != '']
    
    if len(annotated_df) == 0:
        print("Error: No labels found in the dataset.")
        return
        
    print(f"   Found {len(annotated_df)} annotated articles.")
    
    print("\n2. Loading LaBSE Engine to extract features...")
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    
    print("3. Loading Custom Trained Brain...")
    classifier = SentenceClassifier(input_dim=768)
    # Load the weights you trained on the M5
    classifier.load_state_dict(torch.load("models/malayalam_sentence_classifier.pt", weights_only=True))
    classifier.eval() # Lock the model into evaluation/inference mode
    
    metrics = []
    
    print("\n4. Running the Final Showdown...")
    with torch.no_grad(): # Turn off gradient tracking since we aren't training anymore
        for idx, row in annotated_df.iterrows():
            # Parse human labels
            raw_labels = str(row['Your_Extractive_Labels']).split(',')
            human_labels = set([int(x.strip()) for x in raw_labels if x.strip().isdigit()])
            k = len(human_labels)
            
            if k == 0: continue
            
            # Extract clean sentences
            raw_lines = str(row['Numbered_Source_Text']).split('\n')
            sentences = [re.sub(r'^\[\d+\]\s*', '', line).strip() for line in raw_lines if line.strip()]
            
            if len(sentences) <= k:
                model_labels = set(range(len(sentences)))
            else:
                # Convert sentences to math vectors using LaBSE
                embeddings = labse_model.encode(sentences)
                X_tensor = torch.tensor(embeddings, dtype=torch.float32)
                
                # Ask your custom PyTorch model to score them
                probabilities = classifier(X_tensor).squeeze()
                
                if probabilities.dim() == 0: # Handle single-sentence edge case
                    probabilities = probabilities.unsqueeze(0)
                    
                # Pick the top K sentences based on the Neural Network's confidence score
                top_k_indices = torch.argsort(probabilities, descending=True)[:k].tolist()
                model_labels = set(top_k_indices)
            
            # Calculate metrics
            true_positives = len(human_labels.intersection(model_labels))
            precision = true_positives / len(model_labels) if len(model_labels) > 0 else 0
            recall = true_positives / len(human_labels) if len(human_labels) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics.append((precision, recall, f1))
            
            print(f"\n--- Article ID: {row['Article_ID']} ---")
            print(f"Human chose: {sorted(list(human_labels))}")
            print(f"Model chose: {sorted(list(model_labels))}")
            print(f"Overlap:     {true_positives} sentences")
            
    avg_p = sum(m[0] for m in metrics) / len(metrics)
    avg_r = sum(m[1] for m in metrics) / len(metrics)
    avg_f1 = sum(m[2] for m in metrics) / len(metrics)
    
    print("\n==========================================")
    print("SUPERVISED EVALUATION METRICS (Custom AI)")
    print("==========================================")
    print(f"Precision : {avg_p:.2f}")
    print(f"Recall    : {avg_r:.2f}")
    print(f"F1 Score  : {avg_f1:.2f}")
    print("==========================================")

if __name__ == "__main__":
    main()