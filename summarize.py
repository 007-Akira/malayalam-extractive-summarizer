import torch
import torch.nn as nn
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Rebuild the exact Brain Architecture
class SentenceClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super(SentenceClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)

def segment_malayalam_text(text):
    """Splits raw text into sentences cleanly."""
    # FIX: Looks for punctuation and properly keeps them separated
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def extract_with_mmr(embeddings, probabilities, k=3, diversity=0.3):
    """
    Maximal Marginal Relevance (MMR) Extraction.
    Balances picking the most 'important' sentences with 'diverse' sentences.
    """
    num_sentences = len(probabilities)
    if num_sentences <= k:
        return list(range(num_sentences))

    selected_indices = []
    unselected_indices = list(range(num_sentences))

    # Step 1: Pick the single most important sentence first
    first_idx = int(np.argmax(probabilities))
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)

    # Step 2: Pick the remaining sentences by balancing score vs. similarity
    while len(selected_indices) < k:
        mmr_scores = []
        for idx in unselected_indices:
            importance = probabilities[idx]
            
            # Calculate similarity to ALREADY picked sentences
            sims = cosine_similarity(
                [embeddings[idx]], 
                [embeddings[s] for s in selected_indices]
            )[0]
            max_sim = np.max(sims)
            
            # MMR Formula: Score - (Penalty * Similarity)
            adjusted_score = importance - (diversity * max_sim)
            mmr_scores.append((adjusted_score, idx))
            
        # Sort and pick the winner
        mmr_scores.sort(reverse=True, key=lambda x: x[0])
        best_next_idx = mmr_scores[0][1]
        
        selected_indices.append(best_next_idx)
        unselected_indices.remove(best_next_idx)

    return selected_indices

def summarize_article(raw_text, k=3, diversity=0.3):
    """The main inference pipeline with MMR."""
    sentences = segment_malayalam_text(raw_text)
    
    if len(sentences) <= k:
        return "Article is too short to summarize.", sentences
        
    print(f"-> Article has {len(sentences)} sentences. Extracting top {k} with MMR...")
    
    # 1. Load the Language Engine
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    
    # 2. Load Your Custom Brain
    classifier = SentenceClassifier(input_dim=768)
    classifier.load_state_dict(torch.load("models/malayalam_sentence_classifier.pt", weights_only=True))
    classifier.eval()
    
    # 3. Vectorize the raw sentences
    embeddings = labse_model.encode(sentences)
    X_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    # 4. Neural Network Scoring
    with torch.no_grad():
        probabilities = classifier(X_tensor).squeeze().numpy()
        
    # 5. Extract using MMR
    top_k_indices = extract_with_mmr(embeddings, probabilities, k=k, diversity=diversity)
    
    # 6. Chronological Reordering
    chronological_indices = sorted(top_k_indices)
    summary_sentences = [sentences[i] for i in chronological_indices]
    
    # FIX: Added the space here so sentences don't stick together
    final_summary = " ".join(summary_sentences)
    
    return final_summary, summary_sentences

def main():
    sample_article = """
    തിരുവനന്തപുരം: സംസ്ഥാനത്ത് ഇന്നും ശക്തമായ മഴയ്ക്ക് സാധ്യത. വിവിധ ജില്ലകളിൽ കേന്ദ്ര കാലാവസ്ഥാ വകുപ്പ് യെല്ലോ അലർട്ട് പ്രഖ്യാപിച്ചു. 
    തിരുവനന്തപുരം, കൊല്ലം, പത്തനംതിട്ട, ആലപ്പുഴ, കോട്ടയം, എറണാകുളം, ഇടുക്കി ജില്ലകളിലാണ് യെല്ലോ അലർട്ട്. 
    പൊതുജനങ്ങൾ ജാഗ്രത പാലിക്കണമെന്ന് ദുരന്തനിവാരണ അതോറിറ്റി നിർദേശിച്ചു. 
    മത്സ്യത്തൊഴിലാളികൾ കടലിൽ പോകരുതെന്നും മുന്നറിയിപ്പുണ്ട്. 
    വരും ദിവസങ്ങളിലും മഴ തുടരുമെന്നാണ് കാലാവസ്ഥാ നിരീക്ഷണ കേന്ദ്രത്തിന്റെ വിലയിരുത്തൽ.
    """
    
    print("\n" + "="*50)
    print("🤖 MALAYALAM AI SUMMARIZER RUNNING (MMR ENABLED)...")
    print("="*50)
    
    # You can tweak diversity between 0.1 (low penalty) and 0.8 (high penalty)
    summary_text, extracted_list = summarize_article(sample_article, k=2, diversity=0.4)
    
    print("\n✨ FINAL EXTRACTIVE SUMMARY ✨")
    print("-" * 50)
    print(summary_text)
    print("-" * 50)

if __name__ == "__main__":
    main()