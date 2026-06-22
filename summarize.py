import torch
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import the new architecture!
from neuro_symbolic_fusion import HybridFusionClassifier, MalayalamFeatureExtractor

def segment_malayalam_text(text):
    """Splits raw text into sentences cleanly."""
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
    """The main inference pipeline with Dual-Path Hybrid AI & MMR."""
    sentences = segment_malayalam_text(raw_text)
    total_sentences = len(sentences)
    
    if total_sentences <= k:
        return "Article is too short to summarize.", sentences
        
    print(f"-> Article has {total_sentences} sentences. Extracting top {k} with Hybrid AI & MMR...")
    
    # 1. Load the Semantic Engine
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    
    # 2. Load the Symbolic Engine
    feature_extractor = MalayalamFeatureExtractor()
    
    # 3. Load Your Custom Hybrid Brain
    classifier = HybridFusionClassifier(labse_dim=768, symbolic_dim=4)
    # Make sure Godly pushes this new .pt file to your GitHub!
    classifier.load_state_dict(torch.load("models/malayalam_hybrid_classifier.pt", weights_only=True))
    classifier.eval()
    
    # 4. Vectorize Data (Path A: Semantics)
    embeddings = labse_model.encode(sentences)
    X_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    # 5. Extract Features (Path B: Linguistics)
    symbolic_features = []
    for i, sent in enumerate(sentences):
        feats = feature_extractor.extract_features(sent, i, total_sentences)
        symbolic_features.append(feats)
    S_tensor = torch.tensor(symbolic_features, dtype=torch.float32)
    
    # 6. Dual-Path Neural Network Scoring
    with torch.no_grad():
        # Notice how we pass BOTH tensors into the network now!
        probabilities = classifier(X_tensor, S_tensor).squeeze().numpy()
        
    # 7. Extract using MMR
    top_k_indices = extract_with_mmr(embeddings, probabilities, k=k, diversity=diversity)
    
    # 8. Chronological Reordering
    chronological_indices = sorted(top_k_indices)
    summary_sentences = [sentences[i] for i in chronological_indices]
    
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
    print("🤖 HYBRID NEURO-SYMBOLIC AI RUNNING...")
    print("="*50)
    
    summary_text, extracted_list = summarize_article(sample_article, k=2, diversity=0.4)
    
    print("\n✨ FINAL EXTRACTIVE SUMMARY ✨")
    print("-" * 50)
    print(summary_text)
    print("-" * 50)

if __name__ == "__main__":
    main()