import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def segment_malayalam_text(text):
    """
    Splits Malayalam text into sentences safely.
    Ignores periods used in decimals (like 5.50).
    """
    # Split on a period that is followed by one or more spaces (\s+)
    # or a period at the very end of the string ($)
    sentences = re.split(r'\.\s+|\.$', text.strip())
    
    # Clean up any empty strings left behind
    return [re.sub(r'\s+', ' ', s).strip() for s in sentences if s.strip()]

def main():
    # 1. Sample Article
    sample_text = """
    കേരളത്തിൽ ആയതുകൊണ്ടാണ് ജീവനോടെയിരിക്കുന്നതെന്നും മധ്യപ്രദേശിൽ
      പോയാൽ ദുരഭിമാനക്കൊല ചെയ്യുമെന്ന് ഭീഷണിയുണ്ടെന്നും പെണ്‍കുട്ടി 
      കോടതിയെ അറിയിച്ചിരുന്നു. തനിക്ക് പ്രായപൂർത്തിയായെന്നാണ് പെണ്‍കുട്ടി
        അവകാശപ്പെടുന്നത്. എന്നാല്‍, പെണ്‍കുട്ടിക്ക് പ്രായപൂർത്തിയായിട്ടില്ലെന്നാണ് 
        മധ്യപ്രദേശ് സർക്കാരിന്‍റെ വാദം. കഴിഞ്ഞ മാർച്ച് 11 ന് തിരുവനന്തപുരത്തെ 
        അരുമാനൂർ ശ്രീ നൈനാർ ദേവ ക്ഷേത്രത്തിൽ വെച്ചായിരുന്നു കുംഭമേള വൈറൽ 
        താരത്തിന്റെ വിവാഹം. വിവാഹം കഴിക്കുമ്പോള്‍ പെണ്‍കുട്ടിക്ക് 16 വയസ് മാത്രമേ
          പ്രായമുള്ളൂ എന്നായിരുന്നു ദേശീയ പട്ടിക വര്‍ഗ കമ്മീഷന്‍റെ അന്വേഷണത്തില്‍ 
          കണ്ടെത്തിയത്. മഹേശ്വര്‍ സര്‍ക്കാര്‍ ആശുപത്രിയിലെ ജനന സര്‍ട്ടിഫിക്കറ്റില്‍ 
          2009 ഡിസംബര്‍ 30 ന് വൈകിട്ട് 5.50 ന് പെണ്‍കുട്ടി ജനിച്ചു എന്നാണ് 
          രേഖപ്പെടുത്തിയിരിക്കുന്നത്. എന്നാല്‍ തിരുവനന്തപുരത്ത് പൊലീസിന് മുമ്പാകെ 
          പെണ്‍കുട്ടി ഹാജരാക്കിയ രേഖകളില്‍ 2008 ജനുവരി ഒന്നാണ് ജനന തീയതി.
            പെണ്‍കുട്ടി സമര്‍പ്പിച്ച ജനന സര്‍ട്ടിഫിക്കറ്റ് വ്യാജമായി തയ്യാറാക്കിയതെന്നാണ്
              കമ്മീഷന്‍ പറയുന്നത്.
    """

    print("1. Segmenting text...")
    sentences = segment_malayalam_text(sample_text)
    
    # 2. Load the Model
    print("2. Loading LaBSE model...")
    model = SentenceTransformer("sentence-transformers/LaBSE")
    
    # 3. Generate Embeddings
    print("3. Generating embeddings...")
    embeddings = model.encode(sentences)
    
    # ==========================================
    # NEW CODE: TASKS 4 & 5
    # ==========================================
    
    print("4. Calculating Document Centroid and Similarity Scores...")
    # Calculate the average vector (the core theme)
    centroid = np.mean(embeddings, axis=0)
    
    # Calculate how close each sentence is to the core theme
    scores = cosine_similarity(embeddings, centroid.reshape(1, -1)).flatten()
    
    # Print individual scores for debugging
    for i, score in enumerate(scores):
        print(f"   Sentence {i+1} Score: {score:.4f}")

    print("\n5. Extracting and Reordering Top-K Sentences...")
    K = 2 # We want a 2-sentence summary
    
    # Get the indices of the highest scoring sentences
    # .argsort() sorts ascending, so [-K:] gets the highest, [::-1] reverses it to highest-first
    top_k_indices = scores.argsort()[-K:][::-1]
    
    # Chronological Reordering: Sort the indices back to their original document order
    chronological_indices = sorted(top_k_indices)
    
    # Reconstruct the summary
    summary_sentences = [sentences[i] for i in chronological_indices]
    final_summary = " ".join(summary_sentences) + "."
    
    print("\n==========================================")
    print("FINAL EXTRACTIVE SUMMARY:")
    print("==========================================")
    print(final_summary)
    print("==========================================")

if __name__ == "__main__":
    main()