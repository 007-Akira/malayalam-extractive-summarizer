import torch
import torch.nn as nn
import re
from sentence_transformers import SentenceTransformer

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
    sentences = re.split(r'\.\s+|\.$', text.strip())
    return [re.sub(r'\s+', ' ', s).strip() for s in sentences if s.strip()]

def summarize_article(raw_text, k=3):
    """The main inference pipeline."""
    sentences = segment_malayalam_text(raw_text)
    
    if len(sentences) <= k:
        return "Article is too short to summarize.", sentences
        
    print(f"-> Article has {len(sentences)} sentences. Extracting top {k}...")
    
    # 1. Load the Language Engine
    labse_model = SentenceTransformer("sentence-transformers/LaBSE")
    
    # 2. Load Your Custom Brain
    classifier = SentenceClassifier(input_dim=768)
    # Load the 90k weights you just trained!
    classifier.load_state_dict(torch.load("models/malayalam_sentence_classifier.pt", weights_only=True))
    classifier.eval() # Lock into prediction mode
    
    # 3. Vectorize the raw sentences
    embeddings = labse_model.encode(sentences)
    X_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    # 4. Ask the Neural Network to grade each sentence (0% to 100% importance)
    with torch.no_grad():
        probabilities = classifier(X_tensor).squeeze()
        
    # 5. Pick the Top-K highest scoring sentences
    top_k_indices = torch.argsort(probabilities, descending=True)[:k].tolist()
    
    # 6. CHRONOLOGICAL REORDERING (Crucial for readability)
    # We sort the indices so the summary flows in the order the author wrote them
    chronological_indices = sorted(top_k_indices)
    
    summary_sentences = [sentences[i] for i in chronological_indices]
    final_summary = " ".join(summary_sentences) + "."
    
    return final_summary, summary_sentences

def main():
    # A random, raw Malayalam news paragraph to test the AI on
    sample_article = """
    തിരുവനന്തപുരം: സംസ്ഥാനത്ത് ഇന്നും ശക്തമായ മഴയ്ക്ക് സാധ്യത. വിവിധ ജില്ലകളിൽ കേന്ദ്ര കാലാവസ്ഥാ വകുപ്പ് യെല്ലോ അലർട്ട് പ്രഖ്യാപിച്ചു. 
    തിരുവനന്തപുരം, കൊല്ലം, പത്തനംതിട്ട, ആലപ്പുഴ, കോട്ടയം, എറണാകുളം, ഇടുക്കി ജില്ലകളിലാണ് യെല്ലോ അലർട്ട്. 
    പൊതുജനങ്ങൾ ജാഗ്രത പാലിക്കണമെന്ന് ദുരന്തനിവാരണ അതോറിറ്റി നിർദേശിച്ചു. 
    മത്സ്യത്തൊഴിലാളികൾ കടലിൽ പോകരുതെന്നും മുന്നറിയിപ്പുണ്ട്. 
    വരും ദിവസങ്ങളിലും മഴ തുടരുമെന്നാണ് കാലാവസ്ഥാ നിരീക്ഷണ കേന്ദ്രത്തിന്റെ വിലയിരുത്തൽ.
    """
    
    print("\n" + "="*50)
    print("🤖 MALAYALAM AI SUMMARIZER RUNNING...")
    print("="*50)
    
    summary_text, extracted_list = summarize_article(sample_article, k=2)
    
    print("\n✨ FINAL EXTRACTIVE SUMMARY ✨")
    print("-" * 50)
    print(summary_text)
    print("-" * 50)

if __name__ == "__main__":
    main()