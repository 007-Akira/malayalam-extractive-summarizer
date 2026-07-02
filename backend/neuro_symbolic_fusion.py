import torch
import torch.nn as nn
import re
import math

# ==========================================
# PART 1: The Malayalam Feature Extractor
# ==========================================
class MalayalamFeatureExtractor:
    """
    Extracts traditional statistical features to help the neural network 
    understand the structural importance of a sentence in Malayalam news.
    """
    def __init__(self):
        pass

    def extract_features(self, sentence, sentence_index, total_sentences):
        # ----------------------------------------------------
        # NEW Feature 1: Soft Position Score (Phase 4 Upgrade)
        # Prevents the AI from blindly picking the first sentence 
        # by using a capped exponential decay instead of a linear drop.
        # ----------------------------------------------------
        tau = max(3, 0.25 * total_sentences)
        raw_pos = math.exp(-sentence_index / tau)
        position_score = 0.5 * raw_pos
        
        # Feature 2: Sentence Length (Normalized)
        # Extremely short sentences are rarely good summaries. 
        words = sentence.split()
        length_score = min(len(words) / 30.0, 1.0) # Cap at 30 words
        
        # Feature 3: Complex Word Density (Agglutination Proxy)
        # Malayalam forms long words by combining them. A high ratio of long words 
        # often indicates dense, factual information rather than conversational filler.
        long_words = [w for w in words if len(w) > 7]
        complexity_score = len(long_words) / max(1, len(words))
        
        # Feature 4: Numeral/Data Density
        # Sentences with numbers (dates, casualties, money) are highly critical in news.
        numbers = len(re.findall(r'\d+', sentence))
        numeral_density = min(numbers / 5.0, 1.0) # Cap at 5 numbers
        
        # Return as a tensor
        return [position_score, length_score, complexity_score, numeral_density]


# ==========================================
# PART 2: The Dual-Path Neural Network
# ==========================================
class HybridFusionClassifier(nn.Module):
    def __init__(self, labse_dim=768, symbolic_dim=4):
        super(HybridFusionClassifier, self).__init__()
        
        # PATH A: The Deep Semantic Engine (Compresses LaBSE)
        self.semantic_path = nn.Sequential(
            nn.Linear(labse_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # PATH B: The Linguistic Engine (Processes our custom Malayalam features)
        self.symbolic_path = nn.Sequential(
            nn.Linear(symbolic_dim, 16),
            nn.ReLU()
        )
        
        # THE FUSION LAYER: Combines both "brains" to make a final decision
        # 256 (from semantics) + 16 (from symbolic) = 272
        self.fusion_classifier = nn.Sequential(
            nn.Linear(256 + 16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, labse_vector, symbolic_vector):
        # 1. Process semantics
        semantic_features = self.semantic_path(labse_vector)
        
        # 2. Process linguistic stats
        symbolic_features = self.symbolic_path(symbolic_vector)
        
        # 3. FUSION (Concatenate the vectors side-by-side)
        fused_vector = torch.cat((semantic_features, symbolic_features), dim=1)
        
        # 4. Final Classification
        output = self.fusion_classifier(fused_vector)
        return output