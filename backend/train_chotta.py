import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Import our updated Neuro-Symbolic architecture (with Soft Position Score!)
from neuro_symbolic_fusion import HybridFusionClassifier, MalayalamFeatureExtractor

class MalayalamHybridDataset(Dataset):
    def __init__(self, csv_file):
        print(f"Loading dataset from {csv_file}...")
        self.df = pd.read_csv(csv_file)
        
        # We use LaBSE as the base semantic engine
        self.labse = SentenceTransformer("sentence-transformers/LaBSE")
        self.feature_extractor = MalayalamFeatureExtractor()
        
        # Fallback if indices are missing (though your Oracle should have provided them)
        if 'Sentence_Index' not in self.df.columns:
            self.df['Sentence_Index'] = 0
            self.df['Total_Sentences'] = 10
            
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sentence = str(row['sentence'])
        label = float(row['label']) 
        
        # 1. Path A: Deep Semantics
        labse_embedding = self.labse.encode(sentence)
        labse_tensor = torch.tensor(labse_embedding, dtype=torch.float32)
        
        # 2. Path B: Malayalam Symbolic Features (Now using Soft Lead Bias)
        symbolic_features = self.feature_extractor.extract_features(
            sentence=sentence,
            sentence_index=row['Sentence_Index'],
            total_sentences=row['Total_Sentences']
        )
        symbolic_tensor = torch.tensor(symbolic_features, dtype=torch.float32)
        
        label_tensor = torch.tensor([label], dtype=torch.float32)
        
        return labse_tensor, symbolic_tensor, label_tensor

def main():
    print("="*50)
    print("🧠 TRAINING NEW MODEL: CHOTTA BHEEM (Phase 4.1)")
    print("="*50)
    
    # 1. Load Data
    dataset = MalayalamHybridDataset("data/training_data.csv") 
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 2. Initialize the Dual-Path Brain
    model = HybridFusionClassifier(labse_dim=768, symbolic_dim=4)
    
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on device: {device}")
    model.to(device)
    
    # 3. Standard Loss and Optimizer
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. The Training Loop
    epochs = 3
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for labse_batch, symbolic_batch, labels_batch in progress_bar:
            
            labse_batch = labse_batch.to(device)
            symbolic_batch = symbolic_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            optimizer.zero_grad()
            
            # Feed BOTH inputs to the network
            predictions = model(labse_batch, symbolic_batch)
            
            loss = criterion(predictions, labels_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': total_loss/len(dataloader)})
            
    # 5. Save the new model!
    print("\nSaving new hybrid model weights...")
    torch.save(model.state_dict(), "models/chotta_bheem.pt")
    print("✅ Training Complete! Say hello to models/chotta_bheem.pt")

if __name__ == "__main__":
    main()