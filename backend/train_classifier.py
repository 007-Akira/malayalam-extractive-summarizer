import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sentence_transformers import SentenceTransformer
import os

class SentenceClassifier(nn.Module):
    def __init__(self, input_dim=768):
        super(SentenceClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)

def main():
    # 1. Hardware Detection (The M5 Trigger)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"1. Hardware Engine Activated: {device}")

    print("\n2. Loading Training Data...")
    df = pd.read_csv("data/training_data.csv")
    sentences = df['sentence'].tolist()
    labels = df['label'].tolist()
    
    print("\n3. Loading LaBSE to extract features...")
    model = SentenceTransformer("sentence-transformers/LaBSE")
    
    if os.path.exists("data/cached_embeddings.pt"):
        print("   Found cached embeddings! Loading them directly...")
        X_tensor = torch.load("data/cached_embeddings.pt")
    else:
        print(f"   Generating vectors for {len(sentences)} sentences on {device}...")
        embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True)
        X_tensor = torch.tensor(embeddings, dtype=torch.float32)
        torch.save(X_tensor, "data/cached_embeddings.pt")
        print("   Saved vectors to cache.")

    y_tensor = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    print("\n4. Prepping PyTorch DataLoader...")
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    print("\n5. Initializing the Neural Network...")
    # Push the model to the M5 GPU
    classifier = SentenceClassifier(input_dim=768).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    
    epochs = 10
    print(f"\n6. Beginning Training Loop ({epochs} Epochs)...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_X, batch_y in dataloader:
            # Push data batches to the M5 GPU
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = classifier(batch_X)
            loss = criterion(predictions, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            binary_preds = (predictions >= 0.5).float()
            correct_predictions += (binary_preds == batch_y).sum().item()
            total_predictions += batch_y.size(0)
            
        avg_loss = epoch_loss / len(dataloader)
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    print("\n7. Saving the trained model...")
    # Bring the model back to the CPU before saving so it can run anywhere later
    torch.save(classifier.cpu().state_dict(), "models/malayalam_sentence_classifier.pt")
    print("✅ Model Trained and Saved Successfully!")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    main()