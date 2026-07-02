"""
train_chotta.py

Updated trainer for the Chotta Bheem Malayalam extractive summarizer.

Expected CSV formats:
1) Minimal/backward-compatible:
   sentence,label

2) Recommended/enhanced:
   sentence,label,Article_ID,Sentence_Index,Total_Sentences,Role,...

Only sentence,label are required. If Sentence_Index and Total_Sentences exist, the
MalayalamFeatureExtractor uses them for better position-aware features.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from neuro_symbolic_fusion import HybridFusionClassifier, MalayalamFeatureExtractor


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "training_data_chotta_1000_articles.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "models" / "chotta_bheem.pt"
DEFAULT_CACHE_DIR = BASE_DIR / "cache"


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_sentence_hash(sentences: List[str], model_name: str) -> str:
    h = hashlib.md5()
    h.update(model_name.encode("utf-8"))
    h.update(str(len(sentences)).encode("utf-8"))
    for s in sentences:
        h.update(s.encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()[:16]


class MalayalamHybridDataset(Dataset):
    """Precomputes LaBSE embeddings and symbolic features once, then trains fast."""

    def __init__(
        self,
        csv_file: str | Path,
        embedding_model_name: str = "sentence-transformers/LaBSE",
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        encode_batch_size: int = 64,
        normalize_embeddings: bool = False,
    ) -> None:
        self.csv_file = Path(csv_file)
        print(f"Loading dataset from {self.csv_file}...")

        if not self.csv_file.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_file}")

        self.df = pd.read_csv(self.csv_file)
        required = {"sentence", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        self.df["sentence"] = self.df["sentence"].fillna("").astype(str)
        self.df["label"] = self.df["label"].astype(float)
        self.df = self.df[self.df["sentence"].str.strip().astype(bool)].reset_index(drop=True)

        # Use real article positions when available. Otherwise create safe fallbacks.
        if "Sentence_Index" not in self.df.columns or "Total_Sentences" not in self.df.columns:
            if "Article_ID" in self.df.columns:
                self.df["Sentence_Index"] = self.df.groupby("Article_ID").cumcount()
                self.df["Total_Sentences"] = self.df.groupby("Article_ID")["sentence"].transform("size")
            else:
                print("Warning: Sentence_Index/Total_Sentences missing; using fallback position features.")
                self.df["Sentence_Index"] = 0
                self.df["Total_Sentences"] = 10

        self.sentences: List[str] = self.df["sentence"].tolist()
        self.labels = torch.tensor(self.df["label"].to_numpy(dtype=np.float32)).view(-1, 1)

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = file_sentence_hash(self.sentences, embedding_model_name)
        emb_cache_path = cache_dir / f"labse_embeddings_{cache_key}.npy"

        if emb_cache_path.exists():
            print(f"Loading cached embeddings: {emb_cache_path}")
            embeddings = np.load(emb_cache_path)
        else:
            print(f"Encoding {len(self.sentences):,} sentences with {embedding_model_name}...")
            embedder = SentenceTransformer(embedding_model_name)
            embeddings = embedder.encode(
                self.sentences,
                batch_size=encode_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=normalize_embeddings,
            ).astype(np.float32)
            np.save(emb_cache_path, embeddings)
            print(f"Saved embedding cache: {emb_cache_path}")

        self.semantic = torch.tensor(embeddings, dtype=torch.float32)

        print("Extracting Malayalam symbolic features...")
        feature_extractor = MalayalamFeatureExtractor()
        symbolic_features = []
        for row in tqdm(self.df.itertuples(index=False), total=len(self.df), desc="Symbolic features"):
            sentence = str(getattr(row, "sentence"))
            sentence_index = int(getattr(row, "Sentence_Index"))
            total_sentences = int(getattr(row, "Total_Sentences"))
            try:
                feats = feature_extractor.extract_features(
                    sentence=sentence,
                    sentence_index=sentence_index,
                    total_sentences=total_sentences,
                )
            except TypeError:
                feats = feature_extractor.extract_features(sentence, sentence_index, total_sentences)
            symbolic_features.append(feats)

        self.symbolic = torch.tensor(np.asarray(symbolic_features, dtype=np.float32), dtype=torch.float32)

        if self.symbolic.ndim != 2:
            raise ValueError(f"Symbolic feature tensor must be 2D, got shape {self.symbolic.shape}")

        print(f"Dataset ready: {len(self.df):,} rows")
        print(f"Semantic dim: {self.semantic.shape[1]} | Symbolic dim: {self.symbolic.shape[1]}")
        print(f"Positive labels: {int(self.labels.sum().item()):,} | Negative labels: {len(self.labels) - int(self.labels.sum().item()):,}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.semantic[idx], self.symbolic[idx], self.labels[idx]


def make_train_val_indices(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """Prefer article-level split when Article_ID exists to avoid leakage."""
    rng = np.random.default_rng(seed)

    if "Article_ID" in df.columns:
        article_ids = df["Article_ID"].dropna().unique()
        rng.shuffle(article_ids)
        val_count = max(1, int(len(article_ids) * val_ratio))
        val_articles = set(article_ids[:val_count])
        val_mask = df["Article_ID"].isin(val_articles).to_numpy()
        val_indices = np.where(val_mask)[0].tolist()
        train_indices = np.where(~val_mask)[0].tolist()
    else:
        indices = np.arange(len(df))
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_ratio))
        val_indices = indices[:val_count].tolist()
        train_indices = indices[val_count:].tolist()

    return train_indices, val_indices


def compute_binary_metrics(probs: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).float()
    labels = labels.float()

    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1}


def loss_and_probs(outputs: torch.Tensor, labels: torch.Tensor, output_mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    outputs = outputs.view_as(labels)

    if output_mode == "logit":
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        probs = torch.sigmoid(outputs)
        return loss, probs

    if output_mode == "prob":
        probs = outputs.clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(probs, labels)
        return loss, probs

    # Auto mode: useful if you are unsure, but explicit prob/logit is better.
    with torch.no_grad():
        looks_like_probs = bool(outputs.min().item() >= 0.0 and outputs.max().item() <= 1.0)
    if looks_like_probs:
        probs = outputs.clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(probs, labels)
    else:
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        probs = torch.sigmoid(outputs)
    return loss, probs


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device, output_mode: str) -> dict:
    model.eval()
    losses = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for semantic_batch, symbolic_batch, labels_batch in dataloader:
            semantic_batch = semantic_batch.to(device)
            symbolic_batch = symbolic_batch.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(semantic_batch, symbolic_batch)
            loss, probs = loss_and_probs(outputs, labels_batch, output_mode)

            losses.append(loss.item())
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels_batch.detach().cpu())

    probs = torch.cat(all_probs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    metrics = compute_binary_metrics(probs, labels)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Chotta Bheem Malayalam hybrid summarizer")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to training CSV")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Where to save best model .pt")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR), help="Embedding cache directory")
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/LaBSE")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-mode",
        choices=["prob", "logit", "auto"],
        default="prob",
        help="Use 'prob' if HybridFusionClassifier already applies sigmoid; use 'logit' if it returns raw logits.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🧠 TRAINING MODEL: CHOTTA BHEEM")
    print("=" * 60)
    seed_everything(args.seed)

    dataset = MalayalamHybridDataset(
        csv_file=args.data,
        embedding_model_name=args.embedding_model,
        cache_dir=args.cache_dir,
        encode_batch_size=args.encode_batch_size,
    )

    train_indices, val_indices = make_train_val_indices(dataset.df, args.val_ratio, args.seed)
    print(f"Train rows: {len(train_indices):,} | Val rows: {len(val_indices):,}")

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    semantic_dim = dataset.semantic.shape[1]
    symbolic_dim = dataset.symbolic.shape[1]
    model = HybridFusionClassifier(labse_dim=semantic_dim, symbolic_dim=symbolic_dim)

    device = get_device()
    print(f"Training on device: {device}")
    print(f"Output mode: {args.output_mode}")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_path = output_path.with_name(output_path.stem + "_last.pt")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_losses = []
        running_probs = []
        running_labels = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for semantic_batch, symbolic_batch, labels_batch in pbar:
            semantic_batch = semantic_batch.to(device)
            symbolic_batch = symbolic_batch.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(semantic_batch, symbolic_batch)
            loss, probs = loss_and_probs(outputs, labels_batch, args.output_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_losses.append(loss.item())
            running_probs.append(probs.detach().cpu())
            running_labels.append(labels_batch.detach().cpu())

            pbar.set_postfix({"loss": f"{np.mean(running_losses):.4f}"})

        train_probs = torch.cat(running_probs, dim=0)
        train_labels = torch.cat(running_labels, dim=0)
        train_metrics = compute_binary_metrics(train_probs, train_labels)
        train_loss = float(np.mean(running_losses))

        val_metrics = evaluate(model, val_loader, device, args.output_mode)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_f1={train_metrics['f1']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_f1={val_metrics['f1']:.4f}, "
            f"val_precision={val_metrics['precision']:.4f}, val_recall={val_metrics['recall']:.4f}"
        )

        torch.save(model.state_dict(), last_path)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), output_path)
            print(f"✅ Saved best model to {output_path} (val_loss={best_val_loss:.4f})")

    print("\nTraining complete.")
    print(f"Best model: {output_path}")
    print(f"Last model: {last_path}")


if __name__ == "__main__":
    main()