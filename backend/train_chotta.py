"""
train_chotta_finetune.py

Continued-training / fine-tuning script for the Chotta Bheem Malayalam
neuro-symbolic extractive summarizer.

Key behavior:
- Loads an existing checkpoint with --resume-from.
- Can train on one or multiple CSV files at once.
- Can combine the 10,000-article synthetic set and the earlier 1,000-article set.
- Deduplicates exact duplicate rows so repeated synthetic sentences do not dominate.
- Precomputes/caches LaBSE embeddings.
- Uses article-level train/validation split when Article_ID exists.
- Saves both best checkpoint and last checkpoint.

Expected CSV formats:

Minimal:
    sentence,label

Enhanced:
    sentence,label,Article_ID,Sentence_Index,Total_Sentences,Role,...

Recommended fine-tuning command:

python train_chotta_finetune.py \
  --data data/training_data_chotta_10000_articles.csv data/training_data_chotta_1000_articles.csv \
  --resume-from models/chotta_bheem.pt \
  --output models/chotta_bheem_finetuned.pt \
  --last-output models/chotta_bheem_finetuned_last.pt \
  --epochs 2 \
  --batch-size 64 \
  --lr 0.0001 \
  --output-mode prob
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Iterable, List, Tuple

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


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes likely column name variations without breaking existing files."""
    rename_map = {}

    for col in df.columns:
        c = col.strip()
        lower = c.lower()

        if lower in {"sentence_text", "text", "sent"}:
            rename_map[col] = "sentence"
        elif lower in {"summary_label", "is_summary", "target"}:
            rename_map[col] = "label"
        elif lower in {"article_id", "doc_id", "document_id"}:
            rename_map[col] = "Article_ID"
        elif lower in {"sentence_index", "sent_index", "idx"}:
            rename_map[col] = "Sentence_Index"
        elif lower in {"total_sentences", "num_sentences", "n_sentences"}:
            rename_map[col] = "Total_Sentences"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def load_and_merge_csvs(csv_files: Iterable[str | Path], dedupe: bool = True) -> pd.DataFrame:
    frames = []

    for path in csv_files:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        print(f"Loading dataset: {path}")
        df = pd.read_csv(path)
        df = _normalize_columns(df)

        required = {"sentence", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        df["Source_File"] = path.name

        if "Article_ID" in df.columns:
            df["Article_ID"] = df["Article_ID"].astype(str)
            df["Article_ID"] = path.stem + "::" + df["Article_ID"]
        else:
            pseudo_ids = np.arange(len(df)) // 15
            df["Article_ID"] = path.stem + "::pseudo_" + pd.Series(pseudo_ids).astype(str)

        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    merged["sentence"] = merged["sentence"].fillna("").astype(str).str.strip()
    merged = merged[merged["sentence"].astype(bool)].reset_index(drop=True)
    merged["label"] = merged["label"].astype(float).clip(0.0, 1.0)

    before = len(merged)

    if dedupe:
        # Keeps the 1,000-set useful while preventing exact duplicates from overweighting training.
        merged = merged.drop_duplicates(subset=["sentence", "label"]).reset_index(drop=True)

    after = len(merged)
    print(f"Merged rows: {before:,}")
    if dedupe:
        print(f"After exact sentence+label dedupe: {after:,} rows | Removed: {before - after:,}")

    if "Sentence_Index" not in merged.columns:
        merged["Sentence_Index"] = merged.groupby("Article_ID").cumcount()

    if "Total_Sentences" not in merged.columns:
        merged["Total_Sentences"] = merged.groupby("Article_ID")["sentence"].transform("size")

    merged["Sentence_Index"] = pd.to_numeric(merged["Sentence_Index"], errors="coerce").fillna(0).astype(int)
    merged["Total_Sentences"] = pd.to_numeric(merged["Total_Sentences"], errors="coerce").fillna(10).astype(int)

    return merged


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
        df: pd.DataFrame,
        embedding_model_name: str = "sentence-transformers/LaBSE",
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        encode_batch_size: int = 64,
        normalize_embeddings: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.sentences: List[str] = self.df["sentence"].astype(str).tolist()
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

        positives = int(self.labels.sum().item())
        negatives = len(self.labels) - positives

        print(f"Dataset ready: {len(self.df):,} rows")
        print(f"Semantic dim: {self.semantic.shape[1]} | Symbolic dim: {self.symbolic.shape[1]}")
        print(f"Positive labels: {positives:,} | Negative labels: {negatives:,}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.semantic[idx], self.symbolic[idx], self.labels[idx]


def make_train_val_indices(df: pd.DataFrame, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """Article-level split to avoid leakage between train and validation."""
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


def _extract_state_dict(checkpoint_obj):
    """Supports raw state_dict or wrapped checkpoint dictionaries."""
    if isinstance(checkpoint_obj, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return checkpoint_obj[key]

    return checkpoint_obj


def load_checkpoint_if_provided(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> None:
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    print(f"Loading base checkpoint for continued training: {path}")

    try:
        checkpoint_obj = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint_obj = torch.load(path, map_location=device)

    state_dict = _extract_state_dict(checkpoint_obj)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint could not be loaded into HybridFusionClassifier. "
            "This usually means the architecture in neuro_symbolic_fusion.py changed, "
            "or labse_dim/symbolic_dim does not match the saved checkpoint."
        ) from exc

    print("✅ Base checkpoint loaded successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Chotta Bheem Malayalam hybrid summarizer from an existing checkpoint."
    )

    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="One or more CSV files. Pass both 10000 and 1000 article datasets if available.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        required=True,
        help="Existing base checkpoint to continue training from, e.g. models/chotta_bheem.pt",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(BASE_DIR / "models" / "chotta_bheem_finetuned.pt"),
        help="Where to save best validation checkpoint.",
    )
    parser.add_argument(
        "--last-output",
        type=str,
        default=str(BASE_DIR / "models" / "chotta_bheem_finetuned_last.pt"),
        help="Where to save last epoch checkpoint.",
    )
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--embedding-model", type=str, default="sentence-transformers/LaBSE")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-mode",
        choices=["prob", "logit", "auto"],
        default="prob",
        help="Use 'prob' if HybridFusionClassifier already applies sigmoid; use 'logit' if it returns raw logits.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable exact sentence+label deduplication. Not recommended for synthetic data.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🧠 CONTINUED TRAINING: CHOTTA BHEEM FINE-TUNE")
    print("=" * 70)
    seed_everything(args.seed)

    merged_df = load_and_merge_csvs(args.data, dedupe=not args.no_dedupe)

    dataset = MalayalamHybridDataset(
        df=merged_df,
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
    print(f"Fine-tuning LR: {args.lr}")

    model.to(device)
    load_checkpoint_if_provided(model, args.resume_from, device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    last_path = Path(args.last_output)
    last_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_losses = []
        running_probs = []
        running_labels = []

        pbar = tqdm(train_loader, desc=f"Fine-tune Epoch {epoch}/{args.epochs}")

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
        print(f"Saved last checkpoint: {last_path}")

        if (val_metrics["loss"] < best_val_loss) or (
            np.isclose(val_metrics["loss"], best_val_loss) and val_metrics["f1"] > best_val_f1
        ):
            best_val_loss = val_metrics["loss"]
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), output_path)
            print(
                f"✅ Saved best fine-tuned model to {output_path} "
                f"(val_loss={best_val_loss:.4f}, val_f1={best_val_f1:.4f})"
            )

    print("\nFine-tuning complete.")
    print(f"Base checkpoint: {args.resume_from}")
    print(f"Best fine-tuned model: {output_path}")
    print(f"Last fine-tuned model: {last_path}")


if __name__ == "__main__":
    main()