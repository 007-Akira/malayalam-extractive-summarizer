"""
summarize.py

Malayalam extractive summarizer inference script.

Pipeline:
1. Malayalam sentence segmentation
2. LaBSE sentence embeddings
3. Neuro-symbolic salience scoring
4. Dynamic summary length
5. Coverage-aware Dynamic MMR selection
6. Chronological output ordering

Expected project structure:
project_root/
├── summarize.py
├── neuro_symbolic_fusion.py
└── models/
    └── chotta_bheem.pt
"""

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from neuro_symbolic_fusion import HybridFusionClassifier, MalayalamFeatureExtractor


try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DEFAULT_ENCODER_NAME = "sentence-transformers/LaBSE"
MURIL_ENCODER_NAME = "l3cube-pune/indic-sentence-bert-nli"
LABSE_DIM = 768
SYMBOLIC_DIM = 4
DEFAULT_MODEL_KEY = "chotta_bheem"

MODEL_REGISTRY = {
    "sentence_classifier": {
        "label": "Sentence Classifier",
        "path": BASE_DIR / "models" / "malayalam_sentence_classifier.pt",
        "architecture": "sentence",
        "encoder": DEFAULT_ENCODER_NAME,
    },
    "hybrid_classifier": {
        "label": "Hybrid Classifier",
        "path": BASE_DIR / "models" / "malayalam_hybrid_classifier.pt",
        "architecture": "hybrid",
        "encoder": DEFAULT_ENCODER_NAME,
    },
    "muril_classifier": {
        "label": "MuRIL Classifier",
        "path": BASE_DIR / "models" / "muril_classifier.pt",
        "architecture": "hybrid",
        "encoder": MURIL_ENCODER_NAME,
    },
    "chotta_bheem": {
        "label": "Chotta Bheem",
        "path": BASE_DIR / "models" / "chotta_bheem.pt",
        "architecture": "hybrid",
        "encoder": DEFAULT_ENCODER_NAME,
    },
}


class SentenceClassifier(nn.Module):
    """Single-path classifier used by the first supervised checkpoint."""

    def __init__(self, input_dim: int = LABSE_DIM) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sigmoid(self.linear(x))


# -----------------------------------------------------------------------------
# Sentence segmentation
# -----------------------------------------------------------------------------

def segment_malayalam_text(text: str) -> List[str]:
    """
    Split Malayalam news text into sentences.

    Handles:
    - regular full stop / question mark / exclamation mark
    - Devanagari danda-like punctuation if present
    - line-separated input when punctuation splitting fails
    """
    if not text or not text.strip():
        return []

    raw_lines = [line.strip() for line in text.strip().splitlines() if line.strip()]

    normalized = re.sub(r"\s+", " ", text.strip())
    parts = re.split(r"(?<=[.!?।])\s+", normalized)
    sentences = [clean_sentence(part) for part in parts]
    sentences = [sent for sent in sentences if len(sent) > 2]

    # If punctuation-based splitting failed, fall back to non-empty lines.
    if len(sentences) <= 1 and len(raw_lines) > 1:
        sentences = [clean_sentence(line) for line in raw_lines]
        sentences = [sent for sent in sentences if len(sent) > 2]

    return sentences


def clean_sentence(sentence: str) -> str:
    """Remove common bullet/numbering noise without changing sentence content."""
    sentence = sentence.strip()
    sentence = re.sub(r"^[\-–—•*]+\s*", "", sentence)
    sentence = re.sub(r"^\d+[.)]\s*", "", sentence)
    return sentence.strip()


# -----------------------------------------------------------------------------
# Selection features: specificity, factual status, sentence roles
# -----------------------------------------------------------------------------

MALAYALAM_NUMBER_WORDS = [
    "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്",
    "ഒമ്പത്", "പത്ത്", "പത്തു", "പതിനൊന്ന്", "പന്ത്രണ്ട്", "നൂറ്", "ആയിരം",
    "ലക്ഷം", "കോടി",
]

SPECIFICITY_TERMS = [
    "ആദ്യഘട്ടത്തിൽ", "റൂട്ടുകളിൽ", "റൂട്ടുകളിലാണ്", "ജില്ലകളിൽ", "അലർട്ട്",
    "ജിപിഎസ്", "സിസിടിവി", "ഡിജിറ്റൽ", "മണിക്കൂർ", "ശതമാനം",
    "ലക്ഷം", "കോടി", "തീയതി", "സമയക്രമം", "പദ്ധതി", "സർവീസ്",
]

STRONG_CONDITIONAL_TERMS = [
    "വിജയകരമായാൽ", "എങ്കിൽ", "ആയാൽ", "പരിഗണനയിലാണ്", "പരിഗണിക്കുന്നു",
    "ആലോചിക്കുന്നു", "സാധിച്ചാൽ",
]

WEAK_FUTURE_TERMS = [
    "വ്യാപിപ്പിക്കും", "ആരംഭിക്കാനാണ്", "നടപ്പാക്കാനാണ്", "തുടങ്ങാനാണ്",
]

ROLE_KEYWORDS: Dict[str, Sequence[str]] = {
    "EVENT": [
        "ആരംഭിച്ചു", "തുടങ്ങി", "പ്രഖ്യാപിച്ചു", "നടന്നു", "നടത്തി", "ഉദ്ഘാടനം",
        "സ്ഥാപിച്ചു", "പുറത്തിറക്കി", "അവതരിപ്പിച്ചു",
    ],
    "DATA_SCALE": [
        "ആദ്യഘട്ടത്തിൽ", "റൂട്ടുകളിലാണ്", "റൂട്ടുകളിൽ", "ജില്ലകളിൽ", "ശതമാനം",
        "ലക്ഷം", "കോടി", "മണിക്കൂർ", "അലർട്ട്",
    ],
    "PUBLIC_WARNING": [
        "ജാഗ്രത", "മുന്നറിയിപ്പ്", "ഒഴിവാക്കണം", "പോകരുത്", "പാലിക്കണമെന്ന്",
        "ശ്രദ്ധിക്കണം", "പിന്തുടരണമെന്ന്", "നിർദേശം",
    ],
    "OFFICIAL_RESPONSE": [
        "അധികൃതർ", "വകുപ്പ്", "നഗരസഭ", "ഭരണകൂടം", "അതോറിറ്റി", "സർക്കാർ",
        "പഞ്ചായത്ത്", "പോലീസ്", "പൊലീസ്", "അഗ്നിരക്ഷാ", "ഉദ്യോഗസ്ഥർ",
        "മേയർ", "കളക്ടർ", "മന്ത്രി",
    ],
    "RISK_IMPACT": [
        "ഭീഷണി", "അപകടം", "മരണം", "പരിക്ക്", "നാശനഷ്ടം", "മണ്ണിടിച്ചിൽ",
        "വെള്ളപ്പൊക്കം", "മഴ", "ഗതാഗതക്കുരുക്ക്", "മലിനീകരണം", "നിയന്ത്രണം",
    ],
    "FUTURE_PLAN": [
        "വരും ദിവസങ്ങളിൽ", "വരും ദിവസങ്ങളിലും", "വ്യാപിപ്പിക്കും", "ആരംഭിക്കാനാണ്",
        "തുടരുമെന്നാണ്", "തീരുമാനം", "പദ്ധതി വിജയകരമായാൽ",
    ],
}


def has_digit_or_number_word(sentence: str) -> bool:
    """Return True if the sentence contains a digit or common Malayalam number word."""
    if re.search(r"\d", sentence):
        return True
    return any(word in sentence for word in MALAYALAM_NUMBER_WORDS)


def specificity_bonus(sentence: str) -> float:
    """
    Small bonus for concrete details: scale, route count, dates, numbers,
    named facilities, alert levels, etc.
    """
    bonus = 0.0

    if any(term in sentence for term in SPECIFICITY_TERMS):
        bonus += 0.06

    if has_digit_or_number_word(sentence):
        bonus += 0.04

    return min(bonus, 0.10)


def future_condition_penalty(sentence: str) -> float:
    """
    Penalize uncertain/conditional future sentences slightly.

    Kept small because future information can be important in weather, policy,
    and public-service articles.
    """
    if any(term in sentence for term in STRONG_CONDITIONAL_TERMS):
        return 0.05

    if any(term in sentence for term in WEAK_FUTURE_TERMS):
        return 0.02

    return 0.0


def detect_roles(sentence: str) -> Set[str]:
    """Weak role detector used only for selection-time coverage bonuses."""
    roles: Set[str] = set()

    for role, keywords in ROLE_KEYWORDS.items():
        if any(keyword in sentence for keyword in keywords):
            roles.add(role)

    if has_digit_or_number_word(sentence):
        roles.add("DATA_SCALE")

    if not roles:
        roles.add("GENERAL")

    return roles


# -----------------------------------------------------------------------------
# Dynamic diversity and clustering
# -----------------------------------------------------------------------------

def compute_dynamic_diversity(embeddings: np.ndarray) -> float:
    """
    Estimate article repetitiveness and choose an MMR diversity penalty.

    Higher average semantic overlap => stronger diversity penalty.
    """
    n = len(embeddings)
    if n <= 1:
        return 0.30

    sim_matrix = cosine_similarity(embeddings)
    pairwise = sim_matrix[np.triu_indices(n, k=1)]

    # Negative cosine similarity should not reduce the article-level average.
    pairwise = np.clip(pairwise, 0.0, 1.0)
    avg_sim = float(np.mean(pairwise)) if len(pairwise) else 0.0

    if avg_sim >= 0.75:
        penalty = 0.65
    elif avg_sim >= 0.60:
        penalty = 0.55
    elif avg_sim >= 0.45:
        penalty = 0.45
    elif avg_sim >= 0.30:
        penalty = 0.35
    else:
        penalty = 0.25

    return round(penalty, 2)


def assign_similarity_clusters(embeddings: np.ndarray, threshold: float = 0.72) -> List[int]:
    """
    Greedy clustering for sentence-level semantic groups.

    This is intentionally lightweight: no extra dependency beyond sklearn.
    """
    n = len(embeddings)
    if n == 0:
        return []

    clusters: List[int] = []
    centroids: List[np.ndarray] = []
    members: List[List[int]] = []

    for i, emb in enumerate(embeddings):
        emb_2d = np.asarray(emb).reshape(1, -1)

        if not centroids:
            clusters.append(0)
            centroids.append(np.asarray(emb, dtype=np.float32))
            members.append([i])
            continue

        centroid_matrix = np.vstack(centroids)
        sims = cosine_similarity(emb_2d, centroid_matrix)[0]
        best_cluster = int(np.argmax(sims))
        best_sim = float(sims[best_cluster])

        if best_sim >= threshold:
            clusters.append(best_cluster)
            members[best_cluster].append(i)
            centroids[best_cluster] = np.mean(embeddings[members[best_cluster]], axis=0)
        else:
            new_cluster_id = len(centroids)
            clusters.append(new_cluster_id)
            centroids.append(np.asarray(emb, dtype=np.float32))
            members.append([i])

    return clusters


# -----------------------------------------------------------------------------
# Coverage-aware MMR
# -----------------------------------------------------------------------------

def minmax_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize values to [0, 1]. If all values are equal, return them unchanged."""
    values = np.asarray(values, dtype=np.float32)
    min_val = float(np.min(values))
    max_val = float(np.max(values))

    if abs(max_val - min_val) < 1e-8:
        return values

    return (values - min_val) / (max_val - min_val)


def extract_with_mmr(
    sentences: Sequence[str],
    embeddings: np.ndarray,
    probabilities: np.ndarray,
    k: int = 3,
    diversity: Optional[float] = None,
    use_auto_diversity: bool = True,
    use_normalized_relevance: bool = False,
    debug: bool = False,
) -> List[int]:
    """
    Coverage-aware Dynamic MMR extraction.

    Score = relevance
            - semantic_redundancy_penalty
            + role_coverage_bonus
            + cluster_coverage_bonus
            + specificity_bonus
            - future_condition_penalty
    """
    num_sentences = len(probabilities)
    if num_sentences == 0:
        return []

    k = max(1, min(int(k), num_sentences))
    if num_sentences <= k:
        return list(range(num_sentences))

    if diversity is None and use_auto_diversity:
        diversity = compute_dynamic_diversity(embeddings)
        print(f"   [Dynamic MMR] Auto-tuned diversity penalty to: {diversity}")
    elif diversity is None:
        diversity = 0.35

    probabilities = np.asarray(probabilities, dtype=np.float32)
    relevance_scores = minmax_normalize(probabilities) if use_normalized_relevance else probabilities

    clusters = assign_similarity_clusters(embeddings)
    sentence_roles = [detect_roles(sentence) for sentence in sentences]

    selected_indices: List[int] = []
    unselected_indices: List[int] = list(range(num_sentences))

    # First pick: strongest salience. Do not use bonuses here; keep it model-led.
    first_idx = int(np.argmax(relevance_scores))
    selected_indices.append(first_idx)
    unselected_indices.remove(first_idx)

    while len(selected_indices) < k:
        mmr_scores: List[Tuple[float, int, Dict[str, float]]] = []

        selected_embeddings = np.vstack([embeddings[s] for s in selected_indices])
        selected_roles: Set[str] = set()
        for selected_idx in selected_indices:
            selected_roles.update(sentence_roles[selected_idx])

        selected_clusters = {clusters[selected_idx] for selected_idx in selected_indices}

        for idx in unselected_indices:
            importance = float(relevance_scores[idx])

            sims = cosine_similarity(
                np.asarray(embeddings[idx]).reshape(1, -1),
                selected_embeddings,
            )[0]
            max_sim = float(np.max(sims)) if len(sims) else 0.0

            new_roles = sentence_roles[idx] - selected_roles
            role_bonus = min(0.10, 0.04 * len(new_roles))

            if clusters[idx] not in selected_clusters:
                cluster_bonus = 0.05
            else:
                cluster_bonus = -0.04

            spec_bonus = specificity_bonus(sentences[idx])
            fut_penalty = future_condition_penalty(sentences[idx])

            adjusted_score = (
                importance
                - float(diversity) * max_sim
                + role_bonus
                + cluster_bonus
                + spec_bonus
                - fut_penalty
            )

            components = {
                "importance": importance,
                "max_sim": max_sim,
                "role_bonus": role_bonus,
                "cluster_bonus": cluster_bonus,
                "specificity_bonus": spec_bonus,
                "future_penalty": fut_penalty,
                "adjusted_score": adjusted_score,
            }
            mmr_scores.append((adjusted_score, idx, components))

        mmr_scores.sort(reverse=True, key=lambda item: item[0])
        best_next_idx = mmr_scores[0][1]
        selected_indices.append(best_next_idx)
        unselected_indices.remove(best_next_idx)

        if debug:
            best_components = mmr_scores[0][2]
            print(
                f"   [MMR Pick] idx={best_next_idx} "
                f"score={best_components['adjusted_score']:.4f} "
                f"importance={best_components['importance']:.4f} "
                f"sim={best_components['max_sim']:.4f}"
            )

    return selected_indices


# -----------------------------------------------------------------------------
# Utility/debug output
# -----------------------------------------------------------------------------

def choose_dynamic_k(total_sentences: int) -> int:
    """Choose summary length based on article size."""
    if total_sentences <= 1:
        return total_sentences
    if total_sentences <= 3:
        return 1
    if total_sentences <= 5:
        return 2
    if total_sentences <= 10:
        return 3
    if total_sentences <= 18:
        return 4
    return 5


def print_debug_table(
    sentences: Sequence[str],
    probabilities: np.ndarray,
    selected_indices: Iterable[int],
    embeddings: Optional[np.ndarray] = None,
) -> None:
    """Print sentence-level scores and selection metadata."""
    selected_set = set(selected_indices)
    clusters = assign_similarity_clusters(embeddings) if embeddings is not None and len(embeddings) else [-1] * len(sentences)

    print("\nDEBUG SENTENCE SCORES")
    print("-" * 110)
    print(f"{'SEL':<4} {'IDX':<4} {'PROB':<8} {'CL':<4} {'ROLES':<35} SENTENCE")
    print("-" * 110)

    for i, sentence in enumerate(sentences):
        selected = "YES" if i in selected_set else ""
        prob = float(probabilities[i]) if i < len(probabilities) else 0.0
        roles = ",".join(sorted(detect_roles(sentence)))
        cluster_id = clusters[i] if i < len(clusters) else -1
        print(f"{selected:<4} {i:<4} {prob:<8.4f} {cluster_id:<4} {roles:<35} {sentence}")

    print("-" * 110)


# -----------------------------------------------------------------------------
# Main summarizer class
# -----------------------------------------------------------------------------

class MalayalamSummarizer:
    """Reusable summarizer that loads LaBSE and the classifier only once."""

    def __init__(
        self,
        model_key: str = DEFAULT_MODEL_KEY,
        model_path: Optional[Path] = None,
        encoder_name: Optional[str] = None,
        device: Optional[str] = None,
        classifier_outputs_logits: Optional[bool] = None,
    ) -> None:
        if model_key not in MODEL_REGISTRY:
            available_models = ", ".join(MODEL_REGISTRY)
            raise ValueError(f"Unknown model '{model_key}'. Available models: {available_models}")

        model_config = MODEL_REGISTRY[model_key]
        self.model_key = model_key
        self.model_label = str(model_config["label"])
        self.architecture = str(model_config["architecture"])
        self.model_path = Path(model_path) if model_path else Path(model_config["path"])
        self.encoder_name = encoder_name if encoder_name else str(model_config["encoder"])
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.classifier_outputs_logits = classifier_outputs_logits

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.model_path}\n"
                f"Selected model: {self.model_label} ({self.model_key})"
            )

        self.labse_model = SentenceTransformer(self.encoder_name, device=str(self.device))
        self.feature_extractor = MalayalamFeatureExtractor()

        if self.architecture == "sentence":
            self.classifier = SentenceClassifier(input_dim=LABSE_DIM)
        elif self.architecture == "hybrid":
            self.classifier = HybridFusionClassifier(labse_dim=LABSE_DIM, symbolic_dim=SYMBOLIC_DIM)
        else:
            raise ValueError(f"Unsupported model architecture: {self.architecture}")

        self.classifier.to(self.device)
        self._load_classifier_weights(self.model_path)
        self.classifier.eval()

    def _load_classifier_weights(self, model_path: Path) -> None:
        """Load model weights with compatibility across PyTorch versions."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Expected checkpoint to be a state_dict or a dict containing 'state_dict'. "
                f"Got: {type(checkpoint)}"
            )

        # Remove DataParallel prefix if present.
        cleaned_state_dict = {}
        for key, value in checkpoint.items():
            new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
            cleaned_state_dict[new_key] = value

        self.classifier.load_state_dict(cleaned_state_dict)

    def _classifier_to_probabilities(self, outputs: torch.Tensor) -> np.ndarray:
        """
        Convert classifier outputs to probabilities safely.

        Set classifier_outputs_logits explicitly if you know your architecture:
        - True  => apply sigmoid
        - False => assume model already returns probabilities
        - None  => auto-detect based on output range
        """
        outputs = outputs.squeeze(-1)

        if self.classifier_outputs_logits is True:
            probs = torch.sigmoid(outputs)
        elif self.classifier_outputs_logits is False:
            probs = outputs
        else:
            # Auto-detect. If values fall outside [0, 1], they are almost certainly logits.
            min_val = float(torch.min(outputs).detach().cpu())
            max_val = float(torch.max(outputs).detach().cpu())
            if min_val < 0.0 or max_val > 1.0:
                probs = torch.sigmoid(outputs)
            else:
                probs = outputs

        probs = probs.detach().cpu().numpy()
        probs = np.asarray(probs, dtype=np.float32).reshape(-1)
        probs = np.clip(probs, 0.0, 1.0)
        return probs

    def score_sentences(self, sentences: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return sentence embeddings and classifier probabilities."""
        total_sentences = len(sentences)
        if total_sentences == 0:
            return np.empty((0, LABSE_DIM), dtype=np.float32), np.array([], dtype=np.float32)

        embeddings = self.labse_model.encode(
            list(sentences),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        x_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if self.architecture == "sentence":
                outputs = self.classifier(x_tensor)
            else:
                symbolic_features = []
                for i, sentence in enumerate(sentences):
                    features = self.feature_extractor.extract_features(sentence, i, total_sentences)
                    features = np.asarray(features, dtype=np.float32).reshape(-1)

                    if features.shape[0] != SYMBOLIC_DIM:
                        raise ValueError(
                            f"Expected {SYMBOLIC_DIM} symbolic features, got {features.shape[0]} "
                            f"for sentence index {i}. Features: {features}"
                        )

                    symbolic_features.append(features)

                s_tensor = torch.tensor(np.vstack(symbolic_features), dtype=torch.float32, device=self.device)
                outputs = self.classifier(x_tensor, s_tensor)

            probabilities = self._classifier_to_probabilities(outputs)

        return embeddings, probabilities

    def summarize(
        self,
        raw_text: str,
        k: Optional[int] = None,
        diversity: Union[str, float] = "auto",
        debug: bool = False,
        use_normalized_relevance: bool = False,
    ) -> Tuple[str, List[str]]:
        """Summarize raw Malayalam article text extractively."""
        sentences = segment_malayalam_text(raw_text)
        total_sentences = len(sentences)

        if total_sentences == 0:
            return "", []

        if total_sentences == 1:
            return sentences[0], [sentences[0]]

        if k is None:
            k = choose_dynamic_k(total_sentences)
        else:
            k = int(k)

        # Keep the summary shorter than the article unless the article has only one sentence.
        k = max(1, min(k, total_sentences - 1))

        print(
            f"-> Article has {total_sentences} sentences. "
            f"Extracting top {k} with {self.model_label} + Coverage D-MMR..."
        )

        embeddings, probabilities = self.score_sentences(sentences)

        if diversity == "auto":
            diversity_value = None
            use_auto_diversity = True
        else:
            diversity_value = float(diversity)
            use_auto_diversity = False

        top_k_indices = extract_with_mmr(
            sentences=sentences,
            embeddings=embeddings,
            probabilities=probabilities,
            k=k,
            diversity=diversity_value,
            use_auto_diversity=use_auto_diversity,
            use_normalized_relevance=use_normalized_relevance,
            debug=debug,
        )

        chronological_indices = sorted(top_k_indices)
        summary_sentences = [sentences[i] for i in chronological_indices]
        final_summary = "\n".join(summary_sentences)

        if debug:
            print_debug_table(sentences, probabilities, chronological_indices, embeddings=embeddings)

        return final_summary, summary_sentences


# -----------------------------------------------------------------------------
# Backward-compatible function API
# -----------------------------------------------------------------------------

_SUMMARIZER_CACHE: Dict[str, MalayalamSummarizer] = {}


def get_summarizer(model_key: str = DEFAULT_MODEL_KEY) -> MalayalamSummarizer:
    """Cache the summarizer so repeated calls do not reload models."""
    global _SUMMARIZER_CACHE
    if model_key not in _SUMMARIZER_CACHE:
        _SUMMARIZER_CACHE[model_key] = MalayalamSummarizer(model_key=model_key)
    return _SUMMARIZER_CACHE[model_key]


def summarize_article(
    raw_text: str,
    k: Optional[int] = None,
    diversity: Union[str, float] = "auto",
    model_key: str = DEFAULT_MODEL_KEY,
    debug: bool = False,
    use_normalized_relevance: bool = False,
) -> Tuple[str, List[str]]:
    """Backward-compatible wrapper around MalayalamSummarizer.summarize()."""
    summarizer = get_summarizer(model_key=model_key)
    return summarizer.summarize(
        raw_text=raw_text,
        k=k,
        diversity=diversity,
        debug=debug,
        use_normalized_relevance=use_normalized_relevance,
    )


# -----------------------------------------------------------------------------
# CLI test
# -----------------------------------------------------------------------------

def main() -> None:
    sample_article = """
    തിരുവനന്തപുരം: സംസ്ഥാനത്ത് ഇന്നും ശക്തമായ മഴയ്ക്ക് സാധ്യത. വിവിധ ജില്ലകളിൽ കേന്ദ്ര കാലാവസ്ഥാ വകുപ്പ് യെല്ലോ അലർട്ട് പ്രഖ്യാപിച്ചു.
    തിരുവനന്തപുരം, കൊല്ലം, പത്തനംതിട്ട, ആലപ്പുഴ, കോട്ടയം, എറണാകുളം, ഇടുക്കി ജില്ലകളിലാണ് യെല്ലോ അലർട്ട്.
    പൊതുജനങ്ങൾ ജാഗ്രത പാലിക്കണമെന്ന് ദുരന്തനിവാരണ അതോറിറ്റി നിർദേശിച്ചു.
    മത്സ്യത്തൊഴിലാളികൾ കടലിൽ പോകരുതെന്നും മുന്നറിയിപ്പുണ്ട്.
    വരും ദിവസങ്ങളിലും മഴ തുടരുമെന്നാണ് കാലാവസ്ഥാ നിരീക്ഷണ കേന്ദ്രത്തിന്റെ വിലയിരുത്തൽ.
    """

    print("\n" + "=" * 60)
    print("HYBRID NEURO-SYMBOLIC MALAYALAM SUMMARIZER")
    print("Coverage-aware Dynamic MMR enabled")
    print("=" * 60)

    summary_text, extracted_list = summarize_article(
        sample_article,
        k=2,
        diversity="auto",
        model_key=DEFAULT_MODEL_KEY,
        debug=True,
    )

    print("\nFINAL EXTRACTIVE SUMMARY")
    print("-" * 60)
    print(summary_text)
    print("-" * 60)
    print(f"Extracted {len(extracted_list)} sentence(s).")


if __name__ == "__main__":
    main()
