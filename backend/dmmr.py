"""Configurable, model-agnostic Dynamic MMR selection.

This module intentionally has no torch or transformer dependency. It can be
tested with synthetic vectors and used by both production inference and research
evaluation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import numpy as np


NormalizationMode = Literal["raw", "minmax", "zscore", "rank"]
RedundancyMode = Literal["max", "mean_top2"]


@dataclass(frozen=True)
class DMMRConfig:
    relevance_normalization: NormalizationMode = "rank"
    lambda_min: float = 0.60
    lambda_max: float = 0.85
    role_weight: float = 0.04
    cluster_weight: float = 0.04
    specificity_weight: float = 0.03
    future_penalty_weight: float = 0.04
    role_bonus_cap: float = 1.0
    cluster_bonus_cap: float = 1.0
    specificity_cap: float = 1.0
    future_penalty_cap: float = 1.0
    maximum_total_positive_bonus: float = 0.08
    relevance_floor: float = 0.25
    minimum_candidate_multiplier: int = 2
    minimum_relevance_retention: float = 0.85
    duplicate_threshold: float = 0.90
    redundancy_mode: RedundancyMode = "max"
    enable_relevance_guard: bool = True
    enable_fallback: bool = True
    enable_candidate_floor: bool = True
    use_dynamic_lambda: bool = True
    fixed_lambda: float = 0.70
    variance_scale: float = 2.0
    epsilon: float = 1e-8

    def validate(self) -> None:
        unit_fields = (
            "lambda_min", "lambda_max", "role_weight", "cluster_weight",
            "specificity_weight", "future_penalty_weight", "role_bonus_cap",
            "cluster_bonus_cap", "specificity_cap", "future_penalty_cap",
            "maximum_total_positive_bonus", "relevance_floor",
            "minimum_relevance_retention", "duplicate_threshold", "fixed_lambda",
        )
        for field in unit_fields:
            value = float(getattr(self, field))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} must be in [0, 1], got {value}")
        if self.lambda_min > self.lambda_max:
            raise ValueError("lambda_min cannot exceed lambda_max")
        if self.minimum_candidate_multiplier < 1:
            raise ValueError("minimum_candidate_multiplier must be at least 1")
        if self.variance_scale <= 0 or self.epsilon <= 0:
            raise ValueError("variance_scale and epsilon must be positive")
        if self.relevance_normalization not in {"raw", "minmax", "zscore", "rank"}:
            raise ValueError(f"Unsupported normalization: {self.relevance_normalization}")
        if self.redundancy_mode not in {"max", "mean_top2"}:
            raise ValueError(f"Unsupported redundancy mode: {self.redundancy_mode}")


@dataclass
class DMMRResult:
    selected_indices: list[int]
    normalized_relevance: np.ndarray
    lambda_value: float
    document_variance: float
    normalized_variance: float
    relevance_retention: float
    fallback_used: bool
    rerun_used: bool
    average_selected_redundancy: float
    candidate_pool: list[int]
    diagnostics: list[dict]
    config: dict


def _finite_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)


def normalize_relevance(
    values: Sequence[float] | np.ndarray,
    mode: NormalizationMode = "rank",
    epsilon: float = 1e-8,
) -> np.ndarray:
    scores = _finite_array(values)
    if scores.size == 0:
        return scores
    if mode == "raw":
        normalized = scores
    elif mode == "minmax":
        spread = float(scores.max() - scores.min())
        normalized = (
            np.full_like(scores, 0.5)
            if spread <= epsilon
            else (scores - scores.min()) / (spread + epsilon)
        )
    elif mode == "zscore":
        std = float(scores.std())
        zscores = np.zeros_like(scores) if std <= epsilon else (scores - scores.mean()) / (std + epsilon)
        zscores = np.clip(zscores, -60.0, 60.0)
        normalized = 1.0 / (1.0 + np.exp(-zscores))
    elif mode == "rank":
        if float(scores.max() - scores.min()) <= epsilon:
            normalized = np.full_like(scores, 0.5)
        else:
            order = np.argsort(-scores, kind="stable")
            ranks = np.empty(scores.size, dtype=np.float64)
            position = 0
            while position < scores.size:
                end = position + 1
                while end < scores.size and abs(scores[order[end]] - scores[order[position]]) <= epsilon:
                    end += 1
                average_rank = (position + end - 1) / 2.0
                ranks[order[position:end]] = average_rank
                position = end
            normalized = 1.0 - ranks / max(scores.size - 1, 1)
    else:
        raise ValueError(f"Unsupported relevance normalization: {mode}")
    return np.clip(np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)


def cosine_similarity_matrix(embeddings: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {vectors.shape}")
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe = vectors / np.maximum(norms, epsilon)
    cosine = safe @ safe.T
    return np.clip(np.nan_to_num(cosine, nan=0.0), -1.0, 1.0)


def normalize_cosine_similarity(cosine: np.ndarray | float) -> np.ndarray:
    values = np.asarray(cosine, dtype=np.float64)
    normalized = (np.nan_to_num(values, nan=-1.0, posinf=1.0, neginf=-1.0) + 1.0) / 2.0
    return np.clip(normalized, 0.0, 1.0)


def compute_document_lambda(
    embeddings: np.ndarray, config: DMMRConfig
) -> tuple[float, float, float]:
    config.validate()
    vectors = np.asarray(embeddings, dtype=np.float64)
    if len(vectors) <= 1:
        return config.lambda_min, 0.0, 0.0
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    centroid = vectors.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    vector_norms = np.linalg.norm(vectors, axis=1)
    denominator = np.maximum(vector_norms * max(centroid_norm, config.epsilon), config.epsilon)
    cosine_to_centroid = (vectors @ centroid) / denominator
    cosine_to_centroid = np.clip(np.nan_to_num(cosine_to_centroid, nan=0.0), -1.0, 1.0)
    variance_scores = 1.0 - cosine_to_centroid
    document_variance = float(np.clip(variance_scores.mean(), 0.0, 2.0))
    normalized_variance = float(np.clip(document_variance / config.variance_scale, 0.0, 1.0))
    lambda_value = config.lambda_min + normalized_variance * (config.lambda_max - config.lambda_min)
    return float(np.clip(lambda_value, config.lambda_min, config.lambda_max)), document_variance, normalized_variance


def filter_candidate_pool(relevance: np.ndarray, k: int, config: DMMRConfig) -> list[int]:
    n = len(relevance)
    stable_order = np.argsort(-relevance, kind="stable").tolist()
    if not config.enable_candidate_floor:
        return stable_order
    pool = [index for index in stable_order if relevance[index] >= config.relevance_floor]
    minimum = min(n, max(k, k * config.minimum_candidate_multiplier))
    for index in stable_order:
        if len(pool) >= minimum:
            break
        if index not in pool:
            pool.append(index)
    return pool


def _redundancy(similarities: np.ndarray, mode: RedundancyMode) -> float:
    if similarities.size == 0:
        return 0.0
    ordered = np.sort(similarities)[::-1]
    if mode == "max" or len(ordered) == 1:
        return float(ordered[0])
    return float(ordered[:2].mean())


def _average_selected_redundancy(selected: Sequence[int], similarity: np.ndarray) -> float:
    if len(selected) <= 1:
        return 0.0
    values = [similarity[a, b] for position, a in enumerate(selected) for b in selected[position + 1 :]]
    return float(np.mean(values)) if values else 0.0


def _run_selection(
    relevance: np.ndarray,
    raw_probabilities: np.ndarray,
    similarity: np.ndarray,
    raw_cosine: np.ndarray,
    k: int,
    candidate_pool: Sequence[int],
    lambda_value: float,
    config: DMMRConfig,
    role_scores: np.ndarray,
    cluster_scores: np.ndarray,
    specificity_scores: np.ndarray,
    future_scores: np.ndarray,
    role_sets: Sequence[set[str]] | None,
    cluster_ids: Sequence[int] | None,
    attempt: str,
) -> tuple[list[int], list[dict]]:
    selected: list[int] = []
    remaining = list(candidate_pool)
    diagnostics: list[dict] = []
    while remaining and len(selected) < k:
        iteration = len(selected) + 1
        scored: list[tuple[float, float, int, dict]] = []
        for index in remaining:
            similarities = similarity[index, selected] if selected else np.asarray([], dtype=np.float64)
            raw_similarities = raw_cosine[index, selected] if selected else np.asarray([], dtype=np.float64)
            redundancy = _redundancy(similarities, config.redundancy_mode)
            duplicate = bool(similarities.size and float(similarities.max()) > config.duplicate_threshold)
            if duplicate:
                redundancy = 1.0

            if role_sets is not None:
                covered_roles = set().union(*(role_sets[selected_index] for selected_index in selected)) if selected else set()
                candidate_roles = set(role_sets[index]) - {"GENERAL"}
                role_raw = len(candidate_roles - covered_roles) / max(len(candidate_roles), 1)
            else:
                role_raw = role_scores[index]
            if cluster_ids is not None:
                covered_clusters = {cluster_ids[selected_index] for selected_index in selected}
                cluster_raw = float(not selected or cluster_ids[index] not in covered_clusters)
            else:
                cluster_raw = cluster_scores[index]
            role = float(np.clip(role_raw, 0.0, config.role_bonus_cap))
            cluster = float(np.clip(cluster_raw, 0.0, config.cluster_bonus_cap))
            specificity = float(np.clip(specificity_scores[index], 0.0, config.specificity_cap))
            future = float(np.clip(future_scores[index], 0.0, config.future_penalty_cap))
            relevance_contribution = lambda_value * float(relevance[index])
            redundancy_penalty = (1.0 - lambda_value) * redundancy
            role_contribution = config.role_weight * role
            cluster_contribution = config.cluster_weight * cluster
            specificity_contribution = config.specificity_weight * specificity
            positive_total = role_contribution + cluster_contribution + specificity_contribution
            if positive_total > config.maximum_total_positive_bonus and positive_total > 0:
                scale = config.maximum_total_positive_bonus / positive_total
                role_contribution *= scale
                cluster_contribution *= scale
                specificity_contribution *= scale
                positive_total = config.maximum_total_positive_bonus
            future_contribution = config.future_penalty_weight * future
            final_score = (
                relevance_contribution
                - redundancy_penalty
                + positive_total
                - future_contribution
            )
            record = {
                "attempt": attempt,
                "selection_iteration": iteration,
                "sentence_index": index,
                "raw_probability": float(raw_probabilities[index]),
                "normalized_relevance": float(relevance[index]),
                "similarities_to_selected": [float(value) for value in similarities],
                "raw_cosine_similarities_to_selected": [float(value) for value in raw_similarities],
                "maximum_redundancy_similarity": float(similarities.max()) if similarities.size else 0.0,
                "redundancy": redundancy,
                "duplicate_penalty_applied": duplicate,
                "lambda_value": lambda_value,
                "relevance_contribution": relevance_contribution,
                "redundancy_penalty": redundancy_penalty,
                "role_coverage_bonus": role,
                "role_contribution": role_contribution,
                "cluster_coverage_bonus": cluster,
                "cluster_contribution": cluster_contribution,
                "specificity_bonus": specificity,
                "specificity_contribution": specificity_contribution,
                "future_condition_penalty": future,
                "future_penalty_contribution": future_contribution,
                "total_positive_bonus": positive_total,
                "final_dmmr_score": float(final_score),
                "selected": False,
            }
            scored.append((float(final_score), float(relevance[index]), -index, record))
        winner = max(scored, key=lambda item: (item[0], item[1], item[2]))
        winner_index = int(winner[3]["sentence_index"])
        for _, _, _, record in scored:
            record["selected"] = record["sentence_index"] == winner_index
            diagnostics.append(record)
        selected.append(winner_index)
        remaining.remove(winner_index)
    return selected, diagnostics


def select_dynamic_mmr(
    embeddings: np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    k: int,
    config: DMMRConfig | None = None,
    role_scores: Sequence[float] | None = None,
    cluster_scores: Sequence[float] | None = None,
    specificity_scores: Sequence[float] | None = None,
    future_scores: Sequence[float] | None = None,
    role_sets: Sequence[set[str]] | None = None,
    cluster_ids: Sequence[int] | None = None,
) -> DMMRResult:
    config = config or DMMRConfig()
    config.validate()
    raw = _finite_array(probabilities)
    vectors = np.asarray(embeddings, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) != len(raw):
        raise ValueError("Embeddings and probabilities must contain the same number of sentences")
    n = len(raw)
    if n == 0 or k <= 0:
        return DMMRResult([], np.asarray([]), config.lambda_min, 0.0, 0.0, 1.0, False, False, 0.0, [], [], asdict(config))
    k = min(int(k), n)
    relevance = normalize_relevance(raw, config.relevance_normalization, config.epsilon)
    cosine = cosine_similarity_matrix(vectors, config.epsilon)
    similarity = normalize_cosine_similarity(cosine)
    lambda_value, document_variance, normalized_variance = compute_document_lambda(vectors, config)
    if not config.use_dynamic_lambda:
        lambda_value = float(np.clip(config.fixed_lambda, config.lambda_min, config.lambda_max))
    pool = filter_candidate_pool(relevance, k, config)

    def component(values: Sequence[float] | None) -> np.ndarray:
        if values is None:
            return np.zeros(n, dtype=np.float64)
        array = _finite_array(values)
        if len(array) != n:
            raise ValueError("Every score component must match the number of sentences")
        return np.clip(array, 0.0, 1.0)

    role = component(role_scores)
    cluster = component(cluster_scores)
    specificity = component(specificity_scores)
    future = component(future_scores)
    if role_sets is not None and len(role_sets) != n:
        raise ValueError("role_sets must match the number of sentences")
    if cluster_ids is not None and len(cluster_ids) != n:
        raise ValueError("cluster_ids must match the number of sentences")
    selected, diagnostics = _run_selection(
        relevance, raw, similarity, cosine, k, pool, lambda_value, config,
        role, cluster, specificity, future, role_sets, cluster_ids, "initial",
    )
    classifier_topk = np.argsort(-relevance, kind="stable")[:k].tolist()
    topk_sum = float(relevance[classifier_topk].sum())
    selected_sum = float(relevance[selected].sum())
    retention = selected_sum / topk_sum if topk_sum > config.epsilon else 1.0
    rerun_used = False
    fallback_used = False

    if config.enable_relevance_guard and retention < config.minimum_relevance_retention:
        rerun_used = True
        conservative_lambda = config.lambda_max
        rerun_selected, rerun_diagnostics = _run_selection(
            relevance, raw, similarity, cosine, k, pool, conservative_lambda, config,
            role, cluster, specificity, future, role_sets, cluster_ids, "reduced_diversity_rerun",
        )
        diagnostics.extend(rerun_diagnostics)
        rerun_sum = float(relevance[rerun_selected].sum())
        rerun_retention = rerun_sum / topk_sum if topk_sum > config.epsilon else 1.0
        selected, retention, lambda_value = rerun_selected, rerun_retention, conservative_lambda
        if retention < config.minimum_relevance_retention and config.enable_fallback:
            selected = classifier_topk
            retention = 1.0
            fallback_used = True

    return DMMRResult(
        selected_indices=selected,
        normalized_relevance=relevance,
        lambda_value=lambda_value,
        document_variance=document_variance,
        normalized_variance=normalized_variance,
        relevance_retention=float(retention),
        fallback_used=fallback_used,
        rerun_used=rerun_used,
        average_selected_redundancy=_average_selected_redundancy(selected, similarity),
        candidate_pool=list(pool),
        diagnostics=diagnostics,
        config=asdict(config),
    )


def conservative_rerun_config(config: DMMRConfig) -> DMMRConfig:
    """Public helper useful for tests and experiment configuration."""
    return replace(config, fixed_lambda=config.lambda_max, use_dynamic_lambda=False)


def load_dmmr_config(path: str | Path, fallback: DMMRConfig | None = None) -> DMMRConfig:
    """Load either a plain config JSON object or parameter-search output."""
    config_path = Path(path)
    if not config_path.exists():
        return fallback or DMMRConfig()
    with config_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("config", payload)
    known_fields = set(asdict(DMMRConfig()))
    return DMMRConfig(**{key: value for key, value in values.items() if key in known_fields})
