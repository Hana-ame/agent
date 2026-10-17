"""Tag Matching & Embedding Similarity module for DAG Agent.

Evaluates whether existing `ContentItem` tags match an edge's `input_tags` via:
1. Exact tag equality (`1.0`)
2. Lexical / N-gram similarity
3. Vector embedding cosine similarity (`>= threshold`)

Ranks runnable `(edge, matched_items)` candidates by:
- `edge.layer` ascending (`0` is highest priority)
- `input.gen` ascending (`0` is highest priority, earlier generations processed first)
- `edge.count` ascending (less-executed edges break ties)
- `similarity` descending
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from dag_agent.llm import LLMClientProtocol, deterministic_semantic_embedding
from dag_agent.schema import ContentItem, EdgeSpec


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity in [0.0, 1.0] between two embedding vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def ngram_similarity(tag_a: str, tag_b: str) -> float:
    """Compute character 1-gram + 2-gram cosine similarity with substring boost."""
    a = re.sub(r"[\s_\-:]+", "", str(tag_a).lower().strip())
    b = re.sub(r"[\s_\-:]+", "", str(tag_b).lower().strip())
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    grams_a = list(a) + [a[i : i + 2] for i in range(len(a) - 1)]
    grams_b = list(b) + [b[i : i + 2] for i in range(len(b) - 1)]
    vocab = set(grams_a) | set(grams_b)
    dot = sum(grams_a.count(g) * grams_b.count(g) for g in vocab)
    na = math.sqrt(sum(grams_a.count(g) ** 2 for g in vocab))
    nb = math.sqrt(sum(grams_b.count(g) ** 2 for g in vocab))
    score = dot / (na * nb) if (na and nb) else 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if len(shorter) >= 2 and shorter in longer:
        score = max(score, 0.72 + 0.25 * (len(shorter) / len(longer)))
    return round(min(1.0, score), 4)


@dataclass(frozen=True)
class EdgeMatchCandidate:
    """A runnable candidate pairing an `EdgeSpec` with matched `ContentItem` indices."""

    edge_index: int
    edge: EdgeSpec
    content_indices: Tuple[int, ...]
    similarity: float
    input_gen: int  # Generation of the matched input(s)

    @property
    def sort_key(self) -> Tuple[int, int, int, float, int]:
        """Priority ordering:
        1. `edge.layer` ascending (0 is highest priority)
        2. `input_gen` ascending (smaller gen first)
        3. `edge.count` ascending (less-run edge first on tie)
        4. `-similarity` ascending (higher similarity first)
        5. `edge_index` ascending (definition order)
        """
        return (
            self.edge.layer,
            self.input_gen,
            self.edge.count,
            -round(self.similarity, 4),
            self.edge_index,
        )


class TagMatcher:
    """Computes tag similarity (exact + n-gram + embedding cosine) and selects the highest-priority runnable edge."""

    def __init__(
        self,
        llm_client: Optional[LLMClientProtocol] = None,
        threshold: float = 0.70,
        use_embeddings: bool = True,
        allow_multi_item_direct_match: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.threshold = float(threshold)
        self.use_embeddings = bool(use_embeddings)
        self.allow_multi_item_direct_match = bool(allow_multi_item_direct_match)
        self._embed_cache: Dict[str, List[float]] = {}

    def _get_embedding(self, tag: str) -> List[float]:
        key = tag.strip().lower()
        if key not in self._embed_cache:
            if self.llm_client is not None:
                self._embed_cache[key] = self.llm_client.embed(key)
            else:
                self._embed_cache[key] = deterministic_semantic_embedding(key)
        return self._embed_cache[key]

    def tag_similarity(self, req_tag: str, cand_tag: str) -> float:
        """Return similarity score in [0.0, 1.0] between required tag and candidate tag."""
        r = req_tag.strip().lower()
        c = cand_tag.strip().lower()
        if not r or not c:
            return 0.0
        if r == c:
            return 1.0
        lex_sim = ngram_similarity(r, c)
        if not self.use_embeddings:
            return lex_sim
        emb_sim = cosine_similarity(self._get_embedding(r), self._get_embedding(c))
        return round(max(lex_sim, emb_sim), 4)

    def item_matches_required_tags(
        self,
        required_tags: Sequence[str],
        item_tags: Sequence[str],
    ) -> Optional[float]:
        """Check if a single `ContentItem`'s tags satisfy all `required_tags` with similarity >= threshold.

        Requires injective matching (a single item tag cannot satisfy two distinct required tags).
        Returns average similarity score if all required tags are satisfied, or `None` otherwise.
        """
        if not required_tags or not item_tags or len(item_tags) < len(required_tags):
            return None
        used_item_indices: Set[int] = set()
        scores: List[float] = []
        for req in required_tags:
            best_score = -1.0
            best_idx = -1
            for idx, t in enumerate(item_tags):
                if idx in used_item_indices:
                    continue
                sim = self.tag_similarity(req, t)
                if sim > best_score:
                    best_score = sim
                    best_idx = idx
            if best_idx < 0 or best_score < self.threshold:
                return None
            used_item_indices.add(best_idx)
            scores.append(best_score)
        return sum(scores) / len(scores)

    def find_best_candidate(
        self,
        edges: Sequence[EdgeSpec],
        contents: Sequence[ContentItem],
        executed_signatures: Set[Tuple[int, Tuple[int, ...]]],
    ) -> Optional[EdgeMatchCandidate]:
        """Find all runnable `(edge, content_indices)` candidates above threshold and return the top-priority one."""
        candidates = self.find_all_candidates(edges, contents, executed_signatures)
        if not candidates:
            return None
        candidates.sort(key=lambda c: c.sort_key)
        return candidates[0]

    def find_all_candidates(
        self,
        edges: Sequence[EdgeSpec],
        contents: Sequence[ContentItem],
        executed_signatures: Set[Tuple[int, Tuple[int, ...]]],
    ) -> List[EdgeMatchCandidate]:
        """Enumerate all runnable candidates across edges and contents."""
        candidates: List[EdgeMatchCandidate] = []
        for e_idx, edge in enumerate(edges):
            if not edge.input_tags:
                continue
            # 1. Single content item satisfying all of `edge.input_tags`
            for c_idx, item in enumerate(contents):
                sig = (e_idx, (c_idx,))
                if sig in executed_signatures:
                    continue
                if set(edge.output_tags) == set(item.tags) and edge.input_tags != edge.output_tags:
                    continue
                sim = self.item_matches_required_tags(edge.input_tags, item.tags)
                if sim is not None and sim >= self.threshold:
                    candidates.append(
                        EdgeMatchCandidate(
                            edge_index=e_idx,
                            edge=edge,
                            content_indices=(c_idx,),
                            similarity=sim,
                            input_gen=item.gen,
                        )
                    )
            # 2. Optional direct multi-item match (when disabled, multi-item assembly goes through default_processor)
            if self.allow_multi_item_direct_match and len(edge.input_tags) > 1:
                per_tag_matches: List[List[Tuple[int, float]]] = []
                possible = True
                for req in edge.input_tags:
                    matches_for_req = []
                    for c_idx, item in enumerate(contents):
                        best = max((self.tag_similarity(req, t) for t in item.tags), default=0.0)
                        if best >= self.threshold:
                            matches_for_req.append((c_idx, best))
                    if not matches_for_req:
                        possible = False
                        break
                    per_tag_matches.append(matches_for_req)
                if possible:
                    for combo in itertools.product(*per_tag_matches):
                        idxs = tuple(dict.fromkeys(c[0] for c in combo))
                        if len(idxs) < 2:
                            continue
                        sig = (e_idx, idxs)
                        if sig in executed_signatures:
                            continue
                        avg_sim = sum(c[1] for c in combo) / len(combo)
                        max_gen = max(contents[i].gen for i in idxs)
                        candidates.append(
                            EdgeMatchCandidate(
                                edge_index=e_idx,
                                edge=edge,
                                content_indices=idxs,
                                similarity=avg_sim,
                                input_gen=max_gen,
                            )
                        )
        return candidates
