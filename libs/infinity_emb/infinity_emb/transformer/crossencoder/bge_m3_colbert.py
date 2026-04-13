# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil
# BGE-M3 ColBERT reranking support for GATC Health

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

from infinity_emb._optional_imports import CHECK_TORCH
from infinity_emb.args import EngineArgs
from infinity_emb.log_handler import logger
from infinity_emb.transformer.abstract import BaseCrossEncoder

if CHECK_TORCH.is_available:
    import torch

if TYPE_CHECKING:
    pass

__all__ = [
    "BGEColBERTReranker",
]


class BGEColBERTReranker(BaseCrossEncoder):
    """BGE-M3 ColBERT MaxSim reranker.

    Uses FlagEmbedding's BGEM3FlagModel to encode query and document
    separately into per-token ColBERT vectors, then scores via MaxSim:
      score = mean_i(max_j(cosine_sim(q_token_i, d_token_j)))

    This reranker uses the SAME model weights as the embedding engine
    (BAAI/bge-m3) — no second model needed.
    """

    capabilities = {"rerank"}

    def __init__(self, *, engine_args: EngineArgs):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding is required for BGE-M3 ColBERT reranking. "
                "Install with: pip install FlagEmbedding>=1.3.0"
            )

        self.engine_args = engine_args

        logger.info(
            f"Loading BGE-M3 ColBERT reranker: {engine_args.model_name_or_path}"
        )

        ls = engine_args._loading_strategy
        use_fp16 = ls is not None and ls.loading_dtype == torch.float16

        self.model = BGEM3FlagModel(
            engine_args.model_name_or_path,
            use_fp16=use_fp16,
        )

        # Copy tokenizer for thread-safe token counting
        self._infinity_tokenizer = copy.deepcopy(self.model.tokenizer)

        logger.info("BGE-M3 ColBERT reranker loaded successfully")

    def encode_pre(self, input_tuples: list[tuple[str, str]]):
        """Preprocessing: keep the (query, doc) string tuples as-is.

        FlagEmbedding handles its own tokenization internally.
        We return the raw tuples so encode_core can process them.
        """
        return input_tuples

    def encode_core(self, features: list[tuple[str, str]]):
        """Run ColBERT encoding and MaxSim scoring.

        For each (query, doc) pair:
        1. Encode query → per-token ColBERT vectors [Q, D]
        2. Encode doc → per-token ColBERT vectors [T, D]
        3. Compute MaxSim: mean over query tokens of max cosine sim with doc tokens
        """
        if not features:
            return np.array([], dtype=np.float32)

        # Collect unique queries and all documents
        queries = [f[0].strip() for f in features]
        docs = [f[1].strip() for f in features]

        # Encode all texts with ColBERT vectors
        all_texts = list(set(queries)) + docs
        query_set = list(set(queries))

        with torch.no_grad():
            # Encode unique queries
            query_output = self.model.encode(
                query_set,
                batch_size=min(len(query_set), 32),
                max_length=512,  # Queries are typically short
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )
            query_vecs_map = {
                q: query_output["colbert_vecs"][i]
                for i, q in enumerate(query_set)
            }

            # Encode documents
            doc_output = self.model.encode(
                docs,
                batch_size=min(len(docs), self.engine_args.batch_size),
                max_length=8192,  # Documents can be long
                return_dense=False,
                return_sparse=False,
                return_colbert_vecs=True,
            )

        # Compute MaxSim scores for each (query, doc) pair
        scores = []
        for i, (q, d) in enumerate(features):
            q_vecs = query_vecs_map[q.strip()]  # [num_query_tokens, dim]
            d_vecs = doc_output["colbert_vecs"][i]  # [num_doc_tokens, dim]

            # Convert to numpy if needed
            if hasattr(q_vecs, "numpy"):
                q_vecs = q_vecs.numpy()
            if hasattr(d_vecs, "numpy"):
                d_vecs = d_vecs.numpy()

            q_vecs = np.asarray(q_vecs, dtype=np.float32)
            d_vecs = np.asarray(d_vecs, dtype=np.float32)

            if q_vecs.size == 0 or d_vecs.size == 0:
                scores.append(0.0)
                continue

            # MaxSim: for each query token, find max cosine sim with any doc token
            # Normalize for cosine similarity
            q_norm = q_vecs / (np.linalg.norm(q_vecs, axis=1, keepdims=True) + 1e-8)
            d_norm = d_vecs / (np.linalg.norm(d_vecs, axis=1, keepdims=True) + 1e-8)

            # sim_matrix: [Q, T] = q_norm @ d_norm.T
            sim_matrix = np.dot(q_norm, d_norm.T)

            # Max over document tokens for each query token, then average
            max_sims = sim_matrix.max(axis=1)  # [Q]
            score = float(max_sims.mean())
            scores.append(score)

        return np.array(scores, dtype=np.float32)

    def encode_post(self, out_features) -> list[float]:
        """Return the ColBERT MaxSim scores as a flat list."""
        if isinstance(out_features, np.ndarray):
            return out_features.flatten().tolist()
        return list(out_features)

    def tokenize_lengths(self, sentences: list[str]) -> list[int]:
        """Get approximate token lengths for batch priority scheduling."""
        tks = self._infinity_tokenizer(
            sentences,
            add_special_tokens=False,
            return_attention_mask=False,
            return_length=True,
            truncation=True,
            max_length=8192,
        )
        if "length" in tks:
            return tks["length"]
        # Fallback: estimate from encodings
        return [len(s.split()) for s in sentences]
