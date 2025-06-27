from collections import defaultdict
from torch import Tensor
import torch
import torch._dynamo
import math
from einops import rearrange
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Global configuration flags
DISABLE_ILD = False  # Disable Intra-List Diversity computation
ENABLE_VERBOSE = False  # Control debug output verbosity


def print_verbose(*args, **kwargs):
    if ENABLE_VERBOSE:
        print(*args, **kwargs)


def compute_dcg(relevance: list) -> float:
    """Compute Discounted Cumulative Gain"""
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))


def compute_ndcg_for_semantic_ids(pred: Tensor, actual: Tensor, k: int) -> float:
    """Compute NDCG@k for one example of semantic ID tuples"""
    actual_tuple = tuple(actual.tolist())
    relevance = [1 if tuple(row.tolist()) == actual_tuple else 0 for row in pred[:k]]
    dcg = compute_dcg(relevance)
    idcg = compute_dcg(sorted(relevance, reverse=True))
    return dcg / idcg if idcg > 0 else 0.0


class GiniCoefficient:
    """Calculate Gini coefficient (0=perfect equality, 1=perfect inequality)"""

    def gini_coefficient(self, values):
        """Compute Gini coefficient of value array"""
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            return 0.0

        arr = np.sort(arr)
        n = arr.size
        cumvals = np.cumsum(arr)
        mu = arr.mean()
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        return gini

    def calculate_list_gini(self, articles, key="category"):
        """Compute Gini coefficient over frequency distribution of a key"""
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        return self.gini_coefficient(list(freqs.values()))


class IntraListDiversity:
    """Calculate intra-list diversity (ILD) using content embeddings"""

    @torch._dynamo.disable()
    def calculate_ild(self, embeddings):
        """Compute average pairwise cosine distance between embeddings"""
        if len(embeddings) <= 1:
            return 0.0

        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / (
            embeddings.norm(dim=1, keepdim=True) + 1e-8
        )
        similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())
        distances = 1.0 - similarities

        # Calculate average pairwise distance
        n = distances.size(0)
        distances.fill_diagonal_(0.0)
        total_distance = distances.sum() / 2.0
        num_pairs = (n * (n - 1)) / 2.0

        return (total_distance / num_pairs).item()


# class TopKAccumulator:
#     def __init__(self, ks=[1, 5, 10]):
#         self.ks = ks
#         self.reset()

#     def reset(self):
#         self.total = 0
#         self.metrics = defaultdict(float)

#     @torch._dynamo.disable()
#     def accumulate(self, actual: Tensor, top_k: Tensor, tokenizer=None, lookup_table=None) -> None:
#         if lookup_table is not None:
#             print(f"Lookup table: {lookup_table}")
#         B, D = actual.shape
#         pos_match = rearrange(actual, "b d -> b 1 d") == top_k
#         #print("Actual upto 5",actual[:5])
#         #print("topk upto 5", top_k[:5])
#         for i in range(D):
#             match_found, rank = pos_match[..., : i + 1].all(axis=-1).max(axis=-1)
#             matched_rank = rank[match_found]
#             for k in self.ks:
#                 self.metrics[f"h@{k}_slice_:{i+1}"] += len(
#                     matched_rank[matched_rank < k]
#                 )

#             match_found, rank = pos_match[..., i : i + 1].all(axis=-1).max(axis=-1)
#             matched_rank = rank[match_found]
#             for k in self.ks:
#                 self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[matched_rank < k])

#         B = actual.size(0)
#         for b in range(B):
#             gold_docs = actual[b]
#             pred_docs = top_k[b]
#             #print("Gold", gold_docs)
#             #print("pred_docs", pred_docs)
#             for k in self.ks:
#                 topk_pred = pred_docs[:k]
#                 # hits = sum(1 for doc in topk_pred if doc in gold_docs)
#                 hits = torch.any(torch.all(topk_pred == gold_docs, dim=1)).item() # fixed hit calculation
#                 self.metrics[f"h@{k}"] += float(hits > 0)
#                 self.metrics[f"ndcg@{k}"] += compute_ndcg_for_semantic_ids(
#                     pred_docs, gold_docs, k
#                 )
#                 # if the tokinzer is given then for each prediction find the catergoy and add it to the list and then caclulate the gini coefficient
#                 if tokenizer is not None:
#                     list_gini = []
#                     for pred in topk_pred:
#                         idx = str(pred.tolist()[:-1])
#                         category = tokenizer.map_to_category[idx]
#                         list_gini.append({"id": idx, "category": category})
#                     self.metrics[f"gini@{k}"] += GiniCoefficient().calculate_list_gini(
#                         list_gini, key="category"
#                     )

#                 # Calculate intra-list diversity if lookup table is provided and ILD is not disabled
#                 if lookup_table is not None and not DISABLE_ILD:
#                     # Get embeddings for the topk predictions
#                     embeddings = []
#                     #print_verbose(f"\nDEBUG: Starting lookup for batch {b}, k={k}, topk shape: {topk_pred.shape}")
#                     for i, pred in enumerate(topk_pred):
#                         #print_verbose(f"DEBUG: Processing pred[{i}]: shape={pred.shape}, dtype={pred.dtype}, values={pred.tolist()}")
#                         # Only use the first 3 elements of semantic ID for lookup
#                         semantic_id_prefix = pred[:3] if len(pred) >= 3 else pred
#                         embedding = lookup_table.lookup(semantic_id_prefix)
#                         if embedding is not None:
#                             embeddings.append(embedding)
#                             #print_verbose(f"DEBUG1: Found embedding for pred[{i}]: shape={embedding.shape}")
#                         else:
#                             print_verbose(f"DEBUG2: Embedding NOT found for pred[{i}]")

#                     # Calculate ILD if we have at least 2 embeddings
#                     if len(embeddings) >= 2:
#                         embeddings_tensor = torch.stack(embeddings)
#                         ild_score = IntraListDiversity().calculate_ild(embeddings_tensor)
#                         self.metrics[f"ild@{k}"] += ild_score
#                     else:
#                         # If we have fewer than 2 embeddings, ILD is 0
#                         self.metrics[f"ild@{k}"] += 0.0
#                 elif lookup_table is not None and DISABLE_ILD:
#                     # When ILD is disabled, just set the metric to 0
#                     self.metrics[f"ild@{k}"] += 0.0
#         self.total += B

class TopKAccumulator:
    """Accumulate and compute top-k metrics including hits, NDCG, Gini, and ILD"""

    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(float)

    @torch._dynamo.disable()
    def accumulate(
        self,
        actual: Tensor,
        top_k: Tensor,
        tokenizer=None,
        lookup_table=None,
    ):
        """
        • Any predicted semantic-id that cannot be found in
          `tokenizer.map_to_category` (or the lookup-table) is treated as an
          ordinary *miss* – it simply contributes 0-relevance and gets the
          pseudo-category "UNKNOWN".

        •  The function therefore never raises `KeyError`.
        """
        B, D = actual.shape
        pos_match = rearrange(actual, "b d -> b 1 d") == top_k        # (B,k,D)

        # ------------------------------------------------------------------ #
        # hit-rate & NDCG                                                    #
        # ------------------------------------------------------------------ #
        for i in range(D):
            match_found, rank = pos_match[..., : i + 1].all(-1).max(-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_:{i+1}"] += (matched_rank < k).sum().item()

            match_found, rank = pos_match[..., i : i + 1].all(-1).max(-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += (matched_rank < k).sum().item()

        for b in range(B):
            gold = actual[b]
            preds = top_k[b]
            for k in self.ks:
                topk_pred = preds[:k]

                # --- H@k ----------------------------------------------------
                hit = any((topk_pred == gold).all(1))
                self.metrics[f"h@{k}"] += float(hit)

                # --- NDCG@k -------------------------------------------------
                self.metrics[f"ndcg@{k}"] += compute_ndcg_for_semantic_ids(
                    preds, gold, k
                )

                # --- per-list statistics that need tokenizer / lookup -------#
                if tokenizer is not None:
                    # build frequency list for Gini
                    articles = []
                    for p in topk_pred:
                        key = str(p.tolist()[:-1])         # same convention
                        cat = tokenizer.map_to_category.get(key, "UNKNOWN")
                        articles.append({"id": key, "category": cat})
                    self.metrics[f"gini@{k}"] += GiniCoefficient() \
                                                  .calculate_list_gini(articles)

                # --- ILD ----------------------------------------------------
                if lookup_table is not None and not DISABLE_ILD:
                    embs = []
                    for p in topk_pred:
                        prefix = p[:3] if len(p) >= 3 else p
                        emb = lookup_table.lookup(prefix)
                        if emb is not None:
                            embs.append(emb)
                    if len(embs) >= 2:
                        ild = IntraListDiversity() \
                              .calculate_ild(torch.stack(embs))
                        self.metrics[f"ild@{k}"] += ild
                    else:
                        self.metrics[f"ild@{k}"] += 0.0

        self.total += B


    def reduce(self) -> dict:
        """Return averaged metrics over all accumulated batches"""
        return {k: v / self.total for k, v in self.metrics.items()}
