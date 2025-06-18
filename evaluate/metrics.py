from collections import defaultdict
from torch import Tensor
import torch
import torch._dynamo
import math
from einops import rearrange
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Flag to disable ILD computation for debugging
DISABLE_ILD = False



def compute_dcg(relevance: list) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))


def compute_ndcg_for_semantic_ids(pred: Tensor, actual: Tensor, k: int) -> float:
    """
    Compute NDCG@k for one example of semantic ID tuples.
    pred: [K, D] tensor — top-k predicted semantic IDs
    actual: [D] tensor — ground truth semantic ID
    """
    actual_tuple = tuple(actual.tolist())  # Convert to hashable tuple
    relevance = [1 if tuple(row.tolist()) == actual_tuple else 0 for row in pred[:k]]
    dcg = compute_dcg(relevance)
    idcg = compute_dcg(sorted(relevance, reverse=True))
    return dcg / idcg if idcg > 0 else 0.0


class GiniCoefficient:
    """
    A class to calculate the Gini coefficient, a measure of income inequality.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    """

    def gini_coefficient(self, values):
        """
        Compute the Gini coefficient of array of values.
        For a frequency vector, G = sum_i sum_j |x_i - x_j| / (2 * n^2 * mu)
        """
        arr = np.array(values, dtype=float)
        if arr.sum() == 0:
            return 0.0
        # sort and normalize
        arr = np.sort(arr)
        n = arr.size
        cumvals = np.cumsum(arr)
        mu = arr.mean()
        # the formula simplifies to:
        # G = (1 / (n * mu)) * ( sum_i (2*i - n - 1) * arr[i] )
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * arr)) / (n * n * mu)
        return gini

    def calculate_list_gini(self, articles, key="category"):
        """
        Given a list of article dicts and a key (e.g. 'category'), compute the
        Gini coefficient over the frequency distribution of that key.
        """
        # count frequencies
        freqs = {}
        for art in articles:
            val = art.get(key, None) or "UNKNOWN"
            freqs[val] = freqs.get(val, 0) + 1
        return self.gini_coefficient(list(freqs.values()))


@torch._dynamo.disable
class IntraListDiversity:
    """
    A class to calculate intra-list diversity (ILD) using content embeddings.
    ILD measures how diverse the items in a recommendation list are.
    Higher values indicate more diverse recommendations.
    """

    @torch._dynamo.disable
    def calculate_ild(self, embeddings):
        """
        Compute the intra-list diversity of embeddings using cosine distance.

        Args:
            embeddings: A tensor of shape [n, d] containing n embeddings of dimension d

        Returns:
            float: The intra-list diversity score (average pairwise cosine distance)
        """
        if len(embeddings) <= 1:
            return 0.0

        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)

        # Calculate pairwise cosine similarities
        similarities = torch.mm(normalized_embeddings, normalized_embeddings.t())

        # Convert similarities to distances (1 - similarity)
        distances = 1.0 - similarities

        # Zero out the diagonal (self-similarity)
        n = distances.size(0)
        distances.fill_diagonal_(0.0)

        # Calculate average pairwise distance (upper triangular part only)
        total_distance = distances.sum() / 2.0  # Divide by 2 to count each pair once
        num_pairs = (n * (n - 1)) / 2.0

        return (total_distance / num_pairs).item()


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(float)

    def accumulate(self, actual: Tensor, top_k: Tensor, tokenizer=None, lookup_table=None) -> None:
        if lookup_table is not None:
            print(f"Lookup table: {lookup_table}")
        B, D = actual.shape
        pos_match = rearrange(actual, "b d -> b 1 d") == top_k
        for i in range(D):
            match_found, rank = pos_match[..., : i + 1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_:{i+1}"] += len(
                    matched_rank[matched_rank < k]
                )

            match_found, rank = pos_match[..., i : i + 1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[matched_rank < k])

        B = actual.size(0)
        for b in range(B):
            gold_docs = actual[b]
            pred_docs = top_k[b]
            for k in self.ks:
                topk_pred = pred_docs[:k]
                # hits = sum(1 for doc in topk_pred if doc in gold_docs)
                hits = torch.any(torch.all(topk_pred == gold_docs, dim=1)).item() # fixed hit calculation
                self.metrics[f"h@{k}"] += float(hits > 0)
                self.metrics[f"ndcg@{k}"] += compute_ndcg_for_semantic_ids(
                    pred_docs, gold_docs, k
                )
                # if the tokinzer is given then for each prediction find the catergoy and add it to the list and then caclulate the gini coefficient
                if tokenizer is not None:
                    list_gini = []
                    for pred in topk_pred:
                        idx = str(pred.tolist()[:-1])
                        category = tokenizer.map_to_category[idx]
                        list_gini.append({"id": idx, "category": category})
                    self.metrics[f"gini@{k}"] += GiniCoefficient().calculate_list_gini(
                        list_gini, key="category"
                    )

                # Calculate intra-list diversity if lookup table is provided and ILD is not disabled
                if lookup_table is not None and not DISABLE_ILD:
                    # Get embeddings for the topk predictions
                    embeddings = []
                    for pred in topk_pred:
                        embedding = lookup_table.lookup(pred)
                        if embedding is not None:
                            embeddings.append(embedding)
                            print(f"3cz appending embedding: {embedding}")
                        else:
                            print(f"3cz embedding not found for prediction: {pred}")

                    # Calculate ILD if we have at least 2 embeddings
                    if len(embeddings) >= 2:
                        with torch._dynamo.disable():
                            embeddings_tensor = torch.stack(embeddings)
                            ild_score = IntraListDiversity().calculate_ild(embeddings_tensor)
                            self.metrics[f"ild@{k}"] += ild_score
                    else:
                        # If we have fewer than 2 embeddings, ILD is 0
                        self.metrics[f"ild@{k}"] += 0.0
                elif lookup_table is not None and DISABLE_ILD:
                    # When ILD is disabled, just set the metric to 0
                    self.metrics[f"ild@{k}"] += 0.0
        self.total += B

    def reduce(self) -> dict:
        return {k: v / self.total for k, v in self.metrics.items()}
