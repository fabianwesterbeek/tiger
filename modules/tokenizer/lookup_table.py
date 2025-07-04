import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch._dynamo
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from data.utils import batch_to
from modules.utils import eval_mode
from modules.rqvae import RqVae


class SemanticIDLookupTable(nn.Module):
    """
    A lookup table that maps from semantic IDs to content embeddings.
    This allows efficient retrieval of content embeddings based on semantic IDs.
    """

    def __init__(self, rqvae_model):
        """
        Initialize the lookup table with an RQ-VAE model.

        Args:
            rqvae_model: An instance of the RqVae model used to generate embeddings
        """
        super().__init__()
        self.rqvae = rqvae_model
        self.id_to_embedding_map = {}
        self.device = rqvae_model.device

    @torch.no_grad()
    @eval_mode
    def build_lookup_table(self, dataset, batch_size=512):
        """
        Build lookup table from dataset items

        Args:
            dataset: A dataset containing items to process
            batch_size: Batch size for processing (default: 512)

        Returns:
            int: Number of entries in the lookup table
        """
        self.id_to_embedding_map = {}

        sampler = BatchSampler(
            SequentialSampler(range(len(dataset))),
            batch_size=batch_size,
            drop_last=False,
        )

        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            shuffle=False,
            collate_fn=lambda batch: batch[0],
        )

        for batch in dataloader:
            # Move data to device
            item = batch_to(batch, self.device)

            # Get content embedding directly from encoder
            item_tensor = item.x
            embedding = self.rqvae.encode(item_tensor)

            # Get semantic ID through quantization
            quantized = self.rqvae.get_semantic_ids(item_tensor)
            sem_ids = quantized.sem_ids

            # Store in lookup table (convert to tuple for hashability)
            for i in range(sem_ids.shape[0]):
                sem_id = sem_ids[i]
                sem_id_tuple = tuple(sem_id.detach().cpu().tolist())
                self.id_to_embedding_map[sem_id_tuple] = embedding[i].detach().cpu()

                # Debug: Print first 10 entries as they are added
                if len(self.id_to_embedding_map) <= 10:
                    print(f"DEBUG: Added to lookup table - Key: {sem_id_tuple}")

        # Debug: Print a summary of keys
        #print(f"DEBUG: Lookup table built with {len(self.id_to_embedding_map)} entries")
        #print(f"DEBUG: Sample of first 5 keys: {list(self.id_to_embedding_map.keys())[:5]}")
        key_lengths = [len(k) for k in list(self.id_to_embedding_map.keys())[:20]]
        #print(f"DEBUG: Key lengths in lookup table: {key_lengths}")

        return len(self.id_to_embedding_map)

    @torch._dynamo.disable
    def lookup(self, sem_id):
        """
        Get content embedding for a semantic ID

        Args:
            sem_id: A semantic ID (tensor or tuple)

        Returns:
            torch.Tensor: The content embedding for the semantic ID or None if not found
        """
        original_sem_id = sem_id
        if isinstance(sem_id, torch.Tensor):
            sem_id = tuple(sem_id.detach().cpu().tolist())

        # Debug: Print lookup attempts
        result = self.id_to_embedding_map.get(sem_id)
        #if result is None:
            #print(f"DEBUG: Lookup FAILED - Key: {sem_id}, Original tensor shape: {original_sem_id.shape if isinstance(original_sem_id, torch.Tensor) else 'N/A'}")
            # Try to find similar keys
            #if len(self.id_to_embedding_map) > 0:
                #sample_key = next(iter(self.id_to_embedding_map.keys()))
                #print(f"DEBUG: Sample key in table: {sample_key}, length: {len(sample_key)}")
                #print(f"DEBUG: Failed key length: {len(sem_id)}")
        #else:
            #print(f"DEBUG: Lookup SUCCESS - Key: {sem_id[:3]}..., Found embedding of shape: {result.shape}")

        return result

    @torch._dynamo.disable
    def batch_lookup(self, sem_ids):
        """
        Get content embeddings for multiple semantic IDs

        Args:
            sem_ids: A batch of semantic IDs

        Returns:
            torch.Tensor: A batch of content embeddings for the semantic IDs,
                          or None if no embeddings were found
        """
        results = []
        for sem_id in sem_ids:
            embedding = self.lookup(sem_id)
            if embedding is not None:
                results.append(embedding)

        if not results:
            return None

        return torch.stack(results)

    @torch._dynamo.disable
    def get_all_embeddings(self):
        """
        Get all embeddings in the lookup table

        Returns:
            tuple: (list of semantic IDs, tensor of all embeddings)
        """
        if not self.id_to_embedding_map:
            return [], None

        sem_ids = list(self.id_to_embedding_map.keys())
        embeddings = torch.stack([self.id_to_embedding_map[sem_id] for sem_id in sem_ids])

        return sem_ids, embeddings

    def __len__(self):
        """Return the number of entries in the lookup table"""
        return len(self.id_to_embedding_map)
