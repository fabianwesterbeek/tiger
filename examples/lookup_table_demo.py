#!/usr/bin/env python
"""
Example script demonstrating how to use the SemanticIDLookupTable and IntraListDiversity metric.
"""

import sys
import os
import torch
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.processed import ItemData
from data.processed import SeqData
from modules.rqvae import RqVae
from modules.tokenizer.semids import SemanticIdTokenizer
from modules.tokenizer.lookup_table import SemanticIDLookupTable
from evaluate.metrics import IntraListDiversity, TopKAccumulator
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from data.schemas import SeqBatch


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading item dataset from {args.item_dataset}")
    item_dataset = ItemData(args.item_dataset)

    print(f"Loading sequence dataset from {args.seq_dataset}")
    seq_dataset = SeqData(args.seq_dataset)

    # Create tokenizer
    print("Initializing tokenizer")
    tokenizer = SemanticIdTokenizer(
        input_dim=args.input_dim,
        output_dim=args.embed_dim,
        hidden_dims=[args.embed_dim],
        codebook_size=args.codebook_size,
        n_layers=args.n_layers,
        n_cat_feats=args.n_cat_feats,
        rqvae_weights_path=args.rqvae_weights_path
    )

    # Move tokenizer to device
    tokenizer.to(device)

    # Precompute corpus IDs
    print("Precomputing corpus IDs")
    tokenizer.precompute_corpus_ids(item_dataset)

    # Create lookup table
    print("Creating lookup table")
    lookup_table = SemanticIDLookupTable(tokenizer.rq_vae)

    # Build lookup table
    print("Building lookup table")
    num_entries = lookup_table.build_lookup_table(item_dataset, batch_size=args.batch_size)
    print(f"Lookup table built with {num_entries} entries")

    # Demonstration of how to use the lookup table
    # Take a sample batch from the sequence dataset
    sampler = BatchSampler(
        SequentialSampler(range(min(args.demo_samples, len(seq_dataset)))),
        batch_size=args.batch_size,
        drop_last=False,
    )

    dataloader = DataLoader(
        seq_dataset,
        sampler=sampler,
        shuffle=False,
    )

    # Demonstrate using TopKAccumulator with ILD metric
    print("\nDemonstrating TopKAccumulator with ILD metric:")

    for batch in dataloader:
        # Move batch to device
        batch = SeqBatch(
            user_ids=batch.user_ids.to(device),
            ids=batch.ids.to(device),
            ids_fut=batch.ids_fut.to(device),
            seq_mask=batch.seq_mask.to(device),
            x=batch.x.to(device) if hasattr(batch, 'x') else None,
            x_brand_id=batch.x_brand_id.to(device) if hasattr(batch, 'x_brand_id') else None,
        )

        # Tokenize the batch
        tokenized = tokenizer(batch)

        # Get example semantic IDs
        actual_sem_ids = tokenized.sem_ids_fut

        # For demonstration, use the first few items as "predictions"
        # In a real scenario, these would come from your model
        top_k_preds = tokenized.sem_ids[:, :args.k_max]

        # Create accumulator
        accumulator = TopKAccumulator(ks=[1, 3, 5, 10][:min(4, args.k_max)])

        # Accumulate metrics with lookup table for ILD
        accumulator.accumulate(actual_sem_ids, top_k_preds, tokenizer=tokenizer, lookup_table=lookup_table)

        # Print metrics
        metrics = accumulator.reduce()
        print("\nMetrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

        # Demonstrate direct ILD calculation
        print("\nDirect ILD calculation example:")
        for i in range(min(3, top_k_preds.size(0))):
            sem_ids = top_k_preds[i, :5]
            embeddings = []

            print(f"\nSample {i+1} semantic IDs:")
            for j, sem_id in enumerate(sem_ids):
                embedding = lookup_table.lookup(sem_id)
                if embedding is not None:
                    embeddings.append(embedding)
                    print(f"  ID {j+1}: {tuple(sem_id.cpu().tolist())} -> Found embedding")
                else:
                    print(f"  ID {j+1}: {tuple(sem_id.cpu().tolist())} -> No embedding found")

            if len(embeddings) >= 2:
                embeddings_tensor = torch.stack(embeddings)
                ild_score = IntraListDiversity().calculate_ild(embeddings_tensor)
                #print(f"  ILD score: {ild_score:.4f}")
            else:
                print("  Not enough embeddings to calculate ILD")

        break  # Just demonstrate with one batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lookup table and ILD metric demo")
    parser.add_argument("--item_dataset", type=str, default="dataset/ml-1m-movie",
                        help="Path to item dataset")
    parser.add_argument("--seq_dataset", type=str, default="dataset/ml-1m",
                        help="Path to sequence dataset")
    parser.add_argument("--rqvae_weights_path", type=str, required=True,
                        help="Path to pretrained RQ-VAE weights")
    parser.add_argument("--input_dim", type=int, default=18,
                        help="Input dimension")
    parser.add_argument("--embed_dim", type=int, default=32,
                        help="Embedding dimension")
    parser.add_argument("--codebook_size", type=int, default=32,
                        help="Codebook size")
    parser.add_argument("--n_layers", type=int, default=3,
                        help="Number of layers")
    parser.add_argument("--n_cat_feats", type=int, default=18,
                        help="Number of categorical features")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--demo_samples", type=int, default=10,
                        help="Number of samples to use for demo")
    parser.add_argument("--k_max", type=int, default=10,
                        help="Maximum k for top-k metrics")

    args = parser.parse_args()
    main(args)
