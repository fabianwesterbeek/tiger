#!/usr/bin/env python
# model_diagnostics.py - Focused diagnostic tool for identifying model loading and indexing issues

import os
import sys
import torch
import argparse
import traceback

from data.processed import RecDataset, ItemData, SeqData
from modules.tokenizer.semids import SemanticIdTokenizer
from torch.utils.data import DataLoader


def check_file_exists(file_path):
    """Check if a file exists and print its details"""
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        print(f"✓ File exists: {file_path} ({size:.2f} MB)")
        return True
    else:
        print(f"✗ File does not exist: {file_path}")
        return False


def test_model_loading(model_path, split):
    """Test loading the model checkpoint"""
    print(f"\nTesting model loading for {split}...")

    if not check_file_exists(model_path):
        return False

    try:
        # Try loading the checkpoint
        checkpoint = torch.load(model_path, map_location="cpu")
        print(f"✓ Successfully loaded checkpoint: {model_path}")

        # Check what keys are in the checkpoint
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")

        # Check if the model has the expected structure
        if "model" in checkpoint and "state_dict" not in checkpoint:
            model_state = checkpoint["model"]
            print(f"  Model state keys count: {len(model_state.keys())}")
            print(f"  Sample model keys: {list(model_state.keys())[:5]}")
        elif "state_dict" in checkpoint:
            model_state = checkpoint["state_dict"]
            print(f"  Model state dict keys count: {len(model_state.keys())}")
            print(f"  Sample model keys: {list(model_state.keys())[:5]}")
        else:
            print(f"  ⚠️ Unexpected checkpoint structure. Keys: {list(checkpoint.keys())}")

        return True
    except Exception as e:
        print(f"✗ Error loading model checkpoint: {e}")
        traceback.print_exc()
        return False


def test_tokenizer_init(model_path, split):
    """Test initializing the tokenizer with the model checkpoint"""
    print(f"\nTesting tokenizer initialization for {split}...")

    vae_input_dim = 768
    vae_hidden_dims = [512, 256, 128]
    vae_embed_dim = 32
    vae_codebook_size = 256
    vae_n_layers = 3
    vae_n_cat_feats = 0

    try:
        # Initialize tokenizer with the model checkpoint
        tokenizer = SemanticIdTokenizer(
            input_dim=vae_input_dim,
            hidden_dims=vae_hidden_dims,
            output_dim=vae_embed_dim,
            codebook_size=vae_codebook_size,
            n_layers=vae_n_layers,
            n_cat_feats=vae_n_cat_feats,
            rqvae_weights_path=model_path
        )

        print(f"✓ Successfully initialized tokenizer with checkpoint: {model_path}")
        print(f"  Semantic IDs dimension: {tokenizer.sem_ids_dim}")
        print(f"  Has RQVAE: {hasattr(tokenizer, 'rq_vae')}")

        # Check if the RQ-VAE model was properly loaded
        if hasattr(tokenizer, 'rq_vae'):
            print(f"  RQ-VAE model device: {next(tokenizer.rq_vae.parameters()).device}")
            print(f"  RQ-VAE encoder layers: {len(tokenizer.rq_vae.encoder.layers) if hasattr(tokenizer.rq_vae, 'encoder') else 'N/A'}")

        # Test with a dummy input
        dummy_input = torch.randn(1, vae_input_dim)
        try:
            with torch.no_grad():
                sem_ids = tokenizer.get_semantic_ids(dummy_input)
            print(f"  ✓ Successfully generated semantic IDs from dummy input: {sem_ids.shape}")
        except Exception as e:
            print(f"  ✗ Error generating semantic IDs: {e}")
            traceback.print_exc()

        return tokenizer
    except Exception as e:
        print(f"✗ Error initializing tokenizer: {e}")
        traceback.print_exc()
        return None


def test_data_tokenization(split, dataset_folder, tokenizer):
    """Test data loading and tokenization"""
    print(f"\nTesting data loading and tokenization for {split}...")

    if tokenizer is None:
        print("✗ Cannot test tokenization without a valid tokenizer")
        return

    try:
        # Initialize ItemData
        item_dataset = ItemData(
            root=dataset_folder,
            dataset=RecDataset.AMAZON,
            split=split
        )
        print(f"✓ Successfully loaded ItemData: {len(item_dataset)} items")

        # Try to precompute corpus IDs
        try:
            tokenizer.precompute_corpus_ids(item_dataset)
            print(f"✓ Successfully precomputed corpus IDs")
            if hasattr(tokenizer, 'cached_ids'):
                print(f"  Cached IDs shape: {tokenizer.cached_ids.shape}")
        except Exception as e:
            print(f"✗ Error precomputing corpus IDs: {e}")
            traceback.print_exc()

        # Initialize SeqData
        try:
            train_dataset = SeqData(
                root=dataset_folder,
                dataset=RecDataset.AMAZON,
                is_train=True,
                subsample=True,
                split=split
            )
            print(f"✓ Successfully loaded Train SeqData: {len(train_dataset)} sequences")

            # Create a DataLoader
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            batch = next(iter(train_loader))
            print(f"✓ Successfully retrieved batch from train loader")
            print(f"  Batch item IDs shape: {batch.ids.shape}")

            # Try to tokenize the batch
            try:
                # Make sure everything is on CPU
                batch = batch._replace(**{
                    field: value.to('cpu')
                    for field, value in batch._asdict().items()
                    if isinstance(value, torch.Tensor)
                })

                tokenized_batch = tokenizer(batch)
                print(f"✓ Successfully tokenized batch")
                print(f"  Tokenized sem_ids shape: {tokenized_batch.sem_ids.shape}")

                # Examine the data more carefully for potential index errors
                max_id_in_batch = batch.ids.max().item()
                print(f"  Max item ID in batch: {max_id_in_batch}")
                print(f"  Item dataset length: {len(item_dataset)}")

                # Check explicitly if any IDs would cause index errors
                if max_id_in_batch >= len(item_dataset):
                    print(f"  ⚠️ Warning: Max ID {max_id_in_batch} exceeds item count {len(item_dataset)}")

                # Print out the cached_ids shape for debugging
                if hasattr(tokenizer, 'cached_ids'):
                    print(f"  Tokenizer cached_ids shape: {tokenizer.cached_ids.shape}")

                # Print future item IDs and their validity
                future_ids = batch.ids_fut
                print(f"  Future item IDs min/max: {future_ids.min().item()}/{future_ids.max().item()}")
                if future_ids.max().item() >= len(item_dataset):
                    print(f"  ⚠️ Warning: Max future ID {future_ids.max().item()} exceeds item count {len(item_dataset)}")

            except Exception as e:
                print(f"✗ Error tokenizing batch: {e}")
                traceback.print_exc()

                # Print exact error location and values
                print("\nDetailed error inspection:")

                # Try to identify exactly where in the tokenization it's failing
                if "index out of range" in str(e) or "index out of bounds" in str(e):
                    try:
                        # Look for the specific index in the error message
                        import re
                        match = re.search(r'index (\d+) is out of bounds for dimension \d+ with size (\d+)', str(e))
                        if match:
                            bad_index = int(match.group(1))
                            size = int(match.group(2))
                            print(f"  ⚠️ Index error: Trying to access index {bad_index} in array of size {size}")

                            # Check if this is coming from item IDs
                            if bad_index >= len(item_dataset):
                                print(f"  This appears to be an item ID issue - ID {bad_index} exceeds item count {len(item_dataset)}")

                                # Find where this ID appears in the batch
                                locations = torch.where(batch.ids == bad_index)
                                future_locations = torch.where(batch.ids_fut == bad_index)

                                if locations[0].size(0) > 0:
                                    print(f"  Bad ID {bad_index} found in batch.ids at positions: {locations}")
                                if future_locations[0].size(0) > 0:
                                    print(f"  Bad ID {bad_index} found in batch.ids_fut at positions: {future_locations}")
                    except Exception as inner_e:
                        print(f"  Error during error analysis: {inner_e}")

                # Check if tokenizer._tokenize_seq_batch_from_cached is the problem
                try:
                    print("\nTesting tokenization steps individually:")
                    # Flatten the IDs and check if any are invalid
                    ids = batch.ids.flatten()
                    print(f"  Flattened IDs shape: {ids.shape}")
                    print(f"  IDs min/max: {ids.min().item()}/{ids.max().item()}")

                    # Filter out padding (-1) values before indexing
                    valid_ids = ids[ids >= 0]
                    print(f"  Valid IDs shape: {valid_ids.shape}")
                    print(f"  Valid IDs min/max: {valid_ids.min().item()}/{valid_ids.max().item()}")

                    # Check if any valid IDs would cause index errors
                    if valid_ids.max().item() >= len(item_dataset):
                        print(f"  ⚠️ Warning: Max valid ID {valid_ids.max().item()} exceeds item count {len(item_dataset)}")

                        # Find where these problematic IDs are in the original batch
                        for problem_id in valid_ids[valid_ids >= len(item_dataset)].tolist():
                            locations = torch.where(batch.ids == problem_id)
                            print(f"  Problematic ID {problem_id} found at positions: {locations}")
                except Exception as inner_e:
                    print(f"  Error during ID analysis: {inner_e}")

        except Exception as e:
            print(f"✗ Error with train dataset or dataloader: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"✗ Error in data preparation: {e}")
        traceback.print_exc()


def test_tokenize_batch_directly(item_dataset, train_dataset, split):
    """Test direct tokenization of item features without using the tokenizer"""
    print(f"\nTesting direct feature access for {split}...")

    try:
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        batch = next(iter(train_loader))

        # Get item IDs from the batch
        item_ids = batch.ids.flatten()
        valid_ids = item_ids[item_ids >= 0]

        print(f"Valid item IDs: {valid_ids.tolist()}")

        # Try to directly access the item features for these IDs
        for idx in valid_ids[:5].tolist():  # Just try first 5 for brevity
            try:
                item = item_dataset.item_data[idx]
                print(f"✓ Successfully accessed item {idx}, shape: {item.shape}")
            except Exception as e:
                print(f"✗ Error accessing item {idx}: {e}")

        # Try the max ID specifically
        max_id = valid_ids.max().item()
        try:
            item = item_dataset.item_data[max_id]
            print(f"✓ Successfully accessed max item ID {max_id}, shape: {item.shape}")
        except Exception as e:
            print(f"✗ Error accessing max item ID {max_id}: {e}")

        # Try to access item_brand_id
        try:
            for idx in valid_ids[:5].tolist():
                brand_id = train_dataset.item_brand_id[idx]
                print(f"✓ Item {idx} has brand_id: {brand_id}")
        except Exception as e:
            print(f"✗ Error accessing brand_id: {e}")

    except Exception as e:
        print(f"✗ Error in direct tokenization test: {e}")
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="RQ-VAE Model Diagnostics Tool")
    parser.add_argument("--split", type=str, default="toys", help="Dataset split to analyze (beauty, sports, toys)")
    parser.add_argument("--dataset_folder", type=str, default="dataset/amazon", help="Path to the dataset folder")
    args = parser.parse_args()

    split = args.split
    dataset_folder = args.dataset_folder

    # Define paths for all models to make comparison easier
    model_paths = {
        "beauty": "out/rqvae/amazon/checkpoint_beauty_399999.pt",
        "sports": "out/rqvae/amazon/checkpoint_sports_399999.pt",
        "toys": "out/rqvae/amazon/checkpoint_toys_399999.pt"
    }

    print(f"\n{'='*80}\nRQ-VAE MODEL DIAGNOSTICS: {split}\n{'='*80}")
    print(f"Dataset folder: {dataset_folder}")
    print(f"Model path: {model_paths[split]}")

    # Check if the model file exists and try to load it
    if test_model_loading(model_paths[split], split):
        # Test tokenizer initialization if model loads successfully
        tokenizer = test_tokenizer_init(model_paths[split], split)

        # Test data loading and tokenization
        test_data_tokenization(split, dataset_folder, tokenizer)

    # Load datasets directly for additional tests
    try:
        print("\nLoading datasets for direct feature access tests...")
        item_dataset = ItemData(root=dataset_folder, dataset=RecDataset.AMAZON, split=split)
        train_dataset = SeqData(root=dataset_folder, dataset=RecDataset.AMAZON, is_train=True, subsample=True, split=split)
        test_tokenize_batch_directly(item_dataset, train_dataset, split)
    except Exception as e:
        print(f"✗ Error loading datasets for direct tests: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
