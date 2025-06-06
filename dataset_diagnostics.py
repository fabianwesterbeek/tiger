#!/usr/bin/env python
# dataset_diagnostics_improved.py - A comprehensive tool to diagnose dataset issues in RQ-VAE

import os
import argparse
import json
import torch
import numpy as np
from collections import defaultdict

from data.processed import RecDataset, ItemData, SeqData
from data.amazon import AmazonReviews
from data.utils import batch_to
from modules.tokenizer.semids import SemanticIdTokenizer


def print_section(title):
    """Helper function to print formatted section titles"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def analyze_raw_data_files(dataset_folder, split):
    """Analyze raw data files for a specific dataset split"""
    print_section(f"RAW DATA FILE ANALYSIS: {split}")

    raw_path = os.path.join(dataset_folder, "raw", split)

    if not os.path.exists(raw_path):
        print(f"ERROR: Raw data path does not exist: {raw_path}")
        return

    # Check sequential_data.txt
    seq_path = os.path.join(raw_path, "sequential_data.txt")
    if os.path.exists(seq_path):
        with open(seq_path, 'r') as f:
            lines = f.readlines()

        print(f"Sequential data file: {seq_path}")
        print(f"  - Number of sequences: {len(lines)}")

        # Analyze sequence lengths
        lengths = []
        min_id = float('inf')
        max_id = -float('inf')
        for i, line in enumerate(lines[:1000]):  # Sample first 1000 to avoid memory issues
            parsed = list(map(int, line.strip().split()))
            lengths.append(len(parsed))
            min_id = min(min_id, min(parsed))
            max_id = max(max_id, max(parsed))

        print(f"  - Sequence length stats (sample of 1000):")
        print(f"    - Min: {min(lengths)}")
        print(f"    - Max: {max(lengths)}")
        print(f"    - Mean: {sum(lengths)/len(lengths):.2f}")
        print(f"  - ID range (sample of 1000): {min_id} to {max_id}")
    else:
        print(f"WARNING: Sequential data file not found: {seq_path}")

    # Check datamaps.json
    datamaps_path = os.path.join(raw_path, "datamaps.json")
    if os.path.exists(datamaps_path):
        with open(datamaps_path, 'r') as f:
            datamaps = json.load(f)

        print(f"Datamaps file: {datamaps_path}")
        print(f"  - Number of items in item2id: {len(datamaps.get('item2id', {}))}")
        print(f"  - Number of users in user2id: {len(datamaps.get('user2id', {}))}")

        # Check for ID gaps in item2id
        item_ids = [int(v) for v in datamaps.get('item2id', {}).values()]
        if item_ids:
            min_id = min(item_ids)
            max_id = max(item_ids)
            missing_ids = set(range(min_id, max_id + 1)) - set(item_ids)

            print(f"  - Item ID range: {min_id} to {max_id}")
            print(f"  - Missing item IDs: {len(missing_ids)} out of {max_id - min_id + 1}")
            if len(missing_ids) < 10 and len(missing_ids) > 0:
                print(f"    - Missing IDs: {sorted(list(missing_ids))}")
    else:
        print(f"WARNING: Datamaps file not found: {datamaps_path}")

    # Check meta.json.gz
    meta_path = os.path.join(raw_path, "meta.json.gz")
    if os.path.exists(meta_path):
        import gzip

        def parse(path):
            g = gzip.open(path, 'r')
            for l in g:
                yield eval(l)

        print(f"Meta data file: {meta_path}")
        try:
            meta_count = sum(1 for _ in parse(meta_path))
            print(f"  - Number of items in meta data: {meta_count}")

            # Sample some meta data entries
            print("  - Sample meta entries:")
            for i, meta in enumerate(parse(meta_path)):
                if i >= 3:  # Just show first 3
                    break
                sanitized_meta = {k: str(v)[:50] + ('...' if len(str(v)) > 50 else '') for k, v in meta.items()}
                print(f"    - Entry {i+1}: {sanitized_meta}")

        except Exception as e:
            print(f"  - ERROR reading meta data: {str(e)}")
    else:
        print(f"WARNING: Meta data file not found: {meta_path}")


def analyze_amazon_dataset(dataset_folder, split, category="brand"):
    """Analyze Amazon dataset for a specific split"""
    print_section(f"AMAZON DATASET ANALYSIS: {split}")

    try:
        # Initialize dataset
        print(f"Initializing AmazonReviews dataset for split '{split}', category '{category}'")
        raw_dataset = AmazonReviews(root=dataset_folder, split=split, category=category)

        # Print basic dataset info
        print(f"Dataset loaded successfully.")
        print(f"Dataset source path: {raw_dataset.processed_paths[0]}")

        # Check brand mapping
        brand_mapping = raw_dataset.get_brand_mapping()
        print(f"Brand mapping count: {len(brand_mapping)}")
        print(f"First 5 brand mappings: {dict(list(brand_mapping.items())[:5])}")

        # Check train/test split
        history_data = raw_dataset.data[("user", "rated", "item")]["history"]
        for split_name, data in history_data.items():
            print(f"{split_name} split:")
            print(f"  - Number of users: {data['userId'].shape[0]}")
            print(f"  - Sample user ids: {data['userId'][:5].tolist()}")

            if split_name == "train":
                # For train split, calculate sequence length stats safely
                seq_lens = []
                for seq in data["itemId"]:
                    if hasattr(seq, "__len__"):  # Check if sequence has length
                        seq_lens.append(len(seq))

                if seq_lens:
                    print(f"  - Sequence lengths:")
                    print(f"    - Min: {min(seq_lens)}")
                    print(f"    - Max: {max(seq_lens)}")
                    print(f"    - Mean: {sum(seq_lens)/len(seq_lens):.2f}")
                else:
                    print("  - Couldn't analyze sequence lengths")

        # Check item features
        item_data = raw_dataset.data["item"]
        print(f"Item features:")
        print(f"  - Number of items: {item_data['x'].shape[0]}")
        print(f"  - Feature dimension: {item_data['x'].shape[1]}")
        print(f"  - Brand ID count: {len(np.unique(item_data['brand_id']))}")

        # Check for missing values
        if 'brand_id' in item_data:
            missing_brands = (item_data['brand_id'] == -1).sum()
            print(f"  - Items with missing brand: {missing_brands}")

        return raw_dataset

    except Exception as e:
        print(f"ERROR analyzing Amazon dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def analyze_processed_dataset(dataset_folder, split, category="brand"):
    """Analyze processed dataset classes (ItemData and SeqData)"""
    print_section(f"PROCESSED DATASET ANALYSIS: {split}")

    try:
        # Initialize ItemData
        print("Initializing ItemData...")
        item_dataset = ItemData(
            root=dataset_folder,
            dataset=RecDataset.AMAZON,
            split=split,
            category=category
        )

        print(f"ItemData loaded successfully.")
        print(f"  - Number of items: {len(item_dataset)}")

        # Sample items
        if len(item_dataset) > 0:
            sample_indices = min(3, len(item_dataset) - 1)
            print(f"  - Sample item data (index {sample_indices}):")
            sample_item = item_dataset[sample_indices]
            print(f"    - User IDs shape: {sample_item.user_ids.shape}")
            print(f"    - Item IDs shape: {sample_item.ids.shape}")
            print(f"    - Item features shape: {sample_item.x.shape}")
            print(f"    - Item brand ID shape: {sample_item.x_brand_id.shape}")

        # Initialize train SeqData
        print("\nInitializing Train SeqData (subsample=True)...")
        try:
            train_dataset = SeqData(
                root=dataset_folder,
                dataset=RecDataset.AMAZON,
                is_train=True,
                subsample=True,
                split=split,
                category=category
            )

            print(f"Train SeqData loaded successfully.")
            print(f"  - Number of sequences: {len(train_dataset)}")
            print(f"  - Max sequence length: {train_dataset.max_seq_len}")

            # Check if item_brand_id exists and has expected shape
            if hasattr(train_dataset, 'item_brand_id'):
                print(f"  - Item brand ID shape: {train_dataset.item_brand_id.shape}")
                print(f"  - Item data shape: {train_dataset.item_data.shape}")

                # Check for potential indexing issues
                item_count = train_dataset.item_data.shape[0]

                # Find maximum item ID in sequences safely
                max_id = -1
                for seq in train_dataset.sequence_data["itemId"]:
                    if isinstance(seq, list) and seq:
                        max_id = max(max_id, max(seq))

                print(f"  - Item count: {item_count}")
                print(f"  - Max item ID in sequences: {max_id}")

                if max_id >= item_count:
                    print(f"  - WARNING: Maximum item ID ({max_id}) is ≥ item count ({item_count})!")
                    print(f"    This will cause an index out of bounds error in training/evaluation.")

            # Sample a sequence
            if len(train_dataset) > 0:
                print(f"\n  - Sample train sequence (index 0):")
                try:
                    sample_seq = train_dataset[0]
                    print(f"    - User IDs: {sample_seq.user_ids}")
                    print(f"    - Item IDs: {sample_seq.ids}")
                    print(f"    - Item IDs min/max: {sample_seq.ids.min()}/{sample_seq.ids.max()}")
                    print(f"    - Item features shape: {sample_seq.x.shape}")
                    print(f"    - Future item ID: {sample_seq.ids_fut}")
                    print(f"    - Sequence mask shape: {sample_seq.seq_mask.shape}")
                    print(f"    - Sequence mask sum: {sample_seq.seq_mask.sum().item()} (non-padding tokens)")
                except Exception as e:
                    print(f"    - ERROR sampling train sequence: {str(e)}")
        except Exception as e:
            print(f"ERROR initializing train SeqData: {str(e)}")
            import traceback
            traceback.print_exc()

        # Initialize eval SeqData
        print("\nInitializing Eval SeqData (is_train=False)...")
        try:
            eval_dataset = SeqData(
                root=dataset_folder,
                dataset=RecDataset.AMAZON,
                is_train=False,
                subsample=False,
                split=split,
                category=category
            )

            print(f"Eval SeqData loaded successfully.")
            print(f"  - Number of sequences: {len(eval_dataset)}")
            print(f"  - Max sequence length: {eval_dataset.max_seq_len}")

            # Check if item_brand_id exists and has expected shape
            if hasattr(eval_dataset, 'item_brand_id'):
                print(f"  - Item brand ID shape: {eval_dataset.item_brand_id.shape}")
                print(f"  - Item data shape: {eval_dataset.item_data.shape}")

                # Check for potential indexing issues safely
                item_count = eval_dataset.item_data.shape[0]
                max_id_in_sequence = eval_dataset.sequence_data["itemId"].max().item() if hasattr(eval_dataset.sequence_data["itemId"], "max") else -1

                print(f"  - Item count: {item_count}")
                print(f"  - Max item ID in sequences: {max_id_in_sequence}")

                if max_id_in_sequence >= item_count:
                    print(f"  - WARNING: Maximum item ID ({max_id_in_sequence}) is ≥ item count ({item_count})!")
                    print(f"    This will cause an index out of bounds error during evaluation.")

            # Sample a sequence
            if len(eval_dataset) > 0:
                print(f"\n  - Sample eval sequence (index 0):")
                try:
                    sample_seq = eval_dataset[0]
                    print(f"    - User IDs: {sample_seq.user_ids}")
                    print(f"    - Item IDs: {sample_seq.ids}")
                    print(f"    - Item IDs min/max: {sample_seq.ids.min()}/{sample_seq.ids.max()}")
                    print(f"    - Item features shape: {sample_seq.x.shape}")
                    print(f"    - Future item ID: {sample_seq.ids_fut}")
                    print(f"    - Sequence mask shape: {sample_seq.seq_mask.shape}")
                    print(f"    - Sequence mask sum: {sample_seq.seq_mask.sum().item()} (non-padding tokens)")
                except Exception as e:
                    print(f"    - ERROR sampling eval sequence: {str(e)}")
        except Exception as e:
            print(f"ERROR initializing eval SeqData: {str(e)}")
            import traceback
            traceback.print_exc()

        return item_dataset, train_dataset, eval_dataset

    except Exception as e:
        print(f"ERROR analyzing processed dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def analyze_tokenizer_minimal(dataset_folder, split, category="brand"):
    """Analyze the tokenizer with minimal operations to avoid errors"""
    print_section(f"TOKENIZER ANALYSIS (MINIMAL): {split}")

    vae_input_dim = 768
    vae_hidden_dims = [512, 256, 128]
    vae_embed_dim = 32
    vae_codebook_size = 256
    vae_n_layers = 3
    vae_n_cat_feats = 0

    try:
        # Check if the pretrained model exists
        pretrained_path = f"out/rqvae/amazon/checkpoint_{split}_399999.pt"
        if not os.path.exists(pretrained_path):
            print(f"WARNING: Pretrained RQVAE weights don't exist at {pretrained_path}")
            print("Skipping tokenizer initialization")
            return None

        # Initialize tokenizer with same parameters as in train_decoder.py
        print("Initializing SemanticIdTokenizer...")
        tokenizer = SemanticIdTokenizer(
            input_dim=vae_input_dim,
            hidden_dims=vae_hidden_dims,
            output_dim=vae_embed_dim,
            codebook_size=vae_codebook_size,
            n_layers=vae_n_layers,
            n_cat_feats=vae_n_cat_feats,
            rqvae_weights_path=pretrained_path
        )

        print(f"Tokenizer initialized successfully.")
        print(f"  - Semantic IDs dimension: {tokenizer.sem_ids_dim}")

        return tokenizer

    except Exception as e:
        print(f"ERROR initializing tokenizer: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def safe_dataloading_check(train_dataset, eval_dataset):
    """Check if dataloaders can be created and data can be loaded safely"""
    print_section("DATALOADER SAFETY CHECK")

    from torch.utils.data import DataLoader

    if train_dataset is not None:
        print("Testing Train DataLoader:")
        try:
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
            print(f"  - Train DataLoader created successfully")

            # Try getting one batch
            batch = next(iter(train_loader))
            print(f"  - Successfully retrieved batch from train loader")
            print(f"  - Batch shapes:")
            print(f"    - User IDs: {batch.user_ids.shape}")
            print(f"    - Item IDs: {batch.ids.shape}")
            print(f"    - Future Item IDs: {batch.ids_fut.shape}")

            # Check for index errors (items that would cause out-of-bounds)
            item_count = train_dataset.item_data.shape[0]
            max_id = batch.ids.max().item()
            print(f"  - Max item ID in batch: {max_id}, Item count: {item_count}")

            if max_id >= item_count:
                print(f"  - ERROR: Batch contains item ID ({max_id}) that exceeds item count ({item_count})")
            else:
                print(f"  - Indexing check passed")

        except Exception as e:
            print(f"  - ERROR with train DataLoader: {str(e)}")
            import traceback
            traceback.print_exc()

    if eval_dataset is not None:
        print("\nTesting Eval DataLoader:")
        try:
            eval_loader = DataLoader(eval_dataset, batch_size=4, shuffle=True)
            print(f"  - Eval DataLoader created successfully")

            # Try getting one batch
            batch = next(iter(eval_loader))
            print(f"  - Successfully retrieved batch from eval loader")
            print(f"  - Batch shapes:")
            print(f"    - User IDs: {batch.user_ids.shape}")
            print(f"    - Item IDs: {batch.ids.shape}")
            print(f"    - Future Item IDs: {batch.ids_fut.shape}")

            # Check for index errors (items that would cause out-of-bounds)
            item_count = eval_dataset.item_data.shape[0]
            max_id = batch.ids.max().item()
            print(f"  - Max item ID in batch: {max_id}, Item count: {item_count}")

            if max_id >= item_count:
                print(f"  - ERROR: Batch contains item ID ({max_id}) that exceeds item count ({item_count})")
            else:
                print(f"  - Indexing check passed")

        except Exception as e:
            print(f"  - ERROR with eval DataLoader: {str(e)}")
            import traceback
            traceback.print_exc()


def analyze_tokenizer_with_corpus(split, item_dataset):
    """Analyze tokenizer and corpus computations in isolation"""
    print_section(f"TOKENIZER CORPUS ANALYSIS: {split}")

    if item_dataset is None:
        print("No item dataset available, skipping tokenizer corpus analysis")
        return

    # Initialize just the VAE part for analysis
    vae_input_dim = 768
    vae_hidden_dims = [512, 256, 128]
    vae_embed_dim = 32

    try:
        # Check data properties
        print("Item Dataset Analysis:")
        print(f"  - Number of items: {len(item_dataset)}")
        print(f"  - Feature dimension: {item_dataset.item_data.shape[1]}")

        # Sample a few items to check indices
        for i in [0, 1, len(item_dataset)-1]:
            try:
                item = item_dataset[i]
                print(f"  - Item {i}:")
                print(f"    - IDs: {item.ids}")
                print(f"    - Feature shape: {item.x.shape}")
            except Exception as e:
                print(f"  - Error accessing item {i}: {str(e)}")

    except Exception as e:
        print(f"ERROR during tokenizer corpus analysis: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="RQ-VAE Dataset Diagnostics Tool")
    parser.add_argument("--split", type=str, default="toys",
                        help="Dataset split to analyze (beauty, sports, toys)")
    parser.add_argument("--dataset_folder", type=str, default="dataset/amazon",
                        help="Path to the dataset folder")
    parser.add_argument("--category", type=str, default="brand",
                        help="Category to use for analysis")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(f" RQ-VAE DATASET DIAGNOSTICS: {args.split}")
    print("=" * 80)
    print(f"Dataset folder: {args.dataset_folder}")
    print(f"Split: {args.split}")
    print(f"Category: {args.category}")

    # Run the various analysis functions
    analyze_raw_data_files(args.dataset_folder, args.split)
    raw_dataset = analyze_amazon_dataset(args.dataset_folder, args.split, args.category)
    item_dataset, train_dataset, eval_dataset = analyze_processed_dataset(
        args.dataset_folder, args.split, args.category
    )
    tokenizer = analyze_tokenizer_minimal(args.dataset_folder, args.split, args.category)
    safe_dataloading_check(train_dataset, eval_dataset)
    analyze_tokenizer_with_corpus(args.split, item_dataset)


if __name__ == "__main__":
    main()
