#!/usr/bin/env python
"""
Diagnostic script to identify issues in the Amazon datasets (beauty, sports, toys)
that could cause indexing errors during training and evaluation.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
import json
from collections import defaultdict
import gin

# Add the project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.amazon import AmazonReviews
from data.processed import ItemData, RecDataset, SeqData
from modules.tokenizer.semids import SemanticIdTokenizer
from data.utils import batch_to
from modules.utils import eval_mode


class DatasetDiagnostic:
    def __init__(self, dataset_folder="dataset/amazon"):
        self.dataset_folder = dataset_folder
        self.splits = ["beauty", "sports", "toys"]
        self.results = defaultdict(dict)
        self.category = "brand"

    def run_diagnostics(self):
        """Run all diagnostic tests on all dataset splits"""
        for split in self.splits:
            print(f"\n{'='*80}")
            print(f"Running diagnostics on {split} dataset")
            print(f"{'='*80}")

            # Test raw dataset structure
            self._test_raw_dataset(split)

            # Test processed dataset structure
            self._test_processed_dataset(split)

            # Test tokenizer with this dataset
            self._test_tokenizer(split)

            # Test sequence data generation
            self._test_seq_data(split)

        self._print_summary()

    def _test_raw_dataset(self, split):
        """Test properties of the raw dataset"""
        print(f"\nTesting raw dataset for {split}...")

        try:
            # Create the dataset instance
            raw_dataset = AmazonReviews(
                root=self.dataset_folder,
                split=split,
                category=self.category
            )

            # Check if brand exists in metadata and what its structure is
            if hasattr(raw_dataset, 'data') and 'item' in raw_dataset.data:
                if 'brand_id' in raw_dataset.data['item']:
                    brand_ids = raw_dataset.data['item']['brand_id']
                    self.results[split]['raw_brand_id_exists'] = True
                    self.results[split]['raw_brand_id_shape'] = brand_ids.shape
                    self.results[split]['raw_brand_id_min'] = np.min(brand_ids)
                    self.results[split]['raw_brand_id_max'] = np.max(brand_ids)
                    self.results[split]['raw_brand_id_unique'] = len(np.unique(brand_ids))
                    self.results[split]['raw_brand_id_missing'] = np.sum(brand_ids == -1)

                    print(f"  Brand ID exists: Shape={brand_ids.shape}")
                    print(f"  Brand ID range: [{np.min(brand_ids)}, {np.max(brand_ids)}]")
                    print(f"  Unique brand IDs: {len(np.unique(brand_ids))}")
                    print(f"  Missing brand IDs (-1): {np.sum(brand_ids == -1)}")
                else:
                    self.results[split]['raw_brand_id_exists'] = False
                    print("  WARNING: brand_id field doesn't exist in raw dataset!")
            else:
                print("  WARNING: Unable to access item data in raw dataset!")

            # Check if brand_mapping exists and its size
            if hasattr(raw_dataset, 'brand_mapping'):
                self.results[split]['brand_mapping_size'] = len(raw_dataset.brand_mapping)
                print(f"  Brand mapping size: {len(raw_dataset.brand_mapping)}")

                # Print a few examples from the brand mapping
                print("  Sample brand mappings:")
                sample_keys = list(raw_dataset.brand_mapping.keys())[:5]
                for k in sample_keys:
                    print(f"    {k}: {raw_dataset.brand_mapping[k]}")
            else:
                self.results[split]['brand_mapping_size'] = 0
                print("  WARNING: No brand_mapping found!")

        except Exception as e:
            self.results[split]['raw_dataset_error'] = str(e)
            print(f"  ERROR testing raw dataset: {e}")

    def _test_processed_dataset(self, split):
        """Test properties of the processed ItemData dataset"""
        print(f"\nTesting processed ItemData for {split}...")

        try:
            # Create item dataset
            item_dataset = ItemData(
                root=self.dataset_folder,
                dataset=RecDataset.AMAZON,
                split=split,
                category=self.category
            )

            self.results[split]['item_dataset_size'] = len(item_dataset)
            print(f"  ItemData size: {len(item_dataset)}")

            # Test a few samples and check their structure
            for i in range(min(3, len(item_dataset))):
                sample = item_dataset[i]
                if hasattr(sample, 'x_brand_id'):
                    print(f"  Sample {i} x_brand_id: {sample.x_brand_id.shape}, value={sample.x_brand_id}")
                    self.results[split][f'sample_{i}_x_brand_id'] = sample.x_brand_id.tolist()
                else:
                    print(f"  WARNING: Sample {i} has no x_brand_id attribute!")
                    self.results[split][f'sample_{i}_x_brand_id'] = "missing"

            # Check for outlier indices that might cause issues
            try:
                # Get the largest item ID
                all_ids = []
                for i in range(min(1000, len(item_dataset))):
                    sample = item_dataset[i]
                    all_ids.append(sample.ids.max().item())

                max_id = max(all_ids)
                self.results[split]['max_item_id'] = max_id
                print(f"  Max item ID: {max_id}")

                # Check if this would cause an out-of-bounds error
                if hasattr(item_dataset, 'item_brand_id'):
                    if max_id >= len(item_dataset.item_brand_id):
                        print(f"  WARNING: Max item ID {max_id} >= item_brand_id length {len(item_dataset.item_brand_id)}")
                        self.results[split]['item_id_out_of_bounds'] = True
                    else:
                        self.results[split]['item_id_out_of_bounds'] = False
            except Exception as e:
                print(f"  Error checking item IDs: {e}")

        except Exception as e:
            self.results[split]['processed_dataset_error'] = str(e)
            print(f"  ERROR testing processed dataset: {e}")

    def _test_tokenizer(self, split):
        """Test tokenizer with the dataset"""
        print(f"\nTesting tokenizer with {split} dataset...")

        try:
            # Load RQ-VAE checkpoint path from the config
            config_path = f'configs/rqvae_amazon_{split}.gin' if split != 'beauty' else 'configs/rqvae_amazon.gin'

            # Extract rqvae path from the config file
            rqvae_path = None
            try:
                with open(config_path, 'r') as f:
                    for line in f:
                        if 'train.pretrained_rqvae_path' in line:
                            rqvae_path = line.split('=')[1].strip().strip('"')
                            break
            except Exception:
                rqvae_path = None

            print(f"  Using RQ-VAE path: {rqvae_path if rqvae_path else 'None (will initialize randomly)'}")

            # Initialize the tokenizer
            tokenizer = SemanticIdTokenizer(
                input_dim=768,  # From configs
                hidden_dims=[512, 256, 128],
                output_dim=32,
                codebook_size=256,
                n_layers=3,
                n_cat_feats=0,
                rqvae_weights_path=rqvae_path
            )

            # Create item dataset
            item_dataset = ItemData(
                root=self.dataset_folder,
                dataset=RecDataset.AMAZON,
                split=split,
                category=self.category
            )

            print("  Testing precompute_corpus_ids...")

            # Test with a small subset to avoid memory issues
            subset_size = min(100, len(item_dataset))
            subset = torch.utils.data.Subset(item_dataset, range(subset_size))

            # Create a simplified test function that mimics precompute_corpus_ids but with better error handling
            def safe_test_corpus_ids():
                results = {"success": 0, "errors": []}

                # Process items individually to identify problematic ones
                for idx in range(subset_size):
                    try:
                        batch = item_dataset[idx]

                        # Check if batch has x_brand_id
                        has_brand_id = hasattr(batch, 'x_brand_id')
                        if not has_brand_id:
                            results["errors"].append(f"Item {idx} missing x_brand_id attribute")
                            continue

                        # Check if x_brand_id has valid values
                        x_brand_id = batch.x_brand_id
                        if torch.isnan(x_brand_id).any():
                            results["errors"].append(f"Item {idx} has NaN in x_brand_id")
                            continue

                        # Try to call item() on it, which is what causes issues in the tokenizer
                        try:
                            brand_id_value = x_brand_id.item()
                            results["success"] += 1
                        except Exception as e:
                            results["errors"].append(f"Item {idx} x_brand_id.item() failed: {e}")
                    except Exception as e:
                        results["errors"].append(f"Error processing item {idx}: {e}")

                return results

            test_results = safe_test_corpus_ids()
            self.results[split]['tokenizer_success'] = test_results["success"]
            self.results[split]['tokenizer_errors'] = len(test_results["errors"])

            print(f"  Successfully processed: {test_results['success']}/{subset_size} items")
            if test_results["errors"]:
                print(f"  Found {len(test_results['errors'])} errors:")
                for i, err in enumerate(test_results["errors"][:5]):  # Show first 5 errors
                    print(f"    - {err}")
                if len(test_results["errors"]) > 5:
                    print(f"    ... and {len(test_results['errors']) - 5} more.")

        except Exception as e:
            self.results[split]['tokenizer_error'] = str(e)
            print(f"  ERROR testing tokenizer: {e}")

    def _test_seq_data(self, split):
        """Test sequence data generation"""
        print(f"\nTesting SeqData for {split}...")

        try:
            train_dataset = SeqData(
                root=self.dataset_folder,
                dataset=RecDataset.AMAZON,
                is_train=True,
                subsample=False,
                split=split
            )

            self.results[split]['seq_train_size'] = len(train_dataset)
            print(f"  Train SeqData size: {len(train_dataset)}")

            # Test a few samples from the training set
            for i in range(min(3, len(train_dataset))):
                try:
                    sample = train_dataset[i]

                    # Check if item_ids contains any invalid indices
                    item_ids = sample.ids
                    valid_ids = (item_ids >= 0) & (item_ids < len(train_dataset.item_brand_id))
                    invalid_count = torch.sum(~valid_ids).item()

                    print(f"  Train sample {i}: item_ids shape={item_ids.shape}, invalid indices={invalid_count}")
                    if invalid_count > 0:
                        # This is a key indicator of the problem!
                        self.results[split][f'train_sample_{i}_invalid_ids'] = invalid_count
                        print(f"    WARNING: Sample {i} has {invalid_count} invalid item indices!")

                        # Get the problematic indices
                        problem_indices = item_ids[~valid_ids].tolist()
                        self.results[split][f'train_sample_{i}_problem_indices'] = problem_indices[:10]
                        print(f"    Problem indices (first 10): {problem_indices[:10]}")

                    # Check x_brand_id - this is where the issue probably happens
                    try:
                        x_brand_id = torch.Tensor(train_dataset.item_brand_id[item_ids])
                        self.results[split][f'train_sample_{i}_x_brand_id_shape'] = list(x_brand_id.shape)
                    except Exception as e:
                        self.results[split][f'train_sample_{i}_x_brand_id_error'] = str(e)
                        print(f"    ERROR creating x_brand_id: {e}")

                except Exception as e:
                    print(f"  ERROR getting train sample {i}: {e}")

            # Do the same for eval dataset
            eval_dataset = SeqData(
                root=self.dataset_folder,
                dataset=RecDataset.AMAZON,
                is_train=False,
                subsample=False,
                split=split
            )

            self.results[split]['seq_eval_size'] = len(eval_dataset)
            print(f"  Eval SeqData size: {len(eval_dataset)}")

            # Test a few samples from the eval set
            for i in range(min(3, len(eval_dataset))):
                try:
                    sample = eval_dataset[i]

                    # Check if item_ids contains any invalid indices
                    item_ids = sample.ids
                    valid_ids = (item_ids >= 0) & (item_ids < len(eval_dataset.item_brand_id))
                    invalid_count = torch.sum(~valid_ids).item()

                    print(f"  Eval sample {i}: item_ids shape={item_ids.shape}, invalid indices={invalid_count}")
                    if invalid_count > 0:
                        # Check if these are -1 padding values or actual invalid indices
                        invalid_non_padding = torch.sum((item_ids != -1) & ~valid_ids).item()
                        self.results[split][f'eval_sample_{i}_invalid_non_padding'] = invalid_non_padding
                        if invalid_non_padding > 0:
                            print(f"    WARNING: Sample {i} has {invalid_non_padding} invalid non-padding item indices!")

                            # Get the problematic indices
                            problem_indices = item_ids[(item_ids != -1) & ~valid_ids].tolist()
                            self.results[split][f'eval_sample_{i}_problem_indices'] = problem_indices[:10]
                            print(f"    Problem indices (first 10): {problem_indices[:10]}")

                except Exception as e:
                    print(f"  ERROR getting eval sample {i}: {e}")

        except Exception as e:
            self.results[split]['seq_data_error'] = str(e)
            print(f"  ERROR testing SeqData: {e}")

    def _print_summary(self):
        """Print a summary of all findings"""
        print("\n\n" + "="*40)
        print("DIAGNOSTIC SUMMARY")
        print("="*40)

        # Look for significant differences between datasets
        keys_to_compare = [
            'raw_brand_id_exists', 'raw_brand_id_shape', 'raw_brand_id_unique',
            'raw_brand_id_missing', 'brand_mapping_size', 'item_dataset_size',
            'tokenizer_errors', 'seq_train_size', 'seq_eval_size'
        ]

        print("\nKey differences between datasets:")
        for key in keys_to_compare:
            values = {split: self.results[split].get(key, "N/A") for split in self.splits}
            # Check if values differ across splits
            if len(set(str(v) for v in values.values())) > 1:
                print(f"\n{key}:")
                for split, value in values.items():
                    print(f"  {split}: {value}")

        # Check for key error indicators
        print("\nPotential issues:")
        for split in self.splits:
            issues = []

            # Check for raw dataset errors
            if 'raw_dataset_error' in self.results[split]:
                issues.append(f"Raw dataset error: {self.results[split]['raw_dataset_error']}")

            # Check for processed dataset errors
            if 'processed_dataset_error' in self.results[split]:
                issues.append(f"Processed dataset error: {self.results[split]['processed_dataset_error']}")

            # Check for tokenizer errors
            if 'tokenizer_error' in self.results[split]:
                issues.append(f"Tokenizer error: {self.results[split]['tokenizer_error']}")
            elif self.results[split].get('tokenizer_errors', 0) > 0:
                issues.append(f"Found {self.results[split]['tokenizer_errors']} tokenizer processing errors")

            # Check for invalid item IDs
            for i in range(3):
                if self.results[split].get(f'train_sample_{i}_invalid_ids', 0) > 0:
                    issues.append(f"Train sample {i} has {self.results[split][f'train_sample_{i}_invalid_ids']} invalid indices")
                if self.results[split].get(f'eval_sample_{i}_invalid_non_padding', 0) > 0:
                    issues.append(f"Eval sample {i} has {self.results[split][f'eval_sample_{i}_invalid_non_padding']} invalid non-padding indices")

            # Check for item_id out of bounds errors
            if self.results[split].get('item_id_out_of_bounds', False):
                issues.append(f"Item IDs exceed the size of item_brand_id array")

            if issues:
                print(f"\n{split} dataset issues:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print(f"\n{split} dataset appears healthy.")

        # Save results to file
        output_file = "dataset_diagnostic_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nDetailed results saved to {output_file}")


if __name__ == "__main__":
    diagnostic = DatasetDiagnostic(dataset_folder="dataset/amazon")
    diagnostic.run_diagnostics()
