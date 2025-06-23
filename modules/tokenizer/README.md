# Semantic ID Tokenization and Lookup Table

This module provides utilities for tokenizing items into semantic IDs and creating lookup tables that map from semantic IDs to content embeddings.

## Components

### SemanticIdTokenizer

The `SemanticIdTokenizer` class tokenizes batches of item features into semantic IDs using a Residual Quantized Variational Autoencoder (RQ-VAE). It provides methods for:

- Tokenizing items to semantic IDs
- Precomputing corpus IDs for a dataset
- Checking if a semantic ID prefix exists in the corpus

### SemanticIDLookupTable

The `SemanticIDLookupTable` class implements a lookup table that maps semantic IDs to their corresponding content embeddings. This allows for efficient retrieval of content embeddings given a semantic ID, which is useful for various downstream tasks such as intra-list diversity (ILD) calculation.

## Usage

### Creating and Building a Lookup Table

```python
from modules.tokenizer.lookup_table import SemanticIDLookupTable

# Initialize the lookup table with an RQ-VAE model
lookup_table = SemanticIDLookupTable(rqvae_model)

# Build the lookup table from a dataset
num_entries = lookup_table.build_lookup_table(item_dataset)
print(f"Lookup table built with {num_entries} entries")
```

### Looking Up Content Embeddings

```python
# Get embedding for a single semantic ID
embedding = lookup_table.lookup(semantic_id)

# Get embeddings for multiple semantic IDs
embeddings = lookup_table.batch_lookup(semantic_ids)
```

## Intra-List Diversity (ILD) Metric

The lookup table is particularly useful for calculating intra-list diversity (ILD), which measures how diverse a list of recommendations is based on their content embeddings.

ILD is implemented in the `IntraListDiversity` class in the `evaluate.metrics` module. It calculates the average pairwise cosine distance between the embeddings of items in a recommendation list.

### Calculating ILD

```python
from evaluate.metrics import IntraListDiversity

# Get embeddings for a list of semantic IDs
embeddings = [lookup_table.lookup(sem_id) for sem_id in semantic_ids]
embeddings_tensor = torch.stack([emb for emb in embeddings if emb is not None])

# Calculate ILD
ild = IntraListDiversity().calculate_ild(embeddings_tensor)
print(f"Intra-List Diversity: {ild:.4f}")
```

### Integration with TopKAccumulator

The lookup table can be used with the `TopKAccumulator` to calculate ILD for recommendation lists:

```python
from evaluate.metrics import TopKAccumulator

accumulator = TopKAccumulator(ks=[1, 5, 10])
accumulator.accumulate(actual_sem_ids, predicted_sem_ids, tokenizer=tokenizer, lookup_table=lookup_table)
metrics = accumulator.reduce()

# The metrics will include 'ild@k' for each k
print(f"ILD@5: {metrics['ild@5']:.4f}")
```

## Benefits of ILD Metric

- **Diverse Recommendations**: Higher ILD values indicate more diverse recommendations, which can improve user satisfaction by exposing them to a wider range of items.
- **Filter Bubble Prevention**: Helps prevent recommendation systems from creating filter bubbles by encouraging diversity.
- **Evaluation Metric**: Provides an additional metric for evaluating recommendation quality beyond accuracy and relevance.