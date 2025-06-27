import numpy as np
from utils import data_utils

class GiniCoefficient:
    """
    A class to calculate the Gini coefficient, a measure of income inequality.
    The Gini coefficient ranges from 0 (perfect equality) to 1 (perfect inequality).
    """
    def __init__(self, name = "Beauty"):
        # Load asin2category.tsv (no headers)
        asin2cat_df = data_utils.load_tsv_file(
            f"data/{name}/dataset",
            "asin2category.tsv",
            sep="\t",
            header=None,  # No header
            engine="pyarrow"
        )
        asin2cat_df.columns = ['asin', 'category']  # Assign column names
        self.asin2category = dict(zip(asin2cat_df['asin'], asin2cat_df['category']))

        # Load asin2id.tsv (no headers)
        asin2idx_df = data_utils.load_tsv_file(
            f"data/{name}/dataset",
            "asin2idx.tsv",
            sep="\t",
            header=None,
            engine="pyarrow"
        )
        asin2idx_df.columns = ['asin', 'id']
        asin2idx = dict(zip(asin2idx_df['asin'], asin2idx_df['id']))
        self.idx2asin = {v: k for k, v in asin2idx.items()}
    
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
        index = np.arange(1, n+1)
        gini = (np.sum((2 * index - n - 1) * arr) ) / (n * n * mu)
        return gini

    def calculate_list_gini(self, article_ids, key="category"):
        """
        Given a list of article dicts and a key (e.g. 'category'), compute the
        Gini coefficient over the frequency distribution of that key.
        """
        # count frequencies
        
        freqs = {}
        for idx in article_ids:
            asin = self.idx2asin.get(idx, None)
            cat = self.asin2category.get(asin, "UNKNOWN")
            freqs[cat] = freqs.get(cat, 0) + 1

        return self.gini_coefficient(list(freqs.values()))
