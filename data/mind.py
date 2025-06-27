import os
import json
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset
from data.preprocessing import PreprocessingMixin
from typing import Optional, Dict, Callable, List

class MINDPreprocessor(InMemoryDataset, PreprocessingMixin):
    def __init__(
    self,
    root: str,
    # split: str,  # 'beauty', 'sports', 'toys'
    transform: Optional[Callable] = None,
    pre_transform: Optional[Callable] = None,
    force_reload: bool = False,
    # category="brand",
    ) -> None:
        # self.split = split
        self.brand_mapping = {}  # Dictionary to store brand_id -> brand_name mapping
        # self.category = category
        self.split = 'mind'
        super(MINDPreprocessor, self).__init__(
            root, transform, pre_transform, force_reload
        )
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property
    def raw_file_names(self) -> List[str]:
        return ['mind_raw']

    @property
    def processed_file_names(self) -> str:
        return f"data_MIND.pt"

    def _remap_ids(self, values):
        return {val: idx for idx, val in enumerate(values)}

    def _format_sentences(self, df):
        return df.apply(
            lambda row: f"Title: {row['title']}; Category: {row['category']}; Subcategory: {row['subcategory']};",
            axis=1
        )

    def _build_history(self, behaviors_df, user2id, item2id, max_seq_len):
        history_dict = defaultdict(list)
        for _, row in behaviors_df.iterrows():
            uid = user2id.get(row["user_id"])
            if pd.isna(row["history"]):
                continue
            items = [item2id[i] for i in row["history"].split() if i in item2id]
            if items:
                history_dict[uid] = items[-max_seq_len:]
        return history_dict

    def process(self, max_seq_len=20):
        data = HeteroData()

        news_df = pd.read_csv(
            os.path.join(self.raw_dir, self.split, "news.tsv"),
            sep="\t",
            header=None,
            names=["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
        )
        behaviors_df = pd.read_csv(
            os.path.join(self.raw_dir, self.split, "behaviors.tsv"),
            sep="\t",
            header=None,
            names=["impression_id", "user_id", "time", "history", "impressions"]
        )

        self.user2id = self._remap_ids(behaviors_df["user_id"].unique())
        self.item2id = self._remap_ids(news_df["news_id"].unique())

        # history_dict = self._build_history(behaviors_df, self.user2id, self.item2id, max_seq_len)
        # data["user", "read", "item"].history = {
        #     "userId": torch.tensor(list(history_dict.keys())),
        #     "itemId": torch.tensor([history_dict[k] for k in history_dict], dtype=torch.long)
        # }
        data["user", "rated", "item"].history = {
            k: self._df_to_tensor_dict(v, ["itemId"]) for k, v in sequences.items()
        }
        sentences = self._format_sentences(news_df)
        item_emb = self._encode_text_feature(sentences)
        data["item"].x = item_emb
        data["item"].text = np.array(sentences)

        categories = news_df["category"].fillna("Unknown").unique()
        self.category_mapping = {cat: i for i, cat in enumerate(categories)}
        category_ids = news_df["category"].map(lambda x: self.category_mapping.get(x, -1))
        data["item"].category_id = np.array(category_ids)

        gen = torch.Generator()
        gen.manual_seed(42)
        data["item"].is_train = torch.rand(len(news_df), generator=gen) > 0.05

        self.save([data], os.path.join(self.processed_dir, f"data_{self.split}.pt"))

        with open(os.path.join(self.processed_dir, f"category_mapping_{self.split}.json"), "w") as f:
            json.dump(self.category_mapping, f)
        with open(os.path.join(self.processed_dir, f"user_item_id_mapping_{self.split}.json"), "w") as f:
            json.dump({"user2id": self.user2id, "item2id": self.item2id}, f)

    def process_sequence(df):
        #impression_id, user_id, time, history, impressions

    def get_impressions_from_str(impressions):
        impression_list = impressions.split()
        positive_impressions = []
        negative_impressions = []

        for impression in impression_list:
            news_id, click = impression.split('-')
            if click == '0':
                negative_impressions.append(news_id)
            else:
                positive_impressions.append(news_id)

        return positive_impressions, negative_impressions
