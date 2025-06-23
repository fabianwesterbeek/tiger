from collections import defaultdict
import ast
import json
import polars as pl

from torch_geometric.data import HeteroData

class AmazonBooks():
    def __init__(self):
        self.split = "pets"
        self.books_path = './dataset/amazon/raw/'+self.split+'/'
        self.reviews_file = 'reviews.json'
        self.meta_file = 'meta.json'
        
    def generate_sequential_data(self):
        reviews_list = []

        print("----Reading start from "+self.books_path + self.reviews_file+"----")
        with open(self.books_path + self.reviews_file, 'r') as f:
            # reviews_list = []
            # for i, line in enumerate(f):
            #     reviews_list.append(ast.literal_eval(line))
            #     if i>50:
            #         break
            reviews_list = [ast.literal_eval(line) for i, line in enumerate(f)]

        print("----Open meta data file----"+ self.books_path + self.meta_file+"----")
        with open(self.books_path + self.meta_file, 'r') as f:
            meta_data = [ast.literal_eval(line) for line in f]
        print("----Meta data reading complete----")
        valid_ids = self.get_ids_with_prices(meta_data)


        print("----Review list reading complete----")
        grouped_reviews = defaultdict(list)

        for review in reviews_list:
            if review['asin'] not in valid_ids:
                continue
            review_info = {
                "reviewerID": review["reviewerID"],
                "unixReviewTime": review["unixReviewTime"],
                "asin": review["asin"]
            }
            grouped_reviews[review['reviewerID']].append(review_info)

        print("----Grouped Reviews----")
        for reviewer in grouped_reviews:
            grouped_reviews[reviewer].sort(key = lambda x: x['unixReviewTime'])

        print("----Sorted reviews----")
        reviewers = grouped_reviews.keys()

        item_mappings = self.create_asin_mapping()
        user_mappings = self.get_user_mapping(reviewers)

        print("----Obtained Item Mappings")

        user2id = user_mappings['user2id']
        item2id = item_mappings['item2id']

        sequences = []
        for reviewer in reviewers:
            sequence = []
            sequence.append(user2id[reviewer])

            reviewer_data = grouped_reviews[reviewer]

            for review in reviewer_data:
                item = review['asin']
                if item not in item2id:
                    continue
                item_id = item2id[item]
                sequence.append(item_id)
            
            if len(sequence) < 6:
                continue
            sequence_string = " ".join(map(str, sequence))
            sequences.append(sequence_string)

        print("----Generated Sequences. Saving file...----")
        with open(self.books_path + "sequential_data.txt", "w") as f:
            f.write("\n".join(sequences))

        merged_datamaps = item_mappings|user_mappings
        with open(self.books_path + "datamaps.json", "w") as f:
            json.dump(merged_datamaps, f)

        # with open(self.books_path + "../amazon/raw/books/datamaps.json", "a") as datamaps:
            # json.dump(user_mappings, datamaps)
        
    def create_asin_mapping(self, save = False):
        asins = set()
        with open(self.books_path + self.meta_file, 'r') as f:
            for i, line in enumerate(f):
                data = ast.literal_eval(line)
                asin = data['asin']
                asins.add(asin)
                # if i>50:
                #     break

        id_to_asin = {i+1: asin for i, asin in enumerate(asins)}
        asin_to_id = {asin: i for i, asin in id_to_asin.items()}

        data = {"id2item": id_to_asin, "item2id": asin_to_id}

        if save:
            with open(self.books_path + '../amazon/raw/'+self.split+'/datamaps.json', 'w') as f:
                json.dump(data, f)       
        return data
    
    def get_user_mapping(self, reviewers, save = False):

        id2user = {i+1: user for i, user in enumerate(reviewers)}
        user2id = {user: i for i, user in id2user.items()}

        data = {"id2user": id2user, "user2id": user2id}

        if save:
            with open(self.books_path + "../amazon/raw/"+self.split+"/datamaps.json", "a") as datamaps:
                json.dump(data, datamaps)
        return data

    def get_ids_with_prices(self, meta_data):

        ids = set()

        for item in meta_data:
            if 'price' in item.keys():
                ids.add(item['asin'])

        return ids


#'user2id'
# 'id2user'

# 'attribute2id' ####
# 'id2attribute' ####
# 'attributeid2num' ####
if __name__ == "__main__":
    books = AmazonBooks()
    books.generate_sequential_data()
    # books.create_asin_mapping()
        