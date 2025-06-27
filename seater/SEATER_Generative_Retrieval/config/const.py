class Books_Config(object):
    def __init__(self) -> None:
        
        # data file path
        self.datafile_par_path = './data/Books'
        # self.datafile_par_path = './data/Books'
        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'
        # self.itemID_2_attr = 'vocab/item_2_attr_mapping.npy'

        # for tree-based index
        self.tree_data_par_path = './data/Books/tree_data_SASREC'
        # self.tree_data_par_path = './data/Books/tree_data_SASREC'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/Books_SASREC_item_emb.npy'

        # dataset config
        self.users_ID_num = 603671
        self.item_ID_num = 367982 + 1 # zero for padding
        self.item_cate_num = 1600 + 1 # zero for padding

        # experiment const config
        self.reco_his_max_length = 20

class Yelp_Config(object):
    def __init__(self) -> None:
        
        # data file path
        self.datafile_par_path = './data/Yelp'
        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'

        # for tree-based index
        self.tree_data_par_path = './data/Yelp/tree_data_SASREC'
        # self.tree_data_par_path = './data/Yelp/tree_data_SASREC'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/Yelp_SASREC_item_emb.npy'

        #dataset config
        self.users_ID_num = 31668
        self.item_ID_num = 38048 + 1 # zero for padding

        # experiment const config
        self.reco_his_max_length = 20 

class MIND_Config(object):
    def __init__(self) -> None:
        
        # data file path
        self.datafile_par_path = '../../MIND/MIND'
        # self.datafile_par_path = './data/MIND'
        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'

        # for tree-based index
        self.tree_data_par_path = '../../MIND/MIND/tree_data_SASREC'
        # self.tree_data_par_path = './data/MIND/tree_data_SASREC'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'      
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'
    
        self.two_tower_item_emb = 'vocab/MIND_SASREC_item_emb.npy'

        #dataset config
        self.users_ID_num = 50000
        self.item_ID_num = 39865 + 1 # zero for padding

        # experiment const config
        self.reco_his_max_length = 20 

class Toys_Config(object):
    def __init__(self) -> None:

        # data file path
        self.datafile_par_path = './data/Toys'
        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'

        # for tree-based index
        self.tree_data_par_path = './data/Toys/tree_data_SASREC'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/Toys_SASREC_item_emb.npy'

        # dataset config (update with actual numbers if known)
        self.users_ID_num = 19412 #1342911 # 19,412 11,924
        self.item_ID_num = 11924 +1 #327698 + 1  # zero for padding
        self.item_cate_num = 1000 + 1  # optional

        # experiment const config
        self.reco_his_max_length = 20


class Beauty_Config(object):
    def __init__(self) -> None:

        # data file path
        self.datafile_par_path = './data/Beauty'
        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'

        # for tree-based index
        self.tree_data_par_path = './data/Beauty/tree_data_SASREC'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/Beauty_SASREC_item_emb.npy'

        # dataset config (update with actual numbers if known)
        self.users_ID_num = 22363 #1210271 
        self.item_ID_num = 12101+1 #249274 + 1
        self.item_cate_num = 800 + 1

        # experiment const config
        self.reco_his_max_length = 20


class Sports_Config(object):
    def __init__(self) -> None:

        # data file path
        self.datafile_par_path = './data/Sports'
        self.training_file = 'dataset/training.tsv'
        self.validation_file = 'dataset/validation.tsv'
        self.test_file = 'dataset/test.tsv'

        # for tree-based index
        self.tree_data_par_path = './data/Sports/tree_data_SASREC'
        self.tree_based_itemID_2_indexID = 'itemID_2_tree_indexID.npy'
        self.tree_based_prefix_tree = 'tree_node_allowed_next_tokens.npy'

        self.two_tower_item_emb = 'vocab/Sports_SASREC_item_emb.npy'

        # dataset config (update with actual numbers if known)
        self.users_ID_num = 35598 #1990521  # 35,598 18,357
        self.item_ID_num = 18357 #478898 + 1
        self.item_cate_num = 900 + 1

        # experiment const config
        self.reco_his_max_length = 20

